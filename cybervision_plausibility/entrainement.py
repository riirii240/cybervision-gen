"""Étape 3 (Personne 3) — Entraînement des modèles A et B avec le MÊME code.

    perte du générateur = perte WGAN de la Personne 2 + λ × pénalité de plausibilité

 - λ = 0  -> modèle A (référence, identique à la Personne 2) ;
 - λ > 0  -> modèle B (plausibilité).

Tout le reste est repris de la Personne 2 sans modification : générateur, critique,
gradient penalty, tirage équilibré des classes, réglages (model_a.yaml), sauvegardes
et critère de sélection de best.pt (distance de validation). La pénalité de
plausibilité est seulement ENREGISTRÉE sur la validation, elle ne sert pas à choisir.
Le test n'est jamais chargé.

Exemples (depuis la racine de la repo) :
    python cybervision_plausibility/entrainement.py --modele A --smoke     # test rapide (5 batches)
    python cybervision_plausibility/entrainement.py --modele B --lambda 1  # entraînement complet

Modèle C (étape 5) = modèle B + critique à projection (critique_projection.py) :
    python cybervision_plausibility/entrainement.py --modele C --lambda 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

DOSSIER = Path(__file__).resolve().parent
RACINE = DOSSIER.parent
for chemin in (RACINE, RACINE / "cybervision_data", DOSSIER):
    if str(chemin) not in sys.path:
        sys.path.insert(0, str(chemin))

# Code de la Personne 2 (réutilisé tel quel)
from cybervision_gan.models import Critic, Generator  # noqa: E402
from cybervision_gan.training.gradient_penalty import gradient_penalty  # noqa: E402
from cybervision_gan.training.train import (  # noqa: E402
    _ensure_finite_gradients, _save_checkpoint, create_dataloader, load_checkpoint,
    set_seed, validate_generator,
)
from penalites import ARTEFACTS, PenalitesPlausibilite  # noqa: E402  (étape 2)
from critique_projection import CritiqueProjection  # noqa: E402  (étape 5, modèle C)

CONFIG_A = RACINE / "cybervision_gan" / "configs" / "model_a.yaml"
DONNEES = RACINE / "cybervision_data" / "data" / "processed"


# ----------------------------------------------------------------------------
# Données et modèles : exactement comme le notebook de la Personne 2
# ----------------------------------------------------------------------------
def charger_donnees(cfg: dict, dossier_donnees: Path, seed: int):
    layout = json.loads((ARTEFACTS / "matrix_layout.json").read_text(encoding="utf-8"))
    label_to_index = json.loads((ARTEFACTS / "label_encoding.json").read_text(encoding="utf-8"))
    variables = layout["features"]
    loaders = []
    for i, nom in enumerate(("train_scaled.parquet", "val_scaled.parquet")):
        df = pd.read_parquet(dossier_donnees / nom, columns=variables + ["label"])
        x = df[variables].to_numpy(dtype=np.float32)
        y = df["label"].map(label_to_index).to_numpy(dtype=np.int64)
        loaders.append(create_dataloader(
            x, y, cfg["training"]["batch_size"], balanced=True, seed=seed + i,
            num_workers=cfg["training"]["num_workers"], drop_last=True))
    return loaders[0], loaders[1]


def construire_modeles(cfg: dict) -> tuple[Generator, Critic]:
    m = cfg["model"]
    generator = Generator(m["latent_dim"], m["n_classes"], m["n_features"],
                          m["embedding_dim"], m["generator_hidden_dims"])
    if m.get("critique") == "projection":                                         # [P3] modèle C
        critic = CritiqueProjection(m["n_classes"], m["n_features"], m["critic_hidden_dims"])
    else:
        critic = Critic(m["n_classes"], m["n_features"], m["embedding_dim"], m["critic_hidden_dims"])
    return generator, critic


@torch.no_grad()
def penalite_validation(generator, penalites, val_loader, latent_dim, device, max_batches=50) -> float:
    """Pénalité moyenne d'exemples générés pour les labels de la validation (information seulement)."""
    etait_en_train = generator.training
    generator.eval()
    valeurs = []
    for i, (_, labels) in enumerate(val_loader):
        if i >= max_batches:
            break
        labels = labels.to(device)
        faux = generator(torch.randn(len(labels), latent_dim, device=device), labels)
        valeurs.append(float(penalites(faux)[0]))
    generator.train(etait_en_train)
    return float(np.mean(valeurs))


# ----------------------------------------------------------------------------
# Boucle d'entraînement : copie de train_wgan_gp (Personne 2) + λ × pénalité
# Les lignes ajoutées sont marquées « [P3] ».
# ----------------------------------------------------------------------------
def entrainer(generator, critic, train_loader, val_loader, cfg, dossier, penalites,
              *, reprise=None, smoke=False):
    train_cfg = cfg["training"]
    lam = float(cfg["plausibilite"]["lambda"])                                    # [P3]
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    generator.to(device)
    critic.to(device)
    penalites.to(device)                                                          # [P3]
    opt_g = torch.optim.Adam(generator.parameters(), lr=train_cfg["lr_g"], betas=tuple(train_cfg["betas"]))
    opt_c = torch.optim.Adam(critic.parameters(), lr=train_cfg["lr_c"], betas=tuple(train_cfg["betas"]))
    dossier = Path(dossier)
    start_epoch, global_step, history, best_val = 0, 0, [], float("inf")

    if reprise is not None:
        state = load_checkpoint(reprise, generator, critic, opt_g, opt_c,
                                map_location=device, restore_rng=True)
        if state["config"].get("model") != cfg.get("model") or \
                state["config"].get("plausibilite") != cfg.get("plausibilite"):   # [P3]
            raise ValueError("Le checkpoint ne correspond pas à cette configuration (modèle ou λ).")
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        history = list(state.get("history", []))
        best_val = float(state.get("best_val", float("inf")))

    end_epoch = start_epoch + 1 if smoke else int(train_cfg["epochs"])
    if start_epoch >= end_epoch:
        print(f"Entraînement déjà terminé (époque {start_epoch - 1}).")
        return history
    n_critic = int(train_cfg["n_critic"])
    max_batches = int(train_cfg.get("smoke_batches", n_critic)) if smoke else None
    gp_weight = float(train_cfg["gp_weight"])
    latent_dim = int(cfg["model"]["latent_dim"])

    for epoch in range(start_epoch, end_epoch):
        debut = time.time()                                                      # [P3]
        generator.train()
        critic.train()
        sums = {"critic_loss": 0.0, "generator_loss": 0.0, "gradient_penalty": 0.0,
                "penalite_plausibilite": 0.0}                                     # [P3]
        n_batches = n_g_updates = 0
        for batch_index, (real, labels) in enumerate(train_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            real, labels = real.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch_size = real.shape[0]

            # --- Critique (identique à la Personne 2) ---
            opt_c.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = generator(torch.randn(batch_size, latent_dim, device=device), labels)
            real_score = critic(real, labels)
            fake_score = critic(fake, labels)
            gp = gradient_penalty(critic, real, fake, labels)
            critic_loss = fake_score.mean() - real_score.mean() + gp_weight * gp
            critic_loss.backward()
            _ensure_finite_gradients(critic, "le critique")
            opt_c.step()

            # --- Générateur : une mise à jour toutes les n_critic ---
            if (global_step + 1) % n_critic == 0:
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)
                generated = generator(torch.randn(batch_size, latent_dim, device=device), labels)
                generator_loss = -critic(generated, labels).mean()
                penalite, _ = penalites(generated)                                # [P3]
                if lam > 0:                                                       # [P3]
                    generator_loss = generator_loss + lam * penalite              # [P3]
                generator_loss.backward()
                _ensure_finite_gradients(generator, "le générateur")
                opt_g.step()
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)
                sums["generator_loss"] += float(generator_loss.detach().cpu())
                sums["penalite_plausibilite"] += float(penalite.detach().cpu())   # [P3]
                n_g_updates += 1

            sums["critic_loss"] += float(critic_loss.detach().cpu())
            sums["gradient_penalty"] += float(gp.detach().cpu())
            n_batches += 1
            global_step += 1

        if n_batches == 0 or n_g_updates == 0:
            raise RuntimeError("Aucun batch complet traité ; réduisez batch_size ou vérifiez les données.")
        n_val = int(train_cfg.get("validation_batches", 50))
        val_distance = validate_generator(generator, val_loader, latent_dim, device, max_batches=n_val)
        metrics = {
            "epoch": float(epoch),
            "critic_loss": sums["critic_loss"] / n_batches,
            "generator_loss": sums["generator_loss"] / n_g_updates,
            "gradient_penalty": sums["gradient_penalty"] / n_batches,
            "val_distribution_distance": val_distance,
            "penalite_plausibilite": sums["penalite_plausibilite"] / n_g_updates,                 # [P3]
            "val_penalite_plausibilite": penalite_validation(                                     # [P3]
                generator, penalites, val_loader, latent_dim, device, max_batches=n_val),
            "duree_epoque_s": time.time() - debut,                                                 # [P3]
        }
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        # Sélection de best.pt : même critère que la Personne 2 (distance de validation)
        if val_distance < best_val:
            best_val = val_distance
            _save_checkpoint(dossier / "best.pt", generator, critic, opt_g, opt_c,
                             epoch, global_step, cfg, history, best_val)
        _save_checkpoint(dossier / "latest.pt", generator, critic, opt_g, opt_c,
                         epoch, global_step, cfg, history, best_val)
        interval = int(train_cfg.get("checkpoint_every", 10))
        if interval > 0 and (epoch + 1) % interval == 0:
            _save_checkpoint(dossier / f"epoch_{epoch + 1:04d}.pt", generator, critic, opt_g, opt_c,
                             epoch, global_step, cfg, history, best_val)
        dossier.mkdir(parents=True, exist_ok=True)
        (dossier / "metrics.json").write_text(json.dumps(history, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    return history


# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne le modèle A (λ=0), B (λ>0) ou C (B + critique à projection).")
    parser.add_argument("--modele", choices=["A", "B", "C"], required=True)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0,
                        help="force de la pénalité pour les modèles B et C (ignorée pour A)")
    parser.add_argument("--donnees", type=Path, default=DONNEES,
                        help="dossier contenant train_scaled.parquet et val_scaled.parquet")
    parser.add_argument("--sortie", type=Path, default=None,
                        help="dossier des checkpoints (défaut : cybervision_plausibility/sorties/modele_X)")
    parser.add_argument("--epoques", type=int, default=None,
                        help="nombre d'époques (défaut : celui de model_a.yaml, soit 100)")   # [P3] essais courts
    parser.add_argument("--smoke", action="store_true", help="test rapide : 5 batches seulement")
    parser.add_argument("--reprendre", action="store_true",
                        help="reprendre depuis latest.pt du dossier de sortie s'il existe")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG_A.read_text(encoding="utf-8"))
    cfg["project"]["name"] = f"cybervision-modele-{args.modele}"
    cfg["plausibilite"] = {"lambda": 0.0 if args.modele == "A" else args.lam}
    if args.modele == "C":                                                        # [P3] modèle C
        cfg["model"]["critique"] = "projection"
    if args.epoques is not None:                                                  # [P3] essais courts
        cfg["training"]["epochs"] = args.epoques
    sortie = args.sortie or DOSSIER / "sorties" / f"modele_{args.modele}{'_smoke' if args.smoke else ''}"
    reprise = sortie / "latest.pt" if args.reprendre and (sortie / "latest.pt").exists() else None

    print(f"Modèle {args.modele} | λ = {cfg['plausibilite']['lambda']} | sortie : {sortie}")
    set_seed(cfg["project"]["seed"])
    train_loader, val_loader = charger_donnees(cfg, args.donnees, cfg["project"]["seed"])
    generator, critic = construire_modeles(cfg)
    penalites = PenalitesPlausibilite.depuis_fichiers()
    entrainer(generator, critic, train_loader, val_loader, cfg, sortie, penalites,
              reprise=reprise, smoke=args.smoke)
    print(f"Terminé. Checkpoints et metrics.json dans {sortie}")


if __name__ == "__main__":
    main()
