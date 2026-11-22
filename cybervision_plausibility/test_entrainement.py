"""Tests de l'étape 3 : le script d'entraînement est juste.

Le test principal prouve qu'avec λ = 0 notre boucle donne EXACTEMENT le même résultat
que la boucle de la Personne 2 (mêmes pertes, mêmes poids) : notre modèle A est bien
son modèle A. Données artificielles (aucun fichier de données nécessaire).

Lancer depuis la racine de la repo :
    python -m pytest -q cybervision_plausibility
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from entrainement import construire_modeles, entrainer
from penalites import PenalitesPlausibilite
from cybervision_gan.training.train import create_dataloader, set_seed, train_wgan_gp

CONFIG = {
    "project": {"seed": 42},
    "model": {"latent_dim": 16, "n_classes": 11, "n_features": 63, "embedding_dim": 8,
              "generator_hidden_dims": [32, 32], "critic_hidden_dims": [32, 32]},
    "training": {"epochs": 1, "batch_size": 32, "n_critic": 5, "gp_weight": 10.0,
                 "lr_g": 1e-4, "lr_c": 1e-4, "betas": [0.0, 0.9], "num_workers": 0,
                 "checkpoint_every": 0, "validation_batches": 2, "smoke_batches": 10},
    "device": "cpu",
}


def _preparer(lam: float):
    """Même graine -> mêmes données, mêmes poids initiaux, même ordre des batches."""
    cfg = copy.deepcopy(CONFIG)
    cfg["plausibilite"] = {"lambda": lam}
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(660, 63)).astype(np.float32)
    y = np.repeat(np.arange(11), 60)
    set_seed(42)
    train = create_dataloader(x, y, 32, balanced=True, seed=42)
    val = create_dataloader(x, y, 32, balanced=True, seed=43)
    generator, critic = construire_modeles(cfg)
    return cfg, generator, critic, train, val


def test_lambda_0_identique_au_modele_A_de_la_personne_2(tmp_path):
    cfg, g_p2, c_p2, train, val = _preparer(0.0)
    hist_p2 = train_wgan_gp(g_p2, c_p2, train, val, cfg, tmp_path / "p2", smoke_test=True)

    cfg, g_p3, c_p3, train, val = _preparer(0.0)
    hist_p3 = entrainer(g_p3, c_p3, train, val, cfg, tmp_path / "p3",
                        PenalitesPlausibilite.depuis_fichiers(), smoke=True)

    for cle in ("critic_loss", "generator_loss", "gradient_penalty"):
        assert hist_p3[0][cle] == hist_p2[0][cle], cle
    for p_p2, p_p3 in zip(g_p2.state_dict().values(), g_p3.state_dict().values()):
        assert torch.equal(p_p2, p_p3)


def test_lambda_positif_change_le_generateur_et_enregistre_la_penalite(tmp_path):
    cfg, g_a, c_a, train, val = _preparer(0.0)
    entrainer(g_a, c_a, train, val, cfg, tmp_path / "a", PenalitesPlausibilite.depuis_fichiers(), smoke=True)

    cfg, g_b, c_b, train, val = _preparer(1.0)
    hist_b = entrainer(g_b, c_b, train, val, cfg, tmp_path / "b",
                       PenalitesPlausibilite.depuis_fichiers(), smoke=True)

    assert "penalite_plausibilite" in hist_b[0] and "val_penalite_plausibilite" in hist_b[0]
    assert (tmp_path / "b" / "best.pt").exists() and (tmp_path / "b" / "metrics.json").exists()
    poids_differents = any(not torch.equal(a, b) for a, b in
                           zip(g_a.state_dict().values(), g_b.state_dict().values()))
    assert poids_differents


# ----------------------------------------------------------------------------
# Modèle C : critique à projection (étape 5)
# ----------------------------------------------------------------------------
from critique_projection import CritiqueProjection  # noqa: E402
from cybervision_gan.training.gradient_penalty import gradient_penalty  # noqa: E402


def test_critique_projection_forme_et_gradient_penalty():
    critique = CritiqueProjection(n_classes=11, n_features=63, hidden_dims=(32, 16))
    x_reel, x_faux = torch.randn(8, 63), torch.randn(8, 63)
    labels = torch.arange(8) % 11
    notes = critique(x_reel, labels)
    assert notes.shape == (8,)
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in critique.modules())
    gp = gradient_penalty(critique, x_reel, x_faux, labels)
    assert gp.ndim == 0 and torch.isfinite(gp) and gp.item() >= 0
    gp.backward()
    assert all(torch.isfinite(p.grad).all() for p in critique.parameters() if p.grad is not None)


def test_critique_projection_tient_compte_de_la_classe():
    """Même connexion, classe différente -> note différente : la classe ne peut pas être ignorée."""
    torch.manual_seed(0)
    critique = CritiqueProjection(n_classes=11, n_features=63, hidden_dims=(32, 16))
    with torch.no_grad():
        critique.signature.weight.normal_()          # signatures distinctes (comme après entraînement)
    x = torch.randn(1, 63).repeat(11, 1)
    notes = critique(x, torch.arange(11))
    assert notes.unique().numel() == 11


def test_modele_C_utilise_la_projection_et_s_entraine(tmp_path):
    cfg, _, _, train, val = _preparer(0.1)
    cfg["model"]["critique"] = "projection"
    generator, critic = construire_modeles(cfg)
    assert isinstance(critic, CritiqueProjection)
    hist = entrainer(generator, critic, train, val, cfg, tmp_path / "c",
                     PenalitesPlausibilite.depuis_fichiers(), smoke=True)
    assert np.isfinite(hist[0]["critic_loss"]) and np.isfinite(hist[0]["generator_loss"])
    assert (tmp_path / "c" / "best.pt").exists()
