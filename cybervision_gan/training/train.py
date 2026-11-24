"""Boucle d'entraînement reproductible du WGAN-GP conditionnel."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .gradient_penalty import gradient_penalty


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fixe les générateurs pseudo-aléatoires Python, NumPy, CPU et CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataloader(
    features: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    batch_size: int,
    *,
    balanced: bool,
    seed: int,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """Crée un chargeur, avec tirage uniforme entre classes si ``balanced=True``.

    Le poids d'une ligne est l'inverse de l'effectif de sa classe. Ce choix expose
    régulièrement le critique et le générateur aux classes rares, sans modifier les
    fichiers ni les partitions. La validation peut employer la même règle pour que
    chaque classe contribue équitablement au critère de checkpoint.
    """
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("features/labels doivent décrire un tableau non vide (N, F)/(N,).")
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(x, y)
    if balanced:
        counts = torch.bincount(y)
        if (counts == 0).any():
            raise ValueError("Les indices de classes doivent être contigus et toutes les classes présentes.")
        weights = counts.sum().double() / counts[y].double()
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator,
        drop_last=drop_last, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(
    path: Path,
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_c: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    history: list[dict[str, float]],
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_g": opt_g.state_dict(),
            "optimizer_c": opt_c.state_dict(),
            "config": config,
            "history": history,
            "best_val": best_val,
            "rng_state": _rng_state(),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    generator: torch.nn.Module,
    critic: torch.nn.Module | None = None,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_c: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Charge un checkpoint produit localement par ce module."""
    try:
        state = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        state = torch.load(Path(path), map_location=map_location)
    generator.load_state_dict(state["generator"])
    if critic is not None:
        critic.load_state_dict(state["critic"])
    if optimizer_g is not None:
        optimizer_g.load_state_dict(state["optimizer_g"])
    if optimizer_c is not None:
        optimizer_c.load_state_dict(state["optimizer_c"])
    if restore_rng:
        _restore_rng_state(state["rng_state"])
    return state


@torch.no_grad()
def validate_generator(
    generator: torch.nn.Module,
    val_loader: DataLoader,
    latent_dim: int,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Distance moyenne des moyennes/écarts-types, par batch validation.

    Elle ne touche jamais au test. Plus elle est faible, plus les deux premiers
    moments conditionnels synthétiques se rapprochent de la validation réelle.
    """
    was_training = generator.training
    generator.eval()
    distances: list[float] = []
    for index, (real, labels) in enumerate(val_loader):
        if max_batches is not None and index >= max_batches:
            break
        real, labels = real.to(device), labels.to(device)
        fake = generator(torch.randn(real.shape[0], latent_dim, device=device), labels)
        mean_gap = (real.mean(0) - fake.mean(0)).abs().mean()
        std_gap = (real.std(0, unbiased=False) - fake.std(0, unbiased=False)).abs().mean()
        distances.append(float((mean_gap + std_gap).cpu()))
    generator.train(was_training)
    if not distances:
        raise ValueError("Le DataLoader de validation n'a produit aucun batch.")
    return float(np.mean(distances))


def _ensure_finite_gradients(module: torch.nn.Module, name: str) -> None:
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError(f"Aucun gradient calculé pour {name}.")
    if not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError(f"Gradient NaN/infini détecté dans {name}.")


def train_wgan_gp(
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
    smoke_test: bool = False,
) -> list[dict[str, float]]:
    """Entraîne le modèle A et sélectionne ``best.pt`` uniquement sur validation."""
    cfg = config
    train_cfg = cfg["training"]
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    generator.to(device)
    critic.to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=train_cfg["lr_g"], betas=tuple(train_cfg["betas"]))
    opt_c = torch.optim.Adam(critic.parameters(), lr=train_cfg["lr_c"], betas=tuple(train_cfg["betas"]))
    checkpoint_dir = Path(checkpoint_dir)
    start_epoch, global_step, history, best_val = 0, 0, [], float("inf")

    if resume_from is not None:
        state = load_checkpoint(
            resume_from, generator, critic, opt_g, opt_c,
            map_location=device, restore_rng=True,
        )
        if state["config"].get("model") != cfg.get("model"):
            raise ValueError("Le checkpoint et la configuration ont des architectures différentes.")
        cfg = state["config"]  # configuration enregistrée restaurée
        train_cfg = cfg["training"]
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        history = list(state.get("history", []))
        best_val = float(state.get("best_val", float("inf")))

    end_epoch = start_epoch + 1 if smoke_test else int(train_cfg["epochs"])
    if start_epoch >= end_epoch:
        print(f"Entraînement déjà terminé au checkpoint (époque {start_epoch - 1}).")
        return history
    n_critic = int(train_cfg["n_critic"])
    max_batches = int(train_cfg.get("smoke_batches", n_critic)) if smoke_test else None
    gp_weight = float(train_cfg["gp_weight"])
    latent_dim = int(cfg["model"]["latent_dim"])

    for epoch in range(start_epoch, end_epoch):
        generator.train()
        critic.train()
        sums = {"critic_loss": 0.0, "generator_loss": 0.0, "gradient_penalty": 0.0}
        n_batches = n_g_updates = 0
        for batch_index, (real, labels) in enumerate(train_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            real, labels = real.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch_size = real.shape[0]

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

            # Une mise à jour de G seulement après n_critic mises à jour de C.
            if (global_step + 1) % n_critic == 0:
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)
                generated = generator(torch.randn(batch_size, latent_dim, device=device), labels)
                generator_loss = -critic(generated, labels).mean()  # aucun detach ici
                generator_loss.backward()
                _ensure_finite_gradients(generator, "le générateur")
                opt_g.step()
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)
                sums["generator_loss"] += float(generator_loss.detach().cpu())
                n_g_updates += 1

            sums["critic_loss"] += float(critic_loss.detach().cpu())
            sums["gradient_penalty"] += float(gp.detach().cpu())
            n_batches += 1
            global_step += 1

        if n_batches == 0 or n_g_updates == 0:
            raise RuntimeError("Aucun batch complet traité ; réduisez batch_size ou vérifiez les données.")
        val_distance = validate_generator(
            generator, val_loader, latent_dim, device,
            max_batches=int(train_cfg.get("validation_batches", 50)),
        )
        metrics = {
            "epoch": float(epoch),
            "critic_loss": sums["critic_loss"] / n_batches,
            "generator_loss": sums["generator_loss"] / n_g_updates,
            "gradient_penalty": sums["gradient_penalty"] / n_batches,
            "val_distribution_distance": val_distance,
        }
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        if val_distance < best_val:
            best_val = val_distance
            _save_checkpoint(
                checkpoint_dir / "best.pt", generator, critic, opt_g, opt_c,
                epoch, global_step, cfg, history, best_val,
            )
        _save_checkpoint(
            checkpoint_dir / "latest.pt", generator, critic, opt_g, opt_c,
            epoch, global_step, cfg, history, best_val,
        )
        interval = int(train_cfg.get("checkpoint_every", 10))
        if interval > 0 and (epoch + 1) % interval == 0:
            _save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt", generator, critic, opt_g, opt_c,
                epoch, global_step, cfg, history, best_val,
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "metrics.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return history
