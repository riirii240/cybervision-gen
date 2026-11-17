"""Génération conditionnelle et conversion tabulaire -> matrice."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def generate_conditioned(
    generator: torch.nn.Module,
    class_index: int,
    n_samples: int,
    latent_dim: int,
    device: str | torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Retourne exactement ``(n_samples, n_features)`` valeurs brutes de ``tanh``."""
    if n_samples <= 0 or batch_size <= 0:
        raise ValueError("n_samples et batch_size doivent être strictement positifs.")
    if not 0 <= class_index < generator.n_classes:
        raise ValueError(f"class_index doit être dans [0, {generator.n_classes - 1}].")
    target_device = torch.device(device)
    generator.to(target_device)
    was_training = generator.training
    generator.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, n_samples, batch_size):
        size = min(batch_size, n_samples - start)
        noise = torch.randn(size, latent_dim, device=target_device)
        labels = torch.full((size,), class_index, dtype=torch.long, device=target_device)
        chunks.append(generator(noise, labels).cpu())
    generator.train(was_training)
    return torch.cat(chunks).numpy().astype(np.float32, copy=False)


def vectors_to_matrices(vectors: np.ndarray, layout: str | Path | dict) -> np.ndarray:
    """Applique l'ordre et le masque du layout ; le padding n'est jamais appris."""
    if not isinstance(layout, dict):
        layout = json.loads(Path(layout).read_text(encoding="utf-8"))
    x = np.asarray(vectors, dtype=np.float32)
    n_features, side = int(layout["n_features"]), int(layout["side"])
    if x.ndim != 2 or x.shape[1] != n_features:
        raise ValueError(f"vectors doit avoir la forme (N, {n_features}).")
    mask = np.asarray(layout["mask"], dtype=np.float32)
    if mask.shape != (side, side) or int(mask.sum()) != n_features:
        raise ValueError("Le masque du layout est incohérent avec n_features/side.")
    output = np.full((len(x), side, side), float(layout["pad_value"]), dtype=np.float32)
    for column, feature in enumerate(layout["features"]):
        row, col = layout["cells"][feature]
        output[:, row, col] = x[:, column]
    # Défense explicite : toute cellule masquée reprend toujours la valeur de padding.
    output[:, mask == 0] = float(layout["pad_value"])
    return output[:, None, :, :]
