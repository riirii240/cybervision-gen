"""Générateur MLP conditionnel du modèle A."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Generator(nn.Module):
    """Transforme ``(bruit, classe)`` en 63 caractéristiques dans ``[-1, 1]``."""

    def __init__(
        self,
        latent_dim: int = 128,
        n_classes: int = 11,
        n_features: int = 63,
        embedding_dim: int = 32,
        hidden_dims: Sequence[int] = (256, 512, 512),
    ) -> None:
        super().__init__()
        if min(latent_dim, n_classes, n_features, embedding_dim) <= 0:
            raise ValueError("Toutes les dimensions doivent être strictement positives.")
        if not hidden_dims or any(d <= 0 for d in hidden_dims):
            raise ValueError("hidden_dims doit contenir des dimensions positives.")
        self.latent_dim = int(latent_dim)
        self.n_classes = int(n_classes)
        self.n_features = int(n_features)
        self.embedding = nn.Embedding(n_classes, embedding_dim)

        layers: list[nn.Module] = []
        in_dim = latent_dim + embedding_dim
        for out_dim in hidden_dims:
            layers.extend((nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.LeakyReLU(0.2)))
            in_dim = out_dim
        layers.extend((nn.Linear(in_dim, n_features), nn.Tanh()))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(f"noise doit avoir la forme (N, {self.latent_dim}).")
        labels = labels.long().reshape(-1)
        if labels.shape[0] != noise.shape[0]:
            raise ValueError("noise et labels doivent avoir la même taille de batch.")
        return self.network(torch.cat((noise, self.embedding(labels)), dim=1))
