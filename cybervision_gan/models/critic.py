"""Critique MLP conditionnel du modèle A (aucune sigmoid)."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Critic(nn.Module):
    """Attribue un score Wasserstein réel à un vecteur conditionné par sa classe."""

    def __init__(
        self,
        n_classes: int = 11,
        n_features: int = 63,
        embedding_dim: int = 32,
        hidden_dims: Sequence[int] = (512, 512, 256),
    ) -> None:
        super().__init__()
        if min(n_classes, n_features, embedding_dim) <= 0:
            raise ValueError("Toutes les dimensions doivent être strictement positives.")
        if not hidden_dims or any(d <= 0 for d in hidden_dims):
            raise ValueError("hidden_dims doit contenir des dimensions positives.")
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.embedding = nn.Embedding(n_classes, embedding_dim)

        layers: list[nn.Module] = []
        in_dim = n_features + embedding_dim
        for out_dim in hidden_dims:
            # Pas de BatchNorm : elle couplerait les exemples pendant la gradient penalty.
            layers.extend((nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.2)))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.n_features:
            raise ValueError(f"features doit avoir la forme (N, {self.n_features}).")
        labels = labels.long().reshape(-1)
        if labels.shape[0] != features.shape[0]:
            raise ValueError("features et labels doivent avoir la même taille de batch.")
        return self.network(torch.cat((features, self.embedding(labels)), dim=1)).reshape(-1)
