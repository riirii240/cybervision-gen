"""Étape 5 (Personne 3) — Critique à projection pour le modèle C.

Différence avec le critique de la Personne 2 (models/critic.py) :
 - Personne 2 : la classe est COLLÉE à l'entrée -> le critique peut l'ignorer ;
 - ici (Miyato et Koyama, 2018) : le critique donne DEUX notes qu'il additionne :
       note = réalisme général  +  cohérence avec la classe
            = w · h(x)          +  e(y) · h(x)
   où h(x) est la représentation de la connexion et e(y) une « signature » apprise
   pour chaque classe. Le critique est ainsi obligé de vérifier la classe.

Le reste est identique au critique de la Personne 2 : mêmes couches cachées
[512, 512, 256], LeakyReLU(0.2), pas de sigmoid ni de BatchNorm (gradient penalty),
même initialisation. Aucune perte ni aucun réglage supplémentaire.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class CritiqueProjection(nn.Module):
    """Note Wasserstein d'une connexion (N, 63) pour sa classe (N,)."""

    def __init__(self, n_classes: int = 11, n_features: int = 63,
                 hidden_dims: Sequence[int] = (512, 512, 256)) -> None:
        super().__init__()
        if min(n_classes, n_features) <= 0 or not hidden_dims or any(d <= 0 for d in hidden_dims):
            raise ValueError("Toutes les dimensions doivent être strictement positives.")
        self.n_classes = int(n_classes)
        self.n_features = int(n_features)

        couches: list[nn.Module] = []
        entree = n_features
        for sortie in hidden_dims:
            couches.extend((nn.Linear(entree, sortie), nn.LeakyReLU(0.2)))
            entree = sortie
        self.representation = nn.Sequential(*couches)          # h(x)
        self.realisme = nn.Linear(entree, 1)                     # w · h(x)
        self.signature = nn.Embedding(n_classes, entree)         # e(y)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:              # même initialisation que la Personne 2
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
        h = self.representation(features)
        realisme = self.realisme(h).reshape(-1)
        coherence = (self.signature(labels) * h).sum(dim=1)
        return realisme + coherence
