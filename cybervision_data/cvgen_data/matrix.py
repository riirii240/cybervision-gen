"""Encodage vecteur (≤ 64 variables) <-> matrice 8x8 avec masque de remplissage.

Rappel (fiche 11) : la mise en matrice ne crée AUCUNE relation spatiale physique.
L'ordre des cellules suit des familles sémantiques (flow, fwd, bwd, IAT, drapeaux…)
uniquement pour rendre les visualisations lisibles. L'évaluation reste tabulaire.
"""
from __future__ import annotations

import json

import numpy as np


class MatrixLayout:
    def __init__(self, features: list[str], side: int = 8, pad_value: float = 0.0):
        if len(features) > side * side:
            raise ValueError(f"{len(features)} variables > {side*side} cellules")
        self.features, self.side, self.pad_value = list(features), side, float(pad_value)
        self.n = len(features)
        m = np.zeros(side * side, dtype=np.float32)
        m[: self.n] = 1.0
        self.mask = m.reshape(side, side)       # 1 = variable réelle, 0 = remplissage

    def to_matrix(self, X) -> np.ndarray:
        """(N, n) -> (N, 1, 8, 8) float32, cases de remplissage = pad_value."""
        X = np.asarray(X, dtype=np.float32)
        out = np.full((X.shape[0], self.side * self.side), self.pad_value, dtype=np.float32)
        out[:, : self.n] = X
        return out.reshape(-1, 1, self.side, self.side)

    def from_matrix(self, M) -> np.ndarray:
        """(N, 1, 8, 8) ou (N, 8, 8) -> (N, n) ; ignore le remplissage."""
        M = np.asarray(M, dtype=np.float32).reshape(len(M), -1)
        return M[:, : self.n]

    def cell_of(self, feature: str) -> tuple[int, int]:
        i = self.features.index(feature)
        return divmod(i, self.side)

    def to_json(self, path: str):
        cells = {f: list(self.cell_of(f)) for f in self.features}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"side": self.side, "pad_value": self.pad_value, "n_features": self.n,
                       "features": self.features, "cells": cells,
                       "mask": self.mask.astype(int).tolist()}, fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MatrixLayout":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(d["features"], d["side"], d["pad_value"])
