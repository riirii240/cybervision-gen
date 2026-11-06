"""FlowScaler : log1p signé (variables à queue lourde) puis min-max vers [-1, 1].

- ajusté sur le TRAIN uniquement (jamais val/test, jamais synthétique) ;
- inversible exactement (pour ramener les sorties du GAN en unités brutes) ;
- sérialisé en JSON lisible (versionnable) en plus de joblib.
Les valeurs val/test hors de la plage train ne sont PAS écrêtées (on mesure le décalage).
"""
from __future__ import annotations

import json
from sys import path

import numpy as np
import pandas as pd


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def islog(z):
    return np.sign(z) * np.expm1(np.abs(z))


class FlowScaler:
    def __init__(self, log_transform="nonbinary", skew_threshold=2.0, out_range=(-1.0, 1.0)):
        self.log_transform = log_transform
        self.skew_threshold = skew_threshold
        self.out_range = tuple(out_range)
        self.params_: dict = {}
        self.features_: list[str] = []

    def fit(self, df: pd.DataFrame, features: list[str], schema: dict | None = None):
        types = {f["name"]: f["type"] for f in (schema or {}).get("features", [])}
        self.features_ = list(features)
        for c in features:
            x = df[c].to_numpy(dtype=np.float64)
            is_bin = types.get(c) == "binary" or set(np.unique(x)).issubset({0.0, 1.0})
            if self.log_transform == "none" or is_bin:
                use_log = False
            elif self.log_transform == "auto":
                use_log = abs(pd.Series(x).skew()) > self.skew_threshold
            else:
                use_log = True
            z = slog(x) if use_log else x
            self.params_[c] = {"log": bool(use_log), "min": float(z.min()), "max": float(z.max())}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        lo, hi = self.out_range
        out = {}
        for c in self.features_:
            p = self.params_[c]
            z = df[c].to_numpy(dtype=np.float64)
            z = slog(z) if p["log"] else z
            span = p["max"] - p["min"]
            out[c] = np.zeros_like(z) if span == 0 else lo + (z - p["min"]) * (hi - lo) / span
        return pd.DataFrame(out, index=df.index).astype(np.float32)

    def inverse_transform(self, Xs, postprocess: bool = False, schema: dict | None = None) -> pd.DataFrame:
        """postprocess=True : arrondi des entiers, binaires {0,1}, non-négativité (si schéma)."""
        lo, hi = self.out_range
        Xs = pd.DataFrame(np.asarray(Xs, dtype=np.float64), columns=self.features_) \
            if not isinstance(Xs, pd.DataFrame) else Xs[self.features_].astype(np.float64)
        out = {}
        for c in self.features_:
            p = self.params_[c]
            y = Xs[c].to_numpy()
            span = p["max"] - p["min"]
            z = np.full_like(y, p["min"]) if span == 0 else p["min"] + (y - lo) * span / (hi - lo)
            out[c] = islog(z) if p["log"] else z
        res = pd.DataFrame(out, index=Xs.index)
        if postprocess and schema:
            for f in schema["features"]:
                c = f["name"]
                if f["type"] == "binary":
                    res[c] = (res[c] >= 0.5).astype(float)
                elif f["integer"]:
                    res[c] = np.round(res[c])
                if f["nonnegative"]:
                    res[c] = res[c].clip(lower=0)
                if f["sentinel"] is not None:
                    res[c] = res[c].clip(lower=f["sentinel"])
        return res

    def out_of_range_rate(self, df: pd.DataFrame) -> dict:
        Xs = self.transform(df)
        lo, hi = self.out_range
        return {c: float(((Xs[c] < lo - 1e-6) | (Xs[c] > hi + 1e-6)).mean()) for c in self.features_}

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"log_transform": self.log_transform, "skew_threshold": self.skew_threshold,
                       "out_range": self.out_range, "features": self.features_,
                       "params": self.params_}, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FlowScaler":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        s = cls(d["log_transform"], d["skew_threshold"], d["out_range"])
        s.features_, s.params_ = d["features"], d["params"]
        return s
