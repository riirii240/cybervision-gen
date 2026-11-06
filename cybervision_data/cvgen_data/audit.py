"""Audit des doublons et des fuites.

1. Doublons exacts APRÈS sélection des variables (moins de colonnes => de nouveaux doublons) :
   intra-split et inter-splits (un flux de test identique à un flux de train est retiré).
2. Quasi-doublons : distance au plus proche voisin (espace normalisé), seuil τ fixé sur la
   VALIDATION (centile bas des distances val -> train). Le même τ sert ensuite à mesurer
   la mémorisation des données synthétiques (Personne 4) : `near_duplicate_rate`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .clean import row_hash


def post_selection_dedup(df: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, dict]:
    order = {"train": 0, "val": 1, "test": 2}
    df = df.assign(_o=df["split"].map(order)).sort_values(["_o", "ts", "row_uid"], kind="mergesort")
    h = row_hash(df, feats)
    rep = {"rows_before": int(len(df))}

    # conflits de labels dans l'espace réduit -> supprimés partout
    nlab = pd.DataFrame({"h": h.values, "y": df["label"].values}).groupby("h")["y"].nunique()
    conflict = h.isin(set(nlab.index[nlab > 1])).values
    rep["conflicting_label_rows"] = {
        "total": int(conflict.sum()),
        "per_class": df.loc[conflict, "label"].value_counts().to_dict()}
    df, h = df.loc[~conflict], h[~conflict]

    # première occurrence dans l'ordre train -> val -> test : les copies en val/test partent
    dup = h.duplicated(keep="first").values
    sp = df["split"].values
    first_split = pd.Series(sp, index=h.values).groupby(level=0).first()
    origin = first_split.reindex(h.values).values
    rep["removed"] = {
        "within_split": {s: int((dup & (sp == s) & (origin == s)).sum()) for s in order},
        "cross_split_from_train": {s: int((dup & (sp == s) & (origin == "train")).sum()) for s in ("val", "test")},
        "cross_split_from_val": {"test": int((dup & (sp == "test") & (origin == "val")).sum())},
    }
    df = df.loc[~dup].drop(columns="_o")
    rep["rows_after"] = int(len(df))
    return df, rep


def cross_split_exact_overlap(df: pd.DataFrame, feats: list[str]) -> dict:
    """Contrôle final : doit valoir 0 partout."""
    h = row_hash(df, feats)
    sets = {s: set(h[df["split"] == s]) for s in ("train", "val", "test")}
    return {"train&val": len(sets["train"] & sets["val"]),
            "train&test": len(sets["train"] & sets["test"]),
            "val&test": len(sets["val"] & sets["test"])}


def _nn_dist(ref: np.ndarray, query: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1).fit(ref)
    return nn.kneighbors(query, return_distance=True)[0][:, 0]


def _cap(X: np.ndarray, k: int, rng) -> np.ndarray:
    return X if len(X) <= k else X[rng.choice(len(X), k, replace=False)]


def nn_audit(Xs: pd.DataFrame, labels: pd.Series, split: pd.Series, cfg: dict):
    """Seuils τ par classe (val -> train) + taux de quasi-doublons du test réel."""
    a = cfg["audit"]
    rng = np.random.default_rng(cfg["seed"])
    thresholds, report = {}, {}
    dists = {}
    for c in sorted(labels.unique()):
        tr = _cap(Xs[(labels == c) & (split == "train")].to_numpy(), a["nn_cap_train"], rng)
        va = _cap(Xs[(labels == c) & (split == "val")].to_numpy(), a["nn_cap_eval"], rng)
        te = _cap(Xs[(labels == c) & (split == "test")].to_numpy(), a["nn_cap_eval"], rng)
        if len(tr) == 0 or len(va) == 0:
            continue
        d_val = _nn_dist(tr, va)
        tau = float(np.percentile(d_val, a["near_dup_percentile"]))
        d_te = _nn_dist(tr, te) if len(te) else np.array([])
        thresholds[c] = tau
        report[c] = {
            "tau": tau,
            "val_to_train": {"median": float(np.median(d_val)), "p05": float(np.percentile(d_val, 5))},
            "test_to_train": {"median": float(np.median(d_te)) if len(d_te) else None,
                              "near_dup_rate": float((d_te <= tau).mean()) if len(d_te) else None},
            "n_ref": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        }
        dists[c] = {"val": d_val, "test": d_te, "tau": tau}
    return thresholds, report, dists


def near_duplicate_rate(X_query: np.ndarray, X_train: np.ndarray, tau: float, exact_atol: float = 1e-6) -> dict:
    """Pour les données synthétiques d'UNE classe (espace normalisé) :
    taux de copies exactes et de quasi-copies (distance <= τ) vis-à-vis du train réel."""
    d = _nn_dist(np.asarray(X_train), np.asarray(X_query))
    return {"exact_copy_rate": float((d <= exact_atol).mean()),
            "near_dup_rate": float((d <= tau).mean()),
            "median_nn_dist": float(np.median(d))}


def nn_distances(X_query, X_ref) -> np.ndarray:
    return _nn_dist(np.asarray(X_ref), np.asarray(X_query))
