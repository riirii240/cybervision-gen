"""Nettoyage : inf/NaN, valeurs négatives, labels, doublons exacts, classes rares.

Ordre imposé par la fiche : doublons exacts supprimés AVANT la partition.
Chaque étape renvoie un rapport chiffré (par classe) pour l'article et la data card.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .io import candidate_feature_columns

log = logging.getLogger("cvgen")


def row_hash(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Hash 64 bits d'une ligne sur `cols` (exact, NaN-safe)."""
    return pd.util.hash_pandas_object(df[cols], index=False)


def per_class(mask: pd.Series, labels: pd.Series) -> dict:
    return labels[mask].value_counts().to_dict()


def clean_values(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    feats = candidate_feature_columns(df)
    rep = {}
    X = df[feats]
    inf_mask = np.isinf(X.to_numpy())
    rep["inf_per_column"] = {c: int(v) for c, v in zip(feats, inf_mask.sum(0)) if v}
    df[feats] = X.replace([np.inf, -np.inf], np.nan)
    rep["nan_per_column"] = {c: int(v) for c, v in df[feats].isna().sum().items() if v}

    # Colonnes entièrement vides -> supprimées (et non les lignes)
    all_nan = [c for c in feats if df[c].isna().all()]
    if all_nan:
        df = df.drop(columns=all_nan)
        feats = [c for c in feats if c not in all_nan]
    rep["all_nan_columns_dropped"] = all_nan

    bad = df[feats].isna().any(axis=1)
    rep["rows_dropped_nan_inf"] = {"total": int(bad.sum()), "per_class": per_class(bad, df["label_orig"])}
    df = df.loc[~bad].copy()

    # Négatifs : impossibles physiquement sauf sentinelles (-1 = fenêtre TCP non observée)
    sentinels = set(cfg["cleaning"].get("sentinel_features", []))
    neg_cols = [c for c in feats if c not in sentinels]
    neg = df[neg_cols] < 0
    rep["negative_per_column"] = {c: int(v) for c, v in neg.sum().items() if v}
    for c in sentinels & set(feats):
        rep.setdefault("sentinel_below_minus1", {})[c] = int((df[c] < -1).sum())
    neg_rows = neg.any(axis=1)
    if cfg["cleaning"].get("negative_policy", "drop") == "drop":
        rep["rows_dropped_negative"] = {"total": int(neg_rows.sum()),
                                        "per_class": per_class(neg_rows, df["label_orig"])}
        df = df.loc[~neg_rows].copy()
    else:
        df[neg_cols] = df[neg_cols].clip(lower=0)
        rep["rows_clipped_negative"] = int(neg_rows.sum())
    for c in sentinels & set(feats):           # valeurs < -1 sur sentinelle : anomalie
        df = df.loc[df[c] >= -1]
    return df, rep


def apply_label_policy(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    rep = {"attempted_total": int(df["attempted"].sum())}
    pol = cfg["cleaning"].get("attempted_policy", "drop")
    if rep["attempted_total"]:
        if pol == "drop":
            df = df.loc[~df["attempted"]].copy()
        elif pol == "benign":
            df.loc[df["attempted"], "label_orig"] = "BENIGN"
        elif pol == "keep":
            df.loc[df["attempted"], "label_orig"] = df.loc[df["attempted"], "label_orig"] + " - Attempted"
    rep["policy"] = pol
    return df, rep


def deduplicate(df: pd.DataFrame, feats: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Doublons exacts sur les caractéristiques.

    - même vecteur + même label  -> on garde la 1re occurrence chronologique ;
    - même vecteur + labels différents (conflit) -> toutes les occurrences supprimées
      (impossible à apprendre, et signe d'erreur d'étiquetage).
    """
    feats = feats or candidate_feature_columns(df)
    df = df.sort_values(["ts", "row_uid"], na_position="last", kind="mergesort")
    h = row_hash(df, feats)
    rep = {"rows_before": int(len(df)), "per_class_before": df["label_orig"].value_counts().to_dict()}

    nlab = pd.DataFrame({"h": h.values, "y": df["label_orig"].values}).groupby("h")["y"].nunique()
    conflict_h = set(nlab.index[nlab > 1])
    conflict = h.isin(conflict_h)
    rep["conflicting_label_rows"] = {"total": int(conflict.sum()),
                                     "per_class": per_class(conflict, df["label_orig"])}
    df, h = df.loc[~conflict.values], h[~conflict.values]
    dup = h.duplicated(keep="first")
    rep["exact_duplicates_removed"] = {"total": int(dup.sum()), "per_class": per_class(dup, df["label_orig"])}
    df = df.loc[~dup.values].copy()
    rep["rows_after"] = int(len(df))
    rep["per_class_after"] = df["label_orig"].value_counts().to_dict()
    return df, rep


def map_classes(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """label_orig (conservé pour la partition) -> label (classe finale)."""
    ccfg = cfg["classes"]
    inv = {o: g for g, members in (ccfg.get("groups") or {}).items() for o in members}
    excl = set(ccfg.get("exclude") or [])
    rep = {"excluded_explicit": df.loc[df["label_orig"].isin(excl), "label_orig"].value_counts().to_dict()}
    df = df.loc[~df["label_orig"].isin(excl)].copy()
    df["label"] = df["label_orig"].map(lambda l: inv.get(l, l))
    counts = df["label"].value_counts()
    small = counts[counts < ccfg.get("min_class_count", 0)]
    rep["excluded_min_count"] = small.to_dict()
    df = df.loc[~df["label"].isin(small.index)].copy()
    rep["final_class_counts"] = df["label"].value_counts().to_dict()
    rep["groups"] = ccfg.get("groups") or {}
    return df, rep
