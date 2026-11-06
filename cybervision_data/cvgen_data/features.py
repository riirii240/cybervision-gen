"""Sélection des caractéristiques (sur le TRAIN uniquement) et schéma typé.

Étapes (toutes ajustées sur le train) :
 1. suppression des identifiants / raccourcis (IP, ports, protocole, timestamp) ;
 2. colonnes constantes ;
 3. colonnes strictement identiques à une autre (ex. Subflow Fwd Packets = Total Fwd Packets) ;
 4. seulement si > max_features : élagage par corrélation |rho| > seuil,
    en protégeant les colonnes nécessaires aux contraintes (min/moyenne/max, totaux).

Le schéma (types, bornes, groupes de contraintes) est exporté pour la Personne 3.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

log = logging.getLogger("cvgen")

# Ordre sémantique utilisé pour la matrice 8x8 (lisibilité des visualisations)
SEMANTIC_ORDER = [
    ("flow", r"^(flow_duration|flow_bytes_s|flow_packets_s|down_up_ratio|average_packet_size)$"),
    ("fwd", r"^(total_fwd_packets|total_length_of_fwd_packets|fwd_packet_length_|avg_fwd_segment|fwd_header_length|fwd_packets_s|subflow_fwd|act_data_pkt_fwd|min_seg_size_forward|init_win_bytes_forward)"),
    ("bwd", r"^(total_backward_packets|total_length_of_bwd_packets|bwd_packet_length_|avg_bwd_segment|bwd_header_length|bwd_packets_s|subflow_bwd|init_win_bytes_backward)"),
    ("pkt", r"^(min_packet_length|max_packet_length|packet_length_)"),
    ("iat", r"^(flow_iat_|fwd_iat_|bwd_iat_)"),
    ("flags", r"(flag|psh|urg)"),
    ("bulk", r"bulk"),
    ("activity", r"^(active_|idle_)"),
]

STAT_FAMILIES = {   # famille -> (min, mean, max, std)
    "fwd_packet_length": ("fwd_packet_length_min", "fwd_packet_length_mean", "fwd_packet_length_max", "fwd_packet_length_std"),
    "bwd_packet_length": ("bwd_packet_length_min", "bwd_packet_length_mean", "bwd_packet_length_max", "bwd_packet_length_std"),
    "packet_length": ("min_packet_length", "packet_length_mean", "max_packet_length", "packet_length_std"),
    "flow_iat": ("flow_iat_min", "flow_iat_mean", "flow_iat_max", "flow_iat_std"),
    "fwd_iat": ("fwd_iat_min", "fwd_iat_mean", "fwd_iat_max", "fwd_iat_std"),
    "bwd_iat": ("bwd_iat_min", "bwd_iat_mean", "bwd_iat_max", "bwd_iat_std"),
    "active": ("active_min", "active_mean", "active_max", "active_std"),
    "idle": ("idle_min", "idle_mean", "idle_max", "idle_std"),
}

# Relations entre variables : (nom, type, a, b[, c]) ; "le" : a <= b ; "prod" : a ≈ b*c ; "sq" : a ≈ b²
RELATIONS = [
    ("flow_iat_max<=flow_duration", "le", "flow_iat_max", "flow_duration"),
    ("fwd_iat_total<=flow_duration", "le", "fwd_iat_total", "flow_duration"),
    ("bwd_iat_total<=flow_duration", "le", "bwd_iat_total", "flow_duration"),
    ("fwd_iat_max<=fwd_iat_total", "le", "fwd_iat_max", "fwd_iat_total"),
    ("bwd_iat_max<=bwd_iat_total", "le", "bwd_iat_max", "bwd_iat_total"),
    ("fwd_pkt_len_max<=total_len_fwd", "le", "fwd_packet_length_max", "total_length_of_fwd_packets"),
    ("bwd_pkt_len_max<=total_len_bwd", "le", "bwd_packet_length_max", "total_length_of_bwd_packets"),
    ("fwd_pkt_len_max<=max_pkt_len", "le", "fwd_packet_length_max", "max_packet_length"),
    ("bwd_pkt_len_max<=max_pkt_len", "le", "bwd_packet_length_max", "max_packet_length"),
    ("act_data_pkt_fwd<=total_fwd_packets", "le", "act_data_pkt_fwd", "total_fwd_packets"),
    ("total_len_fwd=n_fwd*mean_fwd", "prod", "total_length_of_fwd_packets", "total_fwd_packets", "fwd_packet_length_mean"),
    ("total_len_bwd=n_bwd*mean_bwd", "prod", "total_length_of_bwd_packets", "total_backward_packets", "bwd_packet_length_mean"),
    ("pkt_len_var=pkt_len_std^2", "sq", "packet_length_variance", "packet_length_std"),
]


def drop_identifiers(feats: list[str], cfg: dict) -> tuple[list[str], list[str]]:
    drop = set(cfg["features"].get("drop_columns", []))
    return [f for f in feats if f not in drop], [f for f in feats if f in drop]


def constant_columns(train: pd.DataFrame, feats: list[str]) -> list[str]:
    return [c for c in feats if train[c].nunique(dropna=False) <= 1]


def identical_columns(train: pd.DataFrame, feats: list[str]) -> dict[str, str]:
    """{colonne supprimée: colonne gardée} pour les colonnes strictement identiques."""
    hashes = {c: pd.util.hash_array(train[c].to_numpy()).sum() for c in feats}  # empreinte rapide
    kept, removed = [], {}
    for c in feats:
        twin = next((k for k in kept if hashes[k] == hashes[c]
                     and np.array_equal(train[k].to_numpy(), train[c].to_numpy())), None)
        if twin is None:
            kept.append(c)
        else:
            removed[c] = twin
    return removed


def protected_columns(feats: list[str]) -> set[str]:
    prot = set()
    for fam in STAT_FAMILIES.values():
        prot.update(c for c in fam[:3] if c in feats)
    for r in RELATIONS:
        prot.update(c for c in r[2:] if c in feats)
    return prot


def correlation_prune(train: pd.DataFrame, feats: list[str], cfg: dict) -> tuple[list[str], list]:
    fc = cfg["features"]
    maxf = fc["max_features"]
    if len(feats) <= maxf:
        return feats, []
    sample = train[feats].sample(min(len(train), fc.get("corr_sample", 200000)), random_state=cfg["seed"])
    corr = sample.rank().corr().abs()        # Spearman
    prot = protected_columns(feats) if fc.get("protect_constraint_groups", True) else set()
    feats = list(feats)
    removed = []
    thr = fc.get("corr_threshold", 0.99)
    while len(feats) > maxf:
        sub = corr.loc[feats, feats].where(~np.eye(len(feats), dtype=bool)).fillna(0)
        cand = [c for c in feats if c not in prot]
        if not cand:
            break
        # colonne non protégée ayant la plus forte corrélation maximale
        c = max(cand, key=lambda x: sub[x].max())
        if sub[c].max() < thr and len(feats) <= maxf:
            break
        removed.append((c, sub[c].idxmax(), float(sub[c].max())))
        feats.remove(c)
    if len(feats) > maxf:
        log.warning("Encore %d variables > %d (toutes protégées ?)", len(feats), maxf)
    return feats, removed


def order_semantically(feats: list[str]) -> list[str]:
    ordered, seen = [], set()
    for _, pat in SEMANTIC_ORDER:
        for f in feats:
            if f not in seen and re.search(pat, f):
                ordered.append(f)
                seen.add(f)
    ordered += [f for f in feats if f not in seen]
    return ordered


def select_features(train: pd.DataFrame, feats: list[str], cfg: dict) -> tuple[list[str], dict]:
    rep = {"n_candidates": len(feats)}
    feats, rep["dropped_identifiers"] = drop_identifiers(feats, cfg)
    const = constant_columns(train, feats)
    feats = [f for f in feats if f not in const]
    rep["dropped_constant"] = const
    ident = identical_columns(train, feats)
    feats = [f for f in feats if f not in ident]
    rep["dropped_identical"] = ident
    feats, pruned = correlation_prune(train, feats, cfg)
    rep["dropped_correlation"] = pruned
    feats = order_semantically(feats)
    rep["n_selected"] = len(feats)
    rep["selected"] = feats
    return feats, rep


def build_schema(train: pd.DataFrame, feats: list[str], cfg: dict) -> dict:
    """Types et bornes observés sur le train (unités brutes) + contraintes candidates."""
    sentinels = set(cfg["cleaning"].get("sentinel_features", []))
    schema = {"features": [], "stat_families": {}, "relations": []}
    for c in feats:
        v = train[c].to_numpy()
        uniq = np.unique(v[: min(len(v), 2_000_000)])
        is_bin = set(np.unique(v)).issubset({0.0, 1.0})
        is_int = bool(np.all(np.mod(v, 1) == 0))
        ftype = "binary" if is_bin else ("count" if is_int else "continuous")
        schema["features"].append({
            "name": c, "type": ftype, "integer": is_int,
            "nonnegative": bool(v.min() >= 0) and c not in sentinels,
            "sentinel": -1 if c in sentinels else None,
            "train_min": float(v.min()), "train_max": float(v.max()),
            "train_q01": float(np.quantile(v, 0.01)), "train_q99": float(np.quantile(v, 0.99)),
            "n_unique_train": int(len(uniq)),
        })
    fs = set(feats)
    for fam, cols in STAT_FAMILIES.items():
        present = {k: c for k, c in zip(("min", "mean", "max", "std"), cols) if c in fs}
        if len(present) >= 2:
            schema["stat_families"][fam] = present
    for r in RELATIONS:
        if all(c in fs for c in r[2:]):
            schema["relations"].append({"name": r[0], "kind": r[1], "cols": list(r[2:])})
    return schema
