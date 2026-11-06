"""Vérification des contraintes de plausibilité, en UNITÉS BRUTES.

Réutilisable par :
 - Personne 1 : taux de violation sur les données RÉELLES (référence, fiabilité des contraintes) ;
 - Personne 3 : choisir quelles contraintes imposer dans la pénalité du modèle B ;
 - Personne 4 : taux de violation des données SYNTHÉTIQUES (après inverse_transform).

Une contrainte violée par plus de 1 % des flux RÉELS n'est pas une loi des données
(bugs connus de CICFlowMeter) : elle est marquée "unreliable" et ne doit pas être imposée.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-3, 1e-6


def check_rows(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """DataFrame booléen (lignes x contraintes) : True = violation."""
    v = {}
    for f in schema["features"]:
        c = f["name"]
        if c not in df:
            continue
        x = df[c].to_numpy(dtype=float)
        if f["nonnegative"]:
            v[f"nonneg:{c}"] = x < -ATOL
        if f["sentinel"] is not None:
            v[f"sentinel_range:{c}"] = x < f["sentinel"] - ATOL
        if f["type"] == "binary":
            v[f"binary:{c}"] = ~(np.isclose(x, 0) | np.isclose(x, 1))
        elif f["integer"]:
            v[f"integer:{c}"] = ~np.isclose(x, np.round(x), atol=1e-6)
    for fam, cols in schema["stat_families"].items():
        g = {k: df[c].to_numpy(dtype=float) for k, c in cols.items()}
        tol = lambda a: ATOL + RTOL * np.abs(a)
        if "min" in g and "mean" in g:
            v[f"order:{fam}:min<=mean"] = g["min"] > g["mean"] + tol(g["mean"])
        if "mean" in g and "max" in g:
            v[f"order:{fam}:mean<=max"] = g["mean"] > g["max"] + tol(g["max"])
        if "min" in g and "max" in g:
            v[f"order:{fam}:min<=max"] = g["min"] > g["max"] + tol(g["max"])
        if "std" in g:
            v[f"nonneg:{fam}:std"] = g["std"] < -ATOL
    for r in schema["relations"]:
        cols = [df[c].to_numpy(dtype=float) for c in r["cols"]]
        if r["kind"] == "le":
            a, b = cols
            v[f"rel:{r['name']}"] = a > b + ATOL + RTOL * np.abs(b)
        elif r["kind"] == "prod":
            a, b, c = cols
            target = b * c
            v[f"rel:{r['name']}"] = ~np.isclose(a, target, rtol=1e-2, atol=1.0)
        elif r["kind"] == "sq":
            a, b = cols
            v[f"rel:{r['name']}"] = ~np.isclose(a, b ** 2, rtol=1e-2, atol=1.0)
    return pd.DataFrame(v, index=df.index)


def violation_report(df: pd.DataFrame, schema: dict, labels: pd.Series | None = None) -> pd.DataFrame:
    """Taux de violation par contrainte (global et par classe si labels fourni)."""
    viol = check_rows(df, schema)
    rep = pd.DataFrame({"global": viol.mean()})
    if labels is not None:
        for c, idx in labels.groupby(labels).groups.items():
            rep[c] = viol.loc[idx].mean()
    return rep.sort_values("global", ascending=False)


def row_violation_rate(df: pd.DataFrame, schema: dict, only_reliable: bool = True) -> float:
    """Proportion de lignes violant au moins une contrainte (fiable)."""
    viol = check_rows(df, schema)
    if only_reliable and "reliability" in schema:
        keep = [c for c in viol.columns if schema["reliability"].get(c) == "reliable"]
        viol = viol[keep]
    return float(viol.any(axis=1).mean()) if viol.shape[1] else 0.0


def annotate_reliability(schema: dict, real_report: pd.DataFrame, max_rate: float) -> dict:
    schema["reliability"] = {k: ("reliable" if r <= max_rate else "unreliable")
                             for k, r in real_report["global"].items()}
    schema["real_violation_rate"] = {k: float(r) for k, r in real_report["global"].items()}
    return schema
