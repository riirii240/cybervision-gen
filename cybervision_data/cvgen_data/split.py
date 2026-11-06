"""Partition temporelle au sein de chaque fenêtre (jour, label original) + régime réduit.

Pourquoi pas une partition par jour ? Chaque attaque n'apparaît qu'un seul jour :
une partition par jour retirerait des classes entières du test (cf. fiche 11).
On coupe donc chaque fenêtre dans le temps : premiers 70 % -> train, 15 % -> val,
derniers 15 % -> test. Les flux d'une même session restent majoritairement du même côté.

Pourquoi par label ORIGINAL ? Brute Force, XSS et SQLi ont lieu à des heures
différentes : couper le groupe "Web Attack" fusionné mettrait XSS/SQLi uniquement
en val/test. On coupe d'abord chaque sous-attaque, puis on fusionne.
"""
from __future__ import annotations

import hashlib
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger("cvgen")
SPLITS = ("train", "val", "test")


def _aligned_cut(t: np.ndarray, c: int) -> int:
    """Déplace l'indice de coupure c vers la frontière de timestamp la plus proche."""
    n = len(t)
    if c <= 0 or c >= n:
        return c
    left = int(np.searchsorted(t, t[c], side="left"))
    right = int(np.searchsorted(t, t[c], side="right"))
    cands = [x for x in (left, right) if 0 < x < n]
    return min(cands, key=lambda x: abs(x - c)) if cands else c


def temporal_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, dict]:
    fr = np.asarray(cfg["split"]["fractions"], dtype=float)
    assert abs(fr.sum() - 1) < 1e-9, "les fractions doivent sommer à 1"
    align = cfg["split"].get("align_to_timestamp", True)
    gap = float(cfg["split"].get("gap_fraction", 0.0))
    has_ts = df["ts"].notna().all()
    out = pd.Series("drop", index=df.index, dtype=object)
    rep = {"time_source": "timestamp" if has_ts else "row_order", "windows": []}

    for (day, lab), g in df.groupby(["day", "label_orig"], sort=True):
        g = g.sort_values(["ts", "row_uid"] if has_ts else ["row_uid"], kind="mergesort")
        n = len(g)
        t = g["ts"].values.astype("int64") if has_ts else g["row_uid"].values
        c1, c2 = int(round(n * fr[0])), int(round(n * (fr[0] + fr[1])))
        if align and has_ts:
            c1a, c2a = _aligned_cut(t, c1), _aligned_cut(t, c2)
            # On n'aligne que si chaque split garde au moins 1 flux
            if 0 < c1a < c2a < n:
                c1, c2 = c1a, c2a
        lab_arr = np.empty(n, dtype=object)
        lab_arr[:c1], lab_arr[c1:c2], lab_arr[c2:] = "train", "val", "test"
        if gap > 0:
            k = int(round(n * gap / 2))
            for c in (c1, c2):
                lab_arr[max(0, c - k):min(n, c + k)] = "drop"
        out.loc[g.index] = lab_arr
        w = {"day": day, "label_orig": lab, "n": n,
             **{s: int((lab_arr == s).sum()) for s in SPLITS}}
        if has_ts:
            w["start"], w["end"] = str(g["ts"].iloc[0]), str(g["ts"].iloc[-1])
            w["cut_train_val"] = str(g["ts"].iloc[c1]) if c1 < n else None
            w["cut_val_test"] = str(g["ts"].iloc[c2]) if c2 < n else None
        if min(w["train"], w["val"], w["test"]) == 0:
            log.warning("Fenêtre %s/%s : un split est vide (%s)", day, lab, w)
        rep["windows"].append(w)
    return out, rep


def check_temporal_order(df: pd.DataFrame) -> list[str]:
    """Vérifie max(ts train) <= min(ts val) <= ... dans chaque fenêtre. Retourne les erreurs."""
    errors = []
    if not df["ts"].notna().all():
        key = "row_uid"
    else:
        key = "ts"
    for (day, lab), g in df.groupby(["day", "label_orig"]):
        parts = {s: g.loc[g["split"] == s, key] for s in SPLITS}
        if len(parts["train"]) and len(parts["val"]) and parts["train"].max() > parts["val"].min():
            errors.append(f"{day}/{lab}: train > val")
        if len(parts["val"]) and len(parts["test"]) and parts["val"].max() > parts["test"].min():
            errors.append(f"{day}/{lab}: val > test")
    return errors


def minority_classes(labels: pd.Series) -> list[str]:
    return sorted(set(labels.unique()) - {"BENIGN"})


def reduced_regime(train: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """Sous-ensembles EMBOÎTÉS du train (1 % ⊂ 5 % ⊂ 10 %) pour les classes choisies.

    Règle d'équité : le générateur ET le classifieur utilisent le même sous-ensemble.
    """
    rc = cfg["reduced_regime"]
    rng = np.random.default_rng(cfg["seed"])
    cls = rc.get("classes", "minority")
    if cls == "minority":
        target = minority_classes(train["label"])
    elif cls == "all":
        target = sorted(train["label"].unique())
    else:
        target = list(cls)
    fracs = sorted(rc["fractions"])
    # Une permutation par classe, tirée une fois -> emboîtement garanti
    perms = {c: rng.permutation(train.index[train["label"] == c].to_numpy())
             for c in sorted(train["label"].unique())}
    out = {}
    for f in fracs:
        keep = []
        for c, p in perms.items():
            if c in target:
                k = max(int(round(len(p) * f)), min(rc.get("min_per_class", 0), len(p)))
                keep.append(p[:k])
            elif c == "BENIGN" and rc.get("benign_cap"):
                keep.append(p[: rc["benign_cap"]])
            else:
                keep.append(p)
        idx = np.sort(np.concatenate(keep))
        out[f"frac{f:g}"] = train.loc[idx, "row_uid"]
    return out


def save_id_list(uids: pd.Series, path: str) -> str:
    """Sauve une liste d'identifiants triée + renvoie son SHA-256 (versionnement)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.sort(uids.to_numpy(dtype=np.int64))
    pd.DataFrame({"row_uid": arr}).to_csv(path, index=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()
