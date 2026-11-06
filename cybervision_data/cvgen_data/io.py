"""Chargement robuste des CSV CIC-IDS2017 (originaux ou version corrigée).

Problèmes connus gérés ici :
- espaces en tête des noms de colonnes (" Flow Duration") ;
- colonne dupliquée "Fwd Header Length.1" ;
- caractère illisible dans les labels Web Attack ("Web Attack � Brute Force") ;
- lignes entièrement vides (fichier Thursday WebAttacks) ;
- timestamps au format jour/mois et heures d'après-midi écrites sur 12 h sans AM/PM ;
- noms de colonnes CICFlowMeter v4 (version corrigée WTMC-2021) différents des originaux.
"""
from __future__ import annotations

import glob
import logging
import os
import re

import numpy as np
import pandas as pd

log = logging.getLogger("cvgen")

META_COLS = {"flow_id", "source_ip", "destination_ip", "source_port", "destination_port",
             "protocol", "timestamp", "label", "id", "attempted_category"}

# Noms CICFlowMeter-v4 / variantes -> noms CIC-IDS2017 originaux normalisés
ALIASES = {
    "src_ip": "source_ip", "dst_ip": "destination_ip",
    "src_port": "source_port", "dst_port": "destination_port",
    "total_fwd_packet": "total_fwd_packets",
    "total_bwd_packets": "total_backward_packets",
    "total_bwd_packet": "total_backward_packets",
    "total_length_of_fwd_packet": "total_length_of_fwd_packets",
    "total_length_of_bwd_packet": "total_length_of_bwd_packets",
    "fwd_packets_length_total": "total_length_of_fwd_packets",
    "bwd_packets_length_total": "total_length_of_bwd_packets",
    "packet_length_min": "min_packet_length",
    "packet_length_max": "max_packet_length",
    "fwd_segment_size_avg": "avg_fwd_segment_size",
    "bwd_segment_size_avg": "avg_bwd_segment_size",
    "fwd_bytes_bulk_avg": "fwd_avg_bytes_bulk",
    "fwd_packet_bulk_avg": "fwd_avg_packets_bulk",
    "fwd_bulk_rate_avg": "fwd_avg_bulk_rate",
    "bwd_bytes_bulk_avg": "bwd_avg_bytes_bulk",
    "bwd_packet_bulk_avg": "bwd_avg_packets_bulk",
    "bwd_bulk_rate_avg": "bwd_avg_bulk_rate",
    "fwd_init_win_bytes": "init_win_bytes_forward",
    "bwd_init_win_bytes": "init_win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "cwr_flag_count": "cwe_flag_count",
}

CANONICAL_LABELS = {
    "benign": "BENIGN", "dos hulk": "DoS Hulk", "dos goldeneye": "DoS GoldenEye",
    "dos slowloris": "DoS slowloris", "dos slowhttptest": "DoS Slowhttptest",
    "ddos": "DDoS", "portscan": "PortScan", "bot": "Bot",
    "ftp patator": "FTP-Patator", "ssh patator": "SSH-Patator",
    "web attack brute force": "Web Attack - Brute Force",
    "web attack xss": "Web Attack - XSS",
    "web attack sql injection": "Web Attack - Sql Injection",
    "infiltration": "Infiltration", "infiltration portscan": "Infiltration - Portscan",
    "heartbleed": "Heartbleed",
}

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]


def norm_col(c: str) -> str:
    c = re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip().lower()).strip("_")
    return ALIASES.get(c, c)


def normalize_label(raw: str) -> tuple[str, bool]:
    """Retourne (label canonique, est_attempted)."""
    s = str(raw)
    attempted = "attempted" in s.lower()
    s = re.sub(r"attempted", "", s, flags=re.I)
    key = re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    return CANONICAL_LABELS.get(key, s.strip()), attempted


def _day_from_name(path: str) -> str | None:
    name = os.path.basename(path).lower()
    for d in DAYS:
        if d in name:
            return d
    return None


def _parse_timestamps(s: pd.Series, fix_12h: str, start_hour: int) -> tuple[pd.Series, dict]:
    info = {}
    raw = s.astype(str).str.strip()
    # Formats ISO (ex. 2017-07-05 14:00:00) : pas de dayfirst
    iso = raw.str.match(r"^\d{4}-\d{2}-\d{2}")
    ts = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        ts[iso] = pd.to_datetime(raw[iso], errors="coerce", format="mixed")
    if (~iso).any():
        ts[~iso] = pd.to_datetime(raw[~iso], errors="coerce", dayfirst=True, format="mixed")
    info["n_unparsed"] = int(ts.isna().sum())
    hours = ts.dt.hour
    info["max_hour_before_fix"] = int(hours.max()) if hours.notna().any() else None
    applied = False
    if fix_12h in ("auto", True, "true") and hours.notna().any():
        # Horloge 12 h sans AM/PM : max(heure) <= 12 et heures < début de journée présentes
        if hours.max() <= 12 and (hours < start_hour).any():
            pm = hours < start_hour
            ts[pm] = ts[pm] + pd.Timedelta(hours=12)
            applied = True
            info["n_shifted_pm"] = int(pm.sum())
    info["fix_12h_applied"] = applied
    return ts, info


def load_raw(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Charge et concatène tous les CSV ; ajoute row_uid, source_file, day, ts, label_orig."""
    src = cfg["source"]
    files = sorted(glob.glob(os.path.join(cfg["paths"]["raw_dir"], src.get("file_glob", "*.csv"))))
    if not files:
        raise FileNotFoundError(f"Aucun CSV dans {cfg['paths']['raw_dir']}")
    frames, report = [], {"files": {}}
    offset = 0
    for f in files:
        df = pd.read_csv(f, encoding=src.get("encoding", "latin-1"), low_memory=False)
        n_read = len(df)
        df.columns = [norm_col(c) for c in df.columns]
        if "label" not in df.columns:
            raise KeyError(f"Pas de colonne label dans {f} : {list(df.columns)[:10]}...")
        df = df.dropna(subset=["label"])                       # lignes vides
        df = df[df["label"].astype(str).str.strip() != ""]
        n_empty = n_read - len(df)
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]       # noms dupliqués exacts
        rep = {"rows_read": n_read, "empty_rows_dropped": n_empty}

        if "timestamp" in df.columns:
            ts, tinfo = _parse_timestamps(df["timestamp"], src.get("fix_12h_clock", "auto"),
                                          src.get("workday_start_hour", 8))
            rep.update(tinfo)
            # Cohérence : corrélation rang ligne / temps (doit être élevée)
            ok = ts.notna()
            if ok.sum() > 10:
                rep["spearman_row_vs_time"] = float(pd.Series(np.arange(ok.sum())).corr(
                    pd.Series(ts[ok].values.astype("int64")), method="spearman"))
            df["ts"] = ts.values
        else:
            df["ts"] = pd.NaT
            rep["warning"] = "pas de Timestamp : ordre des lignes utilisé comme proxy temporel"
            log.warning("%s : pas de Timestamp -> repli sur l'ordre des lignes", os.path.basename(f))

        day = _day_from_name(f)
        if day is None and df["ts"].notna().any():
            day = df["ts"].dropna().iloc[0].day_name().lower()
        df["day"] = day or "unknown"
        df["source_file"] = os.path.basename(f)
        df["row_in_file"] = np.arange(len(df), dtype=np.int64)
        df["row_uid"] = np.arange(offset, offset + len(df), dtype=np.int64)
        offset += len(df)

        labs = df["label"].map(normalize_label)
        df["label_orig"] = [l for l, _ in labs]
        df["attempted"] = [a for _, a in labs]
        rep["labels"] = df["label_orig"].value_counts().to_dict()
        report["files"][os.path.basename(f)] = rep
        frames.append(df.drop(columns=["label"]))
        log.info("Chargé %s : %d lignes", os.path.basename(f), len(df))

    df = pd.concat(frames, ignore_index=True, sort=False)
    # Conversion numérique des colonnes de caractéristiques
    feat = candidate_feature_columns(df)
    for c in feat:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feat] = df[feat].astype(np.float64)
    report["n_rows_total"] = int(len(df))
    report["columns"] = list(df.columns)
    report["has_timestamp"] = bool(df["ts"].notna().any())
    return df, report


INTERNAL_COLS = {"ts", "day", "source_file", "row_in_file", "row_uid", "label_orig",
                 "attempted", "label", "split"}


def candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and c not in INTERNAL_COLS]
