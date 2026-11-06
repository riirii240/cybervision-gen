"""Orchestration du pipeline Personne 1, étape par étape (utilisable depuis le notebook ou en CLI).

    python -m cvgen_data.pipeline --config config.yaml

Chaque étape écrit son rapport dans `self.report[...]` et se termine par des vérifications
(assert) : si une vérification échoue, le pipeline s'arrête au lieu de produire des données fausses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
import yaml

from . import clean, constraints, eda, features, io, split
from .audit import cross_split_exact_overlap, nn_audit, post_selection_dedup
from .matrix import MatrixLayout
from .scaling import FlowScaler

log = logging.getLogger("cvgen")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    for k, v in cfg["paths"].items():
        if not os.path.isabs(v):
            cfg["paths"][k] = os.path.join(base, v)
    return cfg

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


class DataPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "python": platform.python_version(), "pandas": pd.__version__}
        self.P = cfg["paths"]
        for k in ("interim_dir", "processed_dir", "artifacts_dir", "reports_dir"):
            os.makedirs(self.P[k], exist_ok=True)
        self.fig_dir = os.path.join(self.P["reports_dir"], "figures")
        self.df = None

    # ------------------------------------------------------------------ 1
    def step1_load(self):
        self.df, rep = io.load_raw(self.cfg)
        self.report["load"] = rep
        self.raw_counts = self.df["label_orig"].value_counts().to_dict()
        assert len(self.df) > 0
        assert self.df["row_uid"].is_unique
        return rep

    # ------------------------------------------------------------------ 2
    def step2_clean(self):
        df, rep_v = clean.clean_values(self.df, self.cfg)
        df, rep_l = clean.apply_label_policy(df, self.cfg)
        feats = io.candidate_feature_columns(df)
        assert np.isfinite(df[feats].to_numpy()).all(), "inf/NaN restants"
        self.df = df
        self.report["clean"] = {"values": rep_v, "labels": rep_l}
        return self.report["clean"]

    # ------------------------------------------------------------------ 3
    def step3_dedup_and_classes(self):
        feats = io.candidate_feature_columns(self.df)
        df, rep_d = clean.deduplicate(self.df, feats)
        assert not clean.row_hash(df, feats).duplicated().any()
        df, rep_c = clean.map_classes(df, self.cfg)
        self.df = df
        self.dedup_counts = df["label_orig"].value_counts().to_dict()
        self.report["dedup"], self.report["classes"] = rep_d, rep_c
        eda.class_distribution(self.raw_counts, self.dedup_counts,
                               os.path.join(self.fig_dir, "01_class_distribution.png"))
        return {"dedup": rep_d, "classes": rep_c}

    # ------------------------------------------------------------------ 4
    def step4_split(self):
        self.df["split"], rep = split.temporal_split(self.df, self.cfg)
        self.df = self.df.loc[self.df["split"] != "drop"].copy()
        errs = split.check_temporal_order(self.df)
        assert not errs, f"Ordre temporel violé : {errs[:5]}"
        for s in split.SPLITS:      # chaque classe finale présente dans chaque split
            missing = set(self.df["label"].unique()) - set(self.df.loc[self.df["split"] == s, "label"])
            assert not missing, f"Classes absentes de {s} : {missing}"
        rep["counts"] = pd.crosstab(self.df["label"], self.df["split"]).to_dict()
        self.report["split"] = rep
        eda.split_timeline(self.df, os.path.join(self.fig_dir, "02_split_timeline.png"), seed=self.cfg["seed"])
        return rep

    # ------------------------------------------------------------------ 5
    def step5_features(self):
        cand = io.candidate_feature_columns(self.df)
        train = self.df[self.df["split"] == "train"]
        self.feats, rep = features.select_features(train, cand, self.cfg)
        rep["dropped_identifiers"] = sorted(
            set(rep["dropped_identifiers"]) | (io.META_COLS - {"label"}) & set(self.df.columns))
        assert len(self.feats) <= self.cfg["features"]["max_features"]
        # nouveaux doublons dans l'espace réduit
        self.df, rep_pd = post_selection_dedup(self.df, self.feats)
        overlap = cross_split_exact_overlap(self.df, self.feats)
        assert sum(overlap.values()) == 0, f"Chevauchement exact entre splits : {overlap}"
        rep["post_selection_dedup"], rep["cross_split_overlap"] = rep_pd, overlap
        self.report["features"] = rep
        train = self.df[self.df["split"] == "train"]
        eda.feature_histograms(train, self.feats, os.path.join(self.fig_dir, "03_feature_histograms.png"))
        eda.correlation_heatmap(train, self.feats, os.path.join(self.fig_dir, "04_correlations.png"),
                                seed=self.cfg["seed"])
        return rep

    # ------------------------------------------------------------------ 6
    def step6_schema_constraints(self):
        train = self.df[self.df["split"] == "train"]
        schema = features.build_schema(train, self.feats, self.cfg)
        real = constraints.violation_report(train, schema, train["label"])
        schema = constraints.annotate_reliability(schema, real,
                                                  self.cfg["audit"]["constraint_reliable_max_violation"])
        self.schema = schema
        A = self.P["artifacts_dir"]
        with open(os.path.join(A, "feature_schema.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(schema), f, indent=2)
        real.to_csv(os.path.join(self.P["reports_dir"], "constraint_violations_real_train.csv"))
        unrel = [k for k, v in schema["reliability"].items() if v == "unreliable"]
        self.report["constraints"] = {
            "n_constraints": len(schema["reliability"]), "n_unreliable": len(unrel), "unreliable": unrel,
            "row_violation_rate_reliable": constraints.row_violation_rate(train, schema, True)}
        return self.report["constraints"]

    # ------------------------------------------------------------------ 7
    def step7_scale(self):
        sc = self.cfg["scaling"]
        train = self.df[self.df["split"] == "train"]
        self.scaler = FlowScaler(sc["log_transform"], sc["skew_threshold"], sc["range"]).fit(
            train, self.feats, self.schema)
        A = self.P["artifacts_dir"]
        self.scaler.to_json(os.path.join(A, "scaler.json"))
        joblib.dump(self.scaler, os.path.join(A, "scaler.joblib"))
        self.Xs = self.scaler.transform(self.df)
        # contrôles : train exactement dans [-1, 1] ; inversion exacte
        tr = self.Xs[self.df["split"] == "train"]
        assert tr.min().min() >= -1 - 1e-5 and tr.max().max() <= 1 + 1e-5
        back = self.scaler.inverse_transform(tr.iloc[:2000])
        orig = train[self.feats].iloc[:2000]
        err = float(np.max(np.abs(back.to_numpy() - orig.to_numpy()) / (1 + np.abs(orig.to_numpy()))))
        assert err < 1e-3, f"inverse_transform imprécis ({err})"
        oor = {s: self.scaler.out_of_range_rate(self.df[self.df["split"] == s]) for s in ("val", "test")}
        self.report["scaling"] = {
            "n_log": int(sum(p["log"] for p in self.scaler.params_.values())),
            "inverse_rel_error_max": err,
            "out_of_range_mean": {s: float(np.mean(list(v.values()))) for s, v in oor.items()},
            "out_of_range_worst": {s: sorted(v.items(), key=lambda x: -x[1])[:5] for s, v in oor.items()}}
        return self.report["scaling"]

    # ------------------------------------------------------------------ 8
    def step8_matrix(self):
        m = self.cfg["matrix"]
        self.layout = MatrixLayout(self.feats, m["side"], m["pad_value"])
        self.layout.to_json(os.path.join(self.P["artifacts_dir"], "matrix_layout.json"))
        sample = self.Xs.iloc[:1000].to_numpy()
        assert np.allclose(self.layout.from_matrix(self.layout.to_matrix(sample)), sample)
        tr = self.df["split"] == "train"
        eda.class_matrices(self.Xs[tr], self.df.loc[tr, "label"], self.layout,
                           os.path.join(self.fig_dir, "05_matrices_8x8.png"), seed=self.cfg["seed"])
        self.report["matrix"] = {"n_features": self.layout.n, "n_pad": int((self.layout.mask == 0).sum())}
        return self.report["matrix"]

    # ------------------------------------------------------------------ 9
    def step9_save_splits(self):
        D = self.P["processed_dir"]
        meta = ["row_uid", "source_file", "row_in_file", "day", "ts", "label_orig", "label", "split"]
        manifest = {}
        for s in split.SPLITS:
            idx = self.df["split"] == s
            raw = self.df.loc[idx, meta + self.feats]
            raw.to_parquet(os.path.join(D, f"{s}_raw.parquet"), index=False)
            scl = pd.concat([self.df.loc[idx, ["row_uid", "label"]], self.Xs[idx]], axis=1)
            scl.to_parquet(os.path.join(D, f"{s}_scaled.parquet"), index=False)
            manifest[f"splits/{s}_ids.csv.gz"] = split.save_id_list(
                self.df.loc[idx, "row_uid"], os.path.join(D, "splits", f"{s}_ids.csv.gz"))
        train = self.df[self.df["split"] == "train"]
        red = split.reduced_regime(train, self.cfg)
        red_counts = {}
        prev = None
        for name, uids in red.items():
            manifest[f"splits/reduced/train_{name}_ids.csv.gz"] = split.save_id_list(
                uids, os.path.join(D, "splits", "reduced", f"train_{name}_ids.csv.gz"))
            red_counts[name] = train.set_index("row_uid").loc[uids, "label"].value_counts().to_dict()
            cur = set(uids[train.set_index("row_uid").loc[uids, "label"].values != "BENIGN"])
            if prev is not None:
                assert prev <= cur, "régime réduit non emboîté"
            prev = cur
        labels = sorted(self.df["label"].unique())
        with open(os.path.join(self.P["artifacts_dir"], "label_encoding.json"), "w", encoding="utf-8") as f:
            json.dump({l: i for i, l in enumerate(labels)}, f, indent=2)
        with open(os.path.join(D, "MANIFEST.json"), "w", encoding="utf-8") as f:
            json.dump({"config_sha256": hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest(),
                       "files_sha256": manifest}, f, indent=2)
        self.report["saved"] = {"manifest": manifest, "reduced_regime_counts": red_counts}
        return self.report["saved"]

    # ------------------------------------------------------------------ 10
    def step10_audit(self):
        th, rep, dists = nn_audit(self.Xs, self.df["label"], self.df["split"], self.cfg)
        with open(os.path.join(self.P["artifacts_dir"], "nn_thresholds.json"), "w", encoding="utf-8") as f:
            json.dump({"space": "scaled [-1,1]", "metric": "euclidean",
                       "rule": f"percentile {self.cfg['audit']['near_dup_percentile']} des distances val->train",
                       "tau_per_class": th}, f, indent=2)
        eda.nn_histograms(dists, os.path.join(self.fig_dir, "06_nn_distances.png"))
        self.report["nn_audit"] = rep
        return rep

    # ------------------------------------------------------------------ 11
    def step11_datacard(self):
        from .datacard import write_datacard
        self.report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.P["reports_dir"], "pipeline_report.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(self.report), f, indent=2, default=str)
        path = write_datacard(self)
        return path

    def run_all(self):
        steps = [m for m in dir(self) if m.startswith("step")]
        for name in sorted(steps, key=lambda s: int(s[4:].split("_")[0])):
            t0 = time.time()
            log.info("== %s", name)
            getattr(self, name)()
            log.info("   ok (%.1fs)", time.time() - t0)
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    DataPipeline(load_config(a.config)).run_all()


if __name__ == "__main__":
    main()
