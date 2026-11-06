"""Tests de non-régression du protocole (lancer : pytest -q tests/).

Ils vérifient les propriétés qui, si elles cassent, invalident les résultats de l'article :
pas de fuite entre splits, scaler ajusté sur le train seul, ordre temporel, inversibilité.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from cvgen_data import DataPipeline, FlowScaler, MatrixLayout, load_config  # noqa: E402
from cvgen_data.audit import cross_split_exact_overlap, near_duplicate_rate  # noqa: E402
from cvgen_data.constraints import check_rows  # noqa: E402
from cvgen_data.io import normalize_label, norm_col  # noqa: E402
from cvgen_data.split import check_temporal_order  # noqa: E402
import make_mock_data  # noqa: E402


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    d = tmp_path_factory.mktemp("cv")
    make_mock_data.make(str(d / "data" / "raw"), seed=1)
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["classes"]["min_class_count"] = 500
    cfg["audit"]["nn_cap_train"] = 2000
    cfg["reduced_regime"]["benign_cap"] = 2000
    with open(d / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return DataPipeline(load_config(str(d / "config.yaml"))).run_all()


def test_label_normalisation():
    assert normalize_label("Web Attack \x96 Brute Force") == ("Web Attack - Brute Force", False)
    assert normalize_label("Web Attack � XSS")[0] == "Web Attack - XSS"
    assert normalize_label("DoS Hulk - Attempted") == ("DoS Hulk", True)
    assert norm_col(" Fwd Header Length.1") == "fwd_header_length_1"
    assert norm_col("Total Fwd Packet") == "total_fwd_packets"


def test_no_identifiers_in_features(pipe):
    for bad in ("source_ip", "destination_ip", "destination_port", "source_port", "timestamp", "flow_id"):
        assert bad not in pipe.feats


def test_feature_budget(pipe):
    assert len(pipe.feats) <= 64


def test_disjoint_splits(pipe):
    s = {k: set(pipe.df.loc[pipe.df["split"] == k, "row_uid"]) for k in ("train", "val", "test")}
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"])


def test_no_exact_overlap(pipe):
    assert sum(cross_split_exact_overlap(pipe.df, pipe.feats).values()) == 0


def test_temporal_order(pipe):
    assert check_temporal_order(pipe.df) == []


def test_every_class_in_every_split(pipe):
    ct = pd.crosstab(pipe.df["label"], pipe.df["split"])
    assert (ct[["train", "val", "test"]] > 0).all().all()


def test_web_subattacks_in_each_split(pipe):
    w = pipe.df[pipe.df["label"] == "Web Attack"]
    ct = pd.crosstab(w["label_orig"], w["split"])
    assert (ct[["train", "val", "test"]] > 0).all().all()


def test_excluded_classes(pipe):
    assert "Heartbleed" not in set(pipe.df["label_orig"])


def test_scaler_fitted_on_train_only(pipe):
    tr = pipe.df[pipe.df["split"] == "train"]
    ref = FlowScaler("nonbinary").fit(tr, pipe.feats, pipe.schema)
    for c in pipe.feats:
        assert ref.params_[c] == pipe.scaler.params_[c]


def test_scaler_roundtrip(pipe):
    tr = pipe.df[pipe.df["split"] == "train"].iloc[:500]
    back = pipe.scaler.inverse_transform(pipe.scaler.transform(tr))
    np.testing.assert_allclose(back.to_numpy(), tr[pipe.feats].to_numpy(), rtol=1e-4, atol=1e-2)


def test_scaler_json_roundtrip(pipe, tmp_path):
    p = tmp_path / "s.json"
    pipe.scaler.to_json(str(p))
    s2 = FlowScaler.from_json(str(p))
    x = pipe.df.iloc[:100]
    np.testing.assert_allclose(s2.transform(x), pipe.scaler.transform(x))


def test_matrix_roundtrip(pipe):
    lay = MatrixLayout(pipe.feats)
    X = pipe.Xs.iloc[:50].to_numpy()
    M = lay.to_matrix(X)
    assert M.shape == (50, 1, 8, 8)
    np.testing.assert_allclose(lay.from_matrix(M), X)
    assert lay.mask.sum() == len(pipe.feats)
    assert np.all(M.reshape(50, -1)[:, len(pipe.feats):] == 0.0)


def test_reduced_regime_nested(pipe):
    D = pipe.P["processed_dir"]
    ids = {f: set(pd.read_csv(os.path.join(D, "splits", "reduced", f"train_frac{f}_ids.csv.gz"))["row_uid"])
           for f in ("0.01", "0.05", "0.1")}
    train = set(pd.read_csv(os.path.join(D, "splits", "train_ids.csv.gz"))["row_uid"])
    assert ids["0.01"] <= ids["0.05"] <= ids["0.1"] <= train


def test_real_data_respects_reliable_constraints(pipe):
    tr = pipe.df[pipe.df["split"] == "train"]
    v = check_rows(tr, pipe.schema)
    rel = [c for c in v.columns if pipe.schema["reliability"][c] == "reliable"]
    assert v[rel].mean().max() <= 0.01


def test_near_duplicate_detects_copies(pipe):
    tr = pipe.Xs[(pipe.df["split"] == "train") & (pipe.df["label"] == "Bot")].to_numpy()
    r = near_duplicate_rate(tr[:100], tr, tau=1e-3)
    assert r["exact_copy_rate"] == 1.0


def test_saved_files_have_no_ip(pipe):
    df = pd.read_parquet(os.path.join(pipe.P["processed_dir"], "train_raw.parquet"))
    assert not any("ip" == c.split("_")[-1] for c in df.columns)