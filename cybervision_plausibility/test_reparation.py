"""Tests de la réparation déterministe (post-traitement de cohérence).

Lancer depuis la racine de la repo :
    python -m pytest -q cybervision_plausibility
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import comparaison as cp
from cvgen_data.constraints import check_rows
from reparation import effort_reparation, reparer

SCHEMA = json.loads((cp.ARTEFACTS / "feature_schema.json").read_text(encoding="utf-8"))
VARIABLES = [f["name"] for f in SCHEMA["features"]]
GARDEES = pd.read_csv(cp.CATALOGUE).query("decision == 'gardée'")["regle"].tolist()


@pytest.fixture(scope="module")
def reel():
    chemin = cp.DONNEES / "train_raw.parquet"
    if not chemin.exists():
        pytest.skip("train_raw.parquet absent (à récupérer sur le Drive)")
    df = pd.read_parquet(chemin, columns=VARIABLES + ["label"])
    return df.groupby("label").head(1000)[VARIABLES].reset_index(drop=True)


def test_une_vraie_connexion_n_est_jamais_modifiee(reel):
    repare = reparer(reel, SCHEMA)
    assert effort_reparation(reel, repare, VARIABLES)["valeurs_modifiees"] == 0.0


def test_une_connexion_incoherente_est_reparee(reel):
    casse = reel.copy()
    casse["fwd_packet_length_min"] = casse["fwd_packet_length_max"] + 50      # min > max
    casse["packet_length_variance"] = casse["packet_length_std"] ** 2 * 3 + 10  # variance fausse
    casse["total_length_of_fwd_packets"] *= 2                                  # total faux
    casse["act_data_pkt_fwd"] = casse["total_fwd_packets"] + 5                  # partie > tout
    assert check_rows(casse, SCHEMA)[GARDEES].any(axis=1).mean() > 0.5
    repare = reparer(casse, SCHEMA)
    assert check_rows(repare, SCHEMA)[GARDEES].any(axis=1).sum() == 0


def test_la_reparation_garde_les_entiers_et_les_compteurs(reel):
    casse = reel.copy()
    casse["fwd_packet_length_mean"] = casse["fwd_packet_length_mean"] + 0.37
    repare = reparer(casse, SCHEMA)
    entiers = [f["name"] for f in SCHEMA["features"] if f["integer"]]
    assert np.allclose(repare[entiers], np.round(repare[entiers]))
    assert repare["total_fwd_packets"].equals(casse["total_fwd_packets"])      # compteur inchangé
