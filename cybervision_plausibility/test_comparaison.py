"""Tests de l'étape 6 : le script de comparaison mesure et génère correctement.

Lancer depuis la racine de la repo :
    python -m pytest -q cybervision_plausibility
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

import comparaison as cp
from entrainement import construire_modeles
from test_entrainement import CONFIG

TRAIN_RAW = cp.DONNEES / "train_raw.parquet"


def test_vraies_connexions_respectent_les_144_regles_gardees():
    if not TRAIN_RAW.exists():
        pytest.skip("train_raw.parquet absent (à récupérer sur le Drive)")
    schema = json.loads((cp.ARTEFACTS / "feature_schema.json").read_text(encoding="utf-8"))
    cat = pd.read_csv(cp.CATALOGUE).set_index("regle")
    variables = [f["name"] for f in schema["features"]]
    reel = pd.read_parquet(TRAIN_RAW, columns=variables + ["label"])
    reel = reel.groupby("label").head(2000)                 # échantillon de chaque classe
    ligne, _ = cp.plausibilite(reel, cat, schema, "réel")
    assert ligne["taux_144_regles_gardees"] == 0.0           # par construction du catalogue (étape 1)


def test_generation_meme_protocole_que_la_personne_2(tmp_path):
    cfg = json.loads(json.dumps(CONFIG))
    cfg["plausibilite"] = {"lambda": 0.0}
    generateur, critique = construire_modeles(cfg)
    torch.save({"config": cfg, "generator": generateur.state_dict(), "epoch": 0}, tmp_path / "best.pt")
    charge, cfg_charge, epoque = cp.charger_generateur(tmp_path)
    labels = json.loads((cp.ARTEFACTS / "label_encoding.json").read_text(encoding="utf-8"))
    variables = json.loads((cp.ARTEFACTS / "matrix_layout.json").read_text(encoding="utf-8"))["features"]
    faux = cp.generer(charge, cfg_charge, labels, variables)
    assert len(faux) == 3 * cp.N_CIBLE + 8 * cp.N_AUTRE == 23000
    assert faux.groupby("label").size()["Bot"] == 5000 and faux.groupby("label").size()["BENIGN"] == 1000
    assert faux[variables].abs().max().max() <= 1.0          # sortie tanh
    assert faux.equals(cp.generer(charge, cfg_charge, labels, variables))   # graine fixe -> reproductible
