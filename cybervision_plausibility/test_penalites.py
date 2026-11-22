"""Tests de l'étape 2 : les pénalités sont justes et utilisables pour l'entraînement.

Lancer depuis la racine de la repo :
    python -m pytest -q cybervision_plausibility
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from penalites import ARTEFACTS, PenalitesPlausibilite

DATA = ARTEFACTS.parent
sys.path.insert(0, str(DATA))
from cvgen_data.scaling import FlowScaler  # noqa: E402  (code de la Personne 1)

TRAIN_SCALED = DATA / "data" / "processed" / "train_scaled.parquet"


@pytest.fixture(scope="module")
def penalites():
    return PenalitesPlausibilite.depuis_fichiers()


@pytest.fixture(scope="module")
def vraies_connexions(penalites):
    """Jusqu'à 1 000 vraies connexions du train par classe (valeurs normalisées)."""
    if not TRAIN_SCALED.exists():
        pytest.skip("train_scaled.parquet absent (à récupérer sur le Drive)")
    train = pd.read_parquet(TRAIN_SCALED, columns=penalites.variables + ["label"])
    echantillon = train.groupby("label").sample(n=1000, replace=True, random_state=0)
    return torch.tensor(echantillon[penalites.variables].to_numpy(np.float32))


def test_les_144_regles_gardees_sont_chargees(penalites):
    assert penalites.n_regles == 144


def test_inverse_identique_a_celui_de_la_personne_1(penalites, vraies_connexions):
    scaler = FlowScaler.from_json(str(ARTEFACTS / "scaler.json"))
    attendu = scaler.inverse_transform(vraies_connexions.numpy().astype(np.float64)).to_numpy()
    obtenu = penalites.inverse(vraies_connexions.double()).numpy()
    # Tolérance relative 1e-5 : les paramètres sont stockés en float32 (précision de l'entraînement).
    np.testing.assert_allclose(obtenu, attendu, rtol=1e-5, atol=1e-5)


def test_vraies_connexions_quasi_pas_punies(penalites, vraies_connexions):
    total, details = penalites(vraies_connexions)
    assert total.item() < 1e-4, details


def test_connexion_cassee_est_punie(penalites, vraies_connexions):
    _, avant = penalites(vraies_connexions[:1])
    col = {nom: i for i, nom in enumerate(penalites.variables)}
    y = vraies_connexions[:1].clone()
    y[0, col["fwd_packet_length_min"]] = 0.9      # min très grand…
    y[0, col["fwd_packet_length_max"]] = -0.9     # … max très petit : min > max
    y[0, col["total_fwd_packets"]] = 0.123        # nombre de paquets non entier
    y[0, col["fin_flag_count"]] = 0.0             # drapeau = 0,5 au lieu de 0 ou 1
    _, apres = penalites(y)
    for famille in ("ordre", "entier", "binaire"):
        assert apres[famille] > avant[famille] + 1e-3, famille


def test_hasard_beaucoup_plus_puni_que_le_reel(penalites, vraies_connexions):
    torch.manual_seed(0)
    hasard = torch.rand_like(vraies_connexions) * 2 - 1     # comme un générateur non entraîné
    reel, _ = penalites(vraies_connexions)
    faux, _ = penalites(hasard)
    assert faux.item() > 100 * reel.item() + 1e-3


def test_gradients_finis_pour_l_entrainement(penalites):
    torch.manual_seed(0)
    y = (torch.rand(256, len(penalites.variables)) * 2 - 1).requires_grad_(True)
    total, _ = penalites(y)
    total.backward()
    assert torch.isfinite(y.grad).all()
    assert y.grad.abs().sum() > 0
