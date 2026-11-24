"""Étape 2 (Personne 3) — Pénalités de plausibilité différentiables (modèle B).

Idée : le générateur sort 63 valeurs dans [-1, 1]. On les ramène en unités réelles
(inverse du scaler de la Personne 1, réécrit en PyTorch pour que le gradient passe),
puis on mesure DE COMBIEN chaque règle gardée à l'étape 1 est cassée.

    pénalité = 0  si toutes les règles sont respectées
    pénalité > 0  sinon, d'autant plus grande que l'erreur est grande

Pour que les grandes variables (durées jusqu'à 1,2e8 µs) n'écrasent pas les petites
(quelques paquets), les écarts sont mesurés en RELATIF :
 - positivité, ordre, relations : comparaison dans l'espace s = signe(x)·log(1+|x|)
   (même ordre que x, mais une échelle raisonnable) ;
 - entier : distance à l'entier le plus proche divisée par (1 + |x|) ;
 - binaire : distance à 0 ou à 1 (déjà entre 0 et 1).

Chaque famille donne une moyenne ; la pénalité totale est la moyenne des familles
(chaque famille pèse autant, quel que soit son nombre de règles).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

DOSSIER = Path(__file__).resolve().parent
ARTEFACTS = DOSSIER.parent / "cybervision_data" / "artifacts"
CATALOGUE = DOSSIER / "resultats" / "catalogue_contraintes.csv"


def slog(x: torch.Tensor) -> torch.Tensor:
    """signe(x) · log(1 + |x|) : conserve l'ordre, compresse les grandes valeurs."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


class PenalitesPlausibilite(nn.Module):
    """Calcule la pénalité de plausibilité d'un batch (N, 63) normalisé dans [-1, 1]."""

    FAMILLES = ("positivité", "ordre", "relation", "entier", "binaire")

    def __init__(self, scaler: dict, schema: dict, regles: list[str]) -> None:
        super().__init__()
        variables = scaler["features"]
        if variables != [f["name"] for f in schema["features"]]:
            raise ValueError("scaler.json et feature_schema.json n'ont pas le même ordre de variables.")
        self.variables = variables
        col = {nom: i for i, nom in enumerate(variables)}
        p = scaler["params"]

        # --- Inverse du scaler (formule de cvgen_data.scaling.FlowScaler) ---
        lo, hi = scaler["out_range"]
        self.lo, self.hi = float(lo), float(hi)
        self.register_buffer("vmin", torch.tensor([p[v]["min"] for v in variables]))
        self.register_buffer("vmax", torch.tensor([p[v]["max"] for v in variables]))
        self.register_buffer("est_log", torch.tensor([p[v]["log"] for v in variables]))

        # --- Traduction des règles gardées en indices de colonnes ---
        familles = {f: c for f, c in schema["stat_families"].items()}
        relations = {r["name"]: r for r in schema["relations"]}
        sentinelle = {f["name"]: f["sentinel"] for f in schema["features"]}
        pos, pos_seuil, inf_a, inf_b, prod, carre, entier, binaire = [], [], [], [], [], [], [], []

        for regle in regles:
            morceaux = regle.split(":")
            type_regle = morceaux[0]
            if type_regle == "nonneg":                       # x >= 0
                nom = familles[morceaux[1]]["std"] if len(morceaux) == 3 else morceaux[1]
                pos.append(col[nom]); pos_seuil.append(0.0)
            elif type_regle == "sentinel_range":             # x >= -1 (sentinelle)
                pos.append(col[morceaux[1]]); pos_seuil.append(float(slog(torch.tensor(float(sentinelle[morceaux[1]])))))
            elif type_regle == "integer":
                entier.append(col[morceaux[1]])
            elif type_regle == "binary":
                binaire.append(col[morceaux[1]])
            elif type_regle == "order":                      # ex. "fwd_iat:min<=mean"
                a, b = morceaux[2].split("<=")
                inf_a.append(col[familles[morceaux[1]][a]]); inf_b.append(col[familles[morceaux[1]][b]])
            elif type_regle == "rel":
                r = relations[morceaux[1]]
                idx = [col[c] for c in r["cols"]]
                if r["kind"] == "le":                        # a <= b
                    inf_a.append(idx[0]); inf_b.append(idx[1])
                elif r["kind"] == "prod":                    # total = nombre x moyenne
                    prod.append(idx)
                elif r["kind"] == "sq":                      # variance = écart-type²
                    carre.append(idx)
                else:
                    raise ValueError(f"Relation inconnue : {regle}")
            else:
                raise ValueError(f"Type de règle inconnu : {regle}")

        as_long = lambda v: torch.tensor(v, dtype=torch.long)
        self.register_buffer("pos", as_long(pos))
        self.register_buffer("pos_seuil", torch.tensor(pos_seuil))
        self.register_buffer("inf_a", as_long(inf_a))
        self.register_buffer("inf_b", as_long(inf_b))
        self.register_buffer("prod", as_long(prod).reshape(-1, 3))
        self.register_buffer("carre", as_long(carre).reshape(-1, 2))
        self.register_buffer("entier", as_long(entier))
        self.register_buffer("binaire", as_long(binaire))
        self.n_regles = len(regles)

    @classmethod
    def depuis_fichiers(cls, catalogue: str | Path = CATALOGUE, artefacts: str | Path = ARTEFACTS):
        """Construit les pénalités à partir des fichiers de l'étape 1 et de la Personne 1."""
        artefacts = Path(artefacts)
        scaler = json.loads((artefacts / "scaler.json").read_text(encoding="utf-8"))
        schema = json.loads((artefacts / "feature_schema.json").read_text(encoding="utf-8"))
        cat = pd.read_csv(catalogue)
        regles = cat.loc[cat["decision"] == "gardée", "regle"].tolist()
        return cls(scaler, schema, regles)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> unités réelles (même formule que FlowScaler.inverse_transform)."""
        z = self.vmin + (y - self.lo) * (self.vmax - self.vmin) / (self.hi - self.lo)
        return torch.where(self.est_log, torch.sign(z) * torch.expm1(torch.abs(z)), z)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Retourne (pénalité totale, pénalité par famille) pour un batch y de forme (N, 63)."""
        x = self.inverse(y)          # unités réelles
        s = slog(x)                  # même ordre que x, échelle raisonnable
        relu = torch.relu
        zero = y.new_zeros(())
        details = {}

        details["positivité"] = relu(self.pos_seuil - s[:, self.pos]).mean() if len(self.pos) else zero
        details["ordre"] = relu(s[:, self.inf_a] - s[:, self.inf_b]).mean() if len(self.inf_a) else zero

        ecarts = []
        if len(self.prod):           # total = nombre x moyenne  (comparé en espace log)
            t, n, m = self.prod[:, 0], self.prod[:, 1], self.prod[:, 2]
            ecarts.append(s[:, t] - slog(x[:, n] * x[:, m]))
        if len(self.carre):          # variance = écart-type²
            v, e = self.carre[:, 0], self.carre[:, 1]
            ecarts.append(s[:, v] - slog(x[:, e] ** 2))
        details["relation"] = (torch.cat(ecarts, dim=1) ** 2).mean() if ecarts else zero

        if len(self.entier):         # distance à l'entier le plus proche, en relatif
            xe = x[:, self.entier]
            frac = xe - torch.round(xe).detach()
            details["entier"] = (frac / (1.0 + xe.abs().detach())).pow(2).mean()
        else:
            details["entier"] = zero

        if len(self.binaire):        # distance à 0 ou à 1
            xb = x[:, self.binaire]
            details["binaire"] = (xb - torch.round(xb.clamp(0, 1)).detach()).pow(2).mean()
        else:
            details["binaire"] = zero

        # Les relations "a <= b" (rel:...) sont comptées avec l'ordre : même calcul.
        total = torch.stack([details[f] for f in self.FAMILLES]).mean()
        return total, details
