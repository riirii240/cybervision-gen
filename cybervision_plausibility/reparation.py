"""Étape 6 (Personne 3) — Réparation déterministe (post-traitement de cohérence).

Appliquée APRÈS la génération et l'inverse du scaler, en unités réelles. Elle ne fait
pas partie du générateur : les modèles sont évalués avant ET après réparation.

Principe : on garde les valeurs principales produites par le générateur (nombres de
paquets, moyennes) et on ajuste seulement les valeurs qui en découlent (totaux, variance)
ou les extrêmes (min, max). Une valeur n'est modifiée QUE si la règle correspondante est
cassée selon l'outil de la Personne 1 (cvgen_data.constraints.check_rows) : une vraie
connexion cohérente n'est jamais modifiée.

Règles, dans cet ordre (une règle n'en casse pas une autre) :
  1. entiers / drapeaux : arrondi (post-traitement de la Personne 1) ;
  2. total = nombre × moyenne (aller et retour), puis moyenne = total / nombre
     (les tailles sont des octets entiers : la moyenne ne change que de l'arrondi) ;
  3. aucun paquet -> tailles min / moyenne / max / écart-type à 0 ;
  4. min <= moyenne <= max (8 familles), en gardant min et max entiers ;
  5. liens « a <= b » gardés (partie <= tout) ;
  6. variance = écart-type².
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cvgen_data.constraints import check_rows  # Personne 1


def _casse(df: pd.DataFrame, schema: dict, regle: str) -> np.ndarray:
    """Lignes où la règle est cassée, avec exactement la définition de la Personne 1."""
    return check_rows(df, schema)[regle].to_numpy()


def reparer(post: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """`post` : sortie de FlowScaler.inverse_transform(..., postprocess=True) (règle 1 déjà faite)."""
    df = post.copy()
    fam = schema["stat_families"]
    rel = {r["name"]: r["cols"] for r in schema["relations"]}

    # 2. total = nombre × moyenne, puis moyenne = total / nombre (octets entiers)
    for nom in ("total_len_fwd=n_fwd*mean_fwd", "total_len_bwd=n_bwd*mean_bwd"):
        total, nombre, moyenne = rel[nom]
        m = _casse(df, schema, f"rel:{nom}")
        df.loc[m, total] = np.round(df.loc[m, nombre] * df.loc[m, moyenne])
        n = df[nombre].to_numpy(float)
        exacte = np.divide(df[total].to_numpy(float), n, out=np.zeros_like(n), where=n > 0)
        m = (n > 0) & ~np.isclose(df[moyenne].to_numpy(float), exacte, rtol=1e-6, atol=1e-9)
        df.loc[m, moyenne] = exacte[m]

    # 3. aucun paquet -> pas de taille
    for nombre, famille in (("total_fwd_packets", "fwd_packet_length"),
                            ("total_backward_packets", "bwd_packet_length")):
        m = (df[nombre] == 0).to_numpy()
        for c in fam[famille].values():
            df.loc[m, c] = 0.0

    # 4. min <= moyenne <= max (min arrondi vers le bas, max vers le haut : ils restent entiers)
    for nom, cols in fam.items():
        m = _casse(df, schema, f"order:{nom}:min<=mean")
        df.loc[m, cols["min"]] = np.floor(df.loc[m, cols["mean"]])
        m = _casse(df, schema, f"order:{nom}:mean<=max")
        df.loc[m, cols["max"]] = np.ceil(df.loc[m, cols["mean"]])
        m = _casse(df, schema, f"order:{nom}:min<=max")      # sécurité (ne devrait plus arriver)
        df.loc[m, cols["min"]] = df.loc[m, cols["max"]]

    # 5. liens « a <= b » (partie <= tout) : on ajuste la valeur qui n'est ni un compteur ni une moyenne
    for nom, a_ajuster in (("fwd_pkt_len_max<=total_len_fwd", "a"),     # max d'un paquet <= total
                           ("bwd_pkt_len_max<=total_len_bwd", "a"),
                           ("fwd_pkt_len_max<=max_pkt_len", "b"),       # max global >= max aller
                           ("bwd_pkt_len_max<=max_pkt_len", "b"),       # max global >= max retour
                           ("fwd_iat_max<=fwd_iat_total", "b"),         # somme des silences >= plus long
                           ("bwd_iat_max<=bwd_iat_total", "b"),
                           ("act_data_pkt_fwd<=total_fwd_packets", "a")):  # partie <= nombre de paquets
        a, b = rel[nom]
        m = _casse(df, schema, f"rel:{nom}")
        if a_ajuster == "a":
            df.loc[m, a] = df.loc[m, b]
        else:
            df.loc[m, b] = df.loc[m, a]

    # 6. variance = écart-type²
    variance, ecart = rel["pkt_len_var=pkt_len_std^2"]
    m = _casse(df, schema, "rel:pkt_len_var=pkt_len_std^2")
    df.loc[m, variance] = df.loc[m, ecart] ** 2
    return df


def effort_reparation(avant: pd.DataFrame, apres: pd.DataFrame, variables: list[str]) -> dict:
    """Combien la réparation a dû modifier : moins = générateur plus cohérent."""
    x, y = avant[variables].to_numpy(float), apres[variables].to_numpy(float)
    modifie = ~np.isclose(x, y, rtol=1e-9, atol=1e-9)
    relatif = np.abs(y - x) / (np.abs(x) + 1.0)
    return {
        "connexions_modifiees": float(modifie.any(axis=1).mean()),
        "valeurs_modifiees": float(modifie.mean()),
        "variation_mediane_si_modifiee": float(np.median(relatif[modifie])) if modifie.any() else 0.0,
        "valeurs_modifiees_de_plus_de_10pct": float((relatif > 0.10).mean()),
    }
