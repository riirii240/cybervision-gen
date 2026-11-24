"""Étape 1 (Personne 3) — Catalogue des contraintes de plausibilité.

But : décider quelles règles physiques on impose au générateur du modèle B.

On RÉUTILISE l'outil de la Personne 1 (cvgen_data.constraints.check_rows) : il dit,
pour chaque connexion et chaque règle, si la règle est cassée. Ici on ajoute seulement :
  1. une famille lisible pour chaque règle ;
  2. le taux de violation sur les VRAIES connexions du train (global et par classe) ;
  3. une décision : gardée / à surveiller / écartée.

Seules les vraies données du TRAIN sont lues (jamais la validation ni le test).

Lancer depuis la racine de la repo :
    python cybervision_plausibility/catalogue_contraintes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
DATA = RACINE / "cybervision_data"
sys.path.insert(0, str(DATA))
from cvgen_data.constraints import check_rows  # noqa: E402  (code de la Personne 1)

SORTIE = Path(__file__).resolve().parent / "resultats"


def milliers(n: int) -> str:
    """1560917 -> '1 560 917'."""
    return f"{n:,}".replace(",", " ")

# Seuil fixé par la Personne 1 (config.yaml : audit.constraint_reliable_max_violation)
SEUIL = 0.01
# Classes cibles du projet (data card, section 11)
CLASSES_CIBLES = ["Bot", "PortScan", "DoS Slowhttptest"]

# Préfixe du nom de la règle -> famille lisible
FAMILLES = {
    "nonneg": "positivité",          # valeur >= 0 (dont écart-type >= 0)
    "sentinel_range": "positivité",  # valeur >= -1 (sentinelle légitime)
    "integer": "entier",             # compteur sans virgule
    "binary": "binaire",             # drapeau égal à 0 ou 1
    "order": "ordre",                # min <= moyenne <= max
    "rel": "relation",               # liens entre colonnes (a <= b, total = n x moyenne, var = std²)
}


def charger_train() -> tuple[pd.DataFrame, dict]:
    """Charge le schéma et les 63 variables + le label du train réel (unités brutes)."""
    schema = json.loads((DATA / "artifacts" / "feature_schema.json").read_text(encoding="utf-8"))
    variables = [f["name"] for f in schema["features"]]
    train = pd.read_parquet(DATA / "data" / "processed" / "train_raw.parquet",
                            columns=variables + ["label"])
    return train, schema


def calculer_taux(train: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Taux de violation de chaque règle : global + une colonne par classe."""
    violations = check_rows(train, schema)            # True = règle cassée
    taux = pd.DataFrame({"global": violations.mean()})
    for classe, lignes in train.groupby("label").groups.items():
        taux[classe] = violations.loc[lignes].mean()
    return taux


def decider(taux: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la famille et la décision pour chaque règle."""
    cat = taux.copy()
    cat.insert(0, "famille", [FAMILLES[nom.split(":")[0]] for nom in cat.index])
    cat["max_classes_cibles"] = cat[CLASSES_CIBLES].max(axis=1)

    def decision(ligne) -> str:
        if ligne["global"] > SEUIL:
            return "écartée"
        if ligne["max_classes_cibles"] > SEUIL:
            return "à surveiller"
        return "gardée"

    cat.insert(1, "decision", cat.apply(decision, axis=1))
    cat.index.name = "regle"
    return cat.sort_values(["decision", "famille", "global"], ascending=[True, True, False])


def verifier_coherence(cat: pd.DataFrame, schema: dict) -> float:
    """Écart maximal avec les taux publiés par la Personne 1 (doit être ~0)."""
    publie = pd.Series(schema["real_violation_rate"])
    return float((cat["global"] - publie.reindex(cat.index)).abs().max())


def ecrire_rapport(cat: pd.DataFrame, ecart: float, n_lignes: int) -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    cat.to_csv(SORTIE / "catalogue_contraintes.csv", float_format="%.6f")

    pct = lambda x: f"{100 * x:.2f} %"
    resume = pd.crosstab(cat["famille"], cat["decision"], margins=True, margins_name="total")
    non_gardees = cat[cat["decision"] != "gardée"]
    colonnes = ["famille", "decision", "global"] + CLASSES_CIBLES

    lignes = [
        "# Catalogue des contraintes (Personne 3 — étape 1)",
        "",
        f"Données : train réel, {milliers(n_lignes)} connexions (unités brutes). Validation et test non utilisés.",
        f"Outil : `cvgen_data.constraints.check_rows` (Personne 1). Seuil : {pct(SEUIL)}.",
        f"Contrôle : écart maximal avec les taux publiés par la Personne 1 = {ecart:.2e}.",
        "",
        "## Règle de décision",
        f"- **gardée** : cassée par au plus {pct(SEUIL)} des vraies connexions, au total ET dans chaque classe cible ;",
        f"- **à surveiller** : fiable au total, mais cassée par plus de {pct(SEUIL)} des connexions d'une classe cible "
        f"({', '.join(CLASSES_CIBLES)}) ;",
        f"- **écartée** : cassée par plus de {pct(SEUIL)} des vraies connexions au total (ne pas l'imposer au générateur).",
        "",
        "## Nombre de règles par famille et par décision",
        "",
        resume.to_markdown(),
        "",
        "## Règles non gardées (détail)",
        "",
        non_gardees[colonnes].map(lambda v: pct(v) if isinstance(v, float) else v).to_markdown()
        if len(non_gardees) else "Aucune.",
        "",
        "Toutes les règles et tous les taux par classe : `catalogue_contraintes.csv`.",
        "",
        "## Décision pour le modèle B",
        "",
        f"Le modèle B impose uniquement les **{(cat['decision'] == 'gardée').sum()} règles gardées**. "
        "Les règles « à surveiller » et « écartée » ne sont pas imposées : les imposer pousserait le "
        "générateur à produire des connexions plus « propres » que les vraies connexions des classes "
        "cibles. Elles restent mesurées lors de la comparaison des modèles A, B et C.",
    ]
    (SORTIE / "catalogue_contraintes.md").write_text("\n".join(lignes) + "\n", encoding="utf-8")


def main() -> None:
    print("Chargement du train réel…")
    train, schema = charger_train()
    print(f"  {milliers(len(train))} connexions, {train.shape[1] - 1} variables")
    print("Calcul des taux de violation…")
    cat = decider(calculer_taux(train, schema))
    ecart = verifier_coherence(cat, schema)
    ecrire_rapport(cat, ecart, len(train))
    print(f"\nÉcart avec les taux de la Personne 1 : {ecart:.2e} (doit être ~0)")
    print(cat["decision"].value_counts().to_string())
    print(f"\nRésultats écrits dans {SORTIE}")


if __name__ == "__main__":
    main()
