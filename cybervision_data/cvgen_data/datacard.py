"""Génère reports/DATA_CARD.md à partir du rapport du pipeline (chiffres réels, pas de copier-coller)."""
from __future__ import annotations

import os

import pandas as pd


def _table(d: dict, cols=("Clé", "Valeur")) -> str:
    if not d:
        return "_aucun_\n"
    lines = [f"| {cols[0]} | {cols[1]} |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in d.items()]
    return "\n".join(lines) + "\n"


def write_datacard(p) -> str:
    r, cfg = p.report, p.cfg
    split_counts = pd.crosstab(p.df["label"], p.df["split"])[["train", "val", "test"]]
    split_counts.loc["TOTAL"] = split_counts.sum()
    win = pd.DataFrame(r["split"]["windows"])
    cols = [c for c in ["day", "label_orig", "n", "train", "val", "test", "start", "cut_train_val", "cut_val_test", "end"] if c in win]
    nn = pd.DataFrame({c: {"τ": f"{v['tau']:.4g}", "méd. val→train": f"{v['val_to_train']['median']:.4g}",
                           "méd. test→train": f"{v['test_to_train']['median']:.4g}" if v['test_to_train']['median'] is not None else "-",
                           "quasi-doublons test (≤τ)": f"{100*v['test_to_train']['near_dup_rate']:.2f} %" if v['test_to_train']['near_dup_rate'] is not None else "-"}
                       for c, v in r["nn_audit"].items()}).T
    cons = r["constraints"]
    fsel = r["features"]
    dd = r["dedup"]
    pv = r["clean"]["values"]
    red = pd.DataFrame(r["saved"]["reduced_regime_counts"]).fillna(0).astype(int)

    md = f"""# Data card — CyberVision Gen (CIC-IDS2017, prétraité)

_Généré automatiquement par `cvgen_data` le {r.get('finished')} — ne pas éditer à la main les chiffres._

## 1. Origine et licence
- **Source** : CIC-IDS2017, Canadian Institute for Cybersecurity (UNB), Sharafaldin et al., ICISSP 2018.
  Variante utilisée : `{cfg['source']['flavor']}` — {len(r['load']['files'])} fichiers CSV.
- **Usage** : recherche/enseignement, citation obligatoire de la source. Les données brutes ne sont pas redistribuées.
- **Données personnelles** : adresses IP, ports et identifiants de flux sont **supprimés** de toutes les sorties.
- **Source temporelle** : `{r['split']['time_source']}` (`timestamp` = horodatage CICFlowMeter, résolution ≤ 1 s/1 min).

## 2. Nettoyage (unités brutes)
| Étape | Lignes retirées |
|---|---|
| Lignes vides (sans label) | {sum(f['empty_rows_dropped'] for f in r['load']['files'].values())} |
| inf / NaN | {pv['rows_dropped_nan_inf']['total']} |
| Valeurs négatives impossibles | {pv.get('rows_dropped_negative', {}).get('total', 0)} |
| Labels « Attempted » (politique `{r['clean']['labels']['policy']}`) | {r['clean']['labels']['attempted_total']} |
| Vecteurs identiques à labels contradictoires | {dd['conflicting_label_rows']['total']} |
| Doublons exacts (1re occurrence chronologique gardée) | {dd['exact_duplicates_removed']['total']} |
| Doublons après sélection des variables | {fsel['post_selection_dedup']['rows_before'] - fsel['post_selection_dedup']['rows_after']} |

Lignes : {r['load']['n_rows_total']} chargées → {len(p.df)} finales.

**Colonnes avec inf** : {pv['inf_per_column'] or 'aucune'} — **avec NaN** : {pv['nan_per_column'] or 'aucune'}
**Négatifs par colonne** : {pv['negative_per_column'] or 'aucun'}

## 3. Classes
- Regroupements : {r['classes']['groups']}
  (partition faite **avant** fusion, par label original, pour que chaque sous-attaque soit présente dans chaque split).
- Exclues explicitement : {r['classes']['excluded_explicit'] or 'aucune'}
- Exclues (< {cfg['classes']['min_class_count']} flux après dédoublonnage) : {r['classes']['excluded_min_count'] or 'aucune'}

### Effectifs finaux par split
{split_counts.to_markdown()}

## 4. Partition (figée)
Temporelle **dans chaque fenêtre (jour, label original)** : {cfg['split']['fractions']} (train/val/test),
coupures alignées sur des changements de timestamp, gap = {cfg['split']['gap_fraction']}.
Le test réel ne sert jamais à entraîner le générateur, choisir des hyperparamètres ou des seuils.

{win[cols].to_markdown(index=False)}

Chevauchement exact entre splits (espace des variables retenues) : {fsel['cross_split_overlap']} (doit être 0).

### Régime de données réduites (train, emboîté)
Même sous-ensemble pour le générateur **et** le classifieur (règle d'équité).
{red.to_markdown()}

## 5. Variables ({fsel['n_selected']} retenues / {fsel['n_candidates']} candidates)
- Identifiants supprimés : {fsel['dropped_identifiers']}
- Constantes (train) : {fsel['dropped_constant']}
- Identiques à une autre colonne : {fsel['dropped_identical']}
- Élaguées par corrélation : {fsel['dropped_correlation'] or 'aucune'}

Ordre (= ordre des cellules 8×8, ligne par ligne) : `{', '.join(p.feats)}`

## 6. Normalisation
log1p signé ({r['scaling']['n_log']} variables) puis min-max → {cfg['scaling']['range']}, **ajusté sur le train uniquement**.
Erreur relative max de l'inversion : {r['scaling']['inverse_rel_error_max']:.2e}.
Part moyenne de valeurs hors [-1,1] : val {100*r['scaling']['out_of_range_mean']['val']:.3f} %, test {100*r['scaling']['out_of_range_mean']['test']:.3f} % (non écrêtées).

## 7. Représentation 8×8
{r['matrix']['n_features']} cellules réelles + {r['matrix']['n_pad']} de remplissage (valeur {cfg['matrix']['pad_value']}, masque dans `artifacts/matrix_layout.json`).
Encodage de visualisation **sans** relation spatiale physique ; l'évaluation reste tabulaire.

## 8. Contraintes de plausibilité (mesurées sur le train réel)
{cons['n_constraints']} contraintes vérifiées ; {cons['n_unreliable']} violées par > {100*cfg['audit']['constraint_reliable_max_violation']:.0f} % des flux réels
(**à ne pas imposer** au générateur) : {cons['unreliable'] or 'aucune'}.
Lignes réelles violant ≥ 1 contrainte fiable : {100*cons['row_violation_rate_reliable']:.3f} %.
Détail : `reports/constraint_violations_real_train.csv`.

## 9. Audit des quasi-doublons (espace normalisé, distance euclidienne)
τ par classe = centile {cfg['audit']['near_dup_percentile']} des distances validation → train (`artifacts/nn_thresholds.json`).
{nn.to_markdown()}

## 10. Limites connues
- Erreurs d'étiquetage et de calcul de variables signalées dans CIC-IDS2017 (Engelen et al., 2021 ; Lanvin et al., 2022).
- Trafic simulé en laboratoire sur 5 jours : la généralisation à un réseau réel n'est pas démontrée.
- Partition intra-fenêtre : les derniers 15 % d'une attaque peuvent différer de son début (dérive volontairement non corrigée).
- Résolution temporelle des timestamps limitée ; heuristique de correction 12 h documentée dans `pipeline_report.json`.
- Une seule capture : pas de test externe sur un autre réseau.
"""
    path = os.path.join(p.P["reports_dir"], "DATA_CARD.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path
