# cybervision_data — Pipeline de données (Personne 1)

CyberVision Gen · IA Générative Visuelle · CIC-IDS2017 → vecteurs propres, partitions figées, matrices 8×8.

## Démarrage rapide (Colab)
1. Mettre ce dossier sur Drive (ou dans le dépôt GitHub de l'équipe).
2. Télécharger `GeneratedLabelledFlows.zip` (CIC-IDS2017, dossier *TrafficLabelling*) et placer les 8 CSV dans `data/raw/`.
3. Ouvrir `notebooks/01_personne1_pipeline_donnees.ipynb` et exécuter les cellules **dans l'ordre** : chaque étape vérifie ses propres sorties.

Ou en ligne de commande : `python run_pipeline.py --config config.yaml`

Tests (sur un faux CIC-IDS2017 qui reproduit ses défauts) : `pytest -q tests/`

## Structure
```
config.yaml                 toutes les décisions de protocole (fichier qui fige les partitions)
cvgen_data/
  io.py                     chargement robuste (noms, labels, timestamps 12 h, lignes vides)
  clean.py                  inf/NaN, négatifs, doublons exacts, conflits, classes
  split.py                  partition temporelle intra-fenêtre, régime réduit emboîté
  features.py               sélection (train), schéma typé, familles de contraintes
  constraints.py            taux de violation (réel ET synthétique)
  scaling.py                FlowScaler : log1p signé + min-max [-1,1], inversible
  matrix.py                 MatrixLayout : vecteur <-> 8×8 + masque
  audit.py                  doublons post-sélection, seuils τ, quasi-doublons
  eda.py / datacard.py      figures et data card auto-générée
  pipeline.py               orchestration en 11 étapes avec assertions
notebooks/                  notebook Colab pas à pas
tests/                      générateur de données simulées + 17 tests de protocole
docs/METHODOLOGIE.md        justifications, contrats d'interface, checklist, texte pour l'article
docs/exemple_sorties_mock/  à quoi ressemblent les sorties (données SIMULÉES, pas de vrais résultats)
```

## Sorties
| Dossier | Contenu | Git ? |
|---|---|---|
| `data/processed/splits/` | listes d'IDs train/val/test + régime réduit, `MANIFEST.json` (SHA-256) | ✅ |
| `data/processed/*.parquet` | données brutes nettoyées et normalisées par split | ❌ (Drive) |
| `artifacts/` | `scaler.json`, `feature_schema.json`, `matrix_layout.json`, `nn_thresholds.json`, `label_encoding.json` | ✅ |
| `reports/` | `DATA_CARD.md`, `pipeline_report.json`, figures, violations de contraintes | ✅ |

## Déclaration (règle du cours)
Code écrit avec l'assistance d'un outil d'IA générative (Claude, Anthropic), relu, testé et adapté par l'équipe.
