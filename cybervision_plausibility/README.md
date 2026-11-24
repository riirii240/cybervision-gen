# cybervision_plausibility — Personne 3 : plausibilité et conditionnement renforcé

Ce dossier contient les modèles **B** (contraintes de plausibilité) et **C** (conditionnement
renforcé), ainsi que leur évaluation par rapport au modèle de référence **A**.
Il s'appuie sur `cybervision_data/` (Personne 1) et `cybervision_gan/` (Personne 2)
**sans les modifier**. Le jeu de test n'est jamais utilisé.

## Modèles

| Modèle | Description |
|---|---|
| **A** | WGAN-GP conditionnel de référence (Personne 2), réentraîné avec le même code |
| **B** | A + pénalité de plausibilité différentiable (λ choisi sur la validation) |
| **C** | B + critique à projection (renforce le conditionnement par classe) ; échantillonnage équilibré conservé |

Les trois modèles partagent la même architecture de générateur, la même graine, le même nombre
d'époques et le même critère de sélection du checkpoint.

## Contenu

| Fichier | Rôle |
|---|---|
| `catalogue_contraintes.py` | Vérifie chaque contrainte sur les données réelles et décide lesquelles imposer |
| `penalites.py` | Pénalités de plausibilité différentiables (PyTorch) |
| `critique_projection.py` | Critique à projection du modèle C |
| `entrainement.py` | Entraînement des modèles A, B et C avec un seul et même code |
| `reparation.py` | Post-traitement de cohérence (réparation déterministe) |
| `comparaison.py` | Génération, mesures, figures et export des données synthétiques |
| `creer_zip_colab.py` | Prépare le code pour l'exécution sur Google Colab |
| `notebooks/` | Notebook Colab pour les entraînements sur GPU |
| `resultats/` | Catalogue des contraintes, tableaux et figures de la comparaison |
| `test_*.py` | Tests automatiques |

## Reproduire

Depuis la racine du dépôt, avec les données privées placées dans `cybervision_data/data/processed/` :

```bash
python -m pytest -q cybervision_plausibility                    # tests
python cybervision_plausibility/catalogue_contraintes.py         # 1. catalogue des contraintes
python cybervision_plausibility/entrainement.py --modele A       # 2. entraînements (GPU recommandé)
python cybervision_plausibility/entrainement.py --modele B --lambda 0.1
python cybervision_plausibility/entrainement.py --modele C --lambda 0.1
python cybervision_plausibility/comparaison.py                   # 3. comparaison
```

Les checkpoints sont écrits dans `sorties/` (exclu de Git). Les entraînements ont été réalisés
sur Google Colab (GPU T4) avec le notebook fourni.

## Principaux résultats

- **Catalogue** : 144 contraintes imposées sur 147 ; les contraintes temporelles non fiables sur
  les données réelles ne sont pas imposées.
- **Plausibilité** : le taux « au moins une contrainte violée » sature à 100 % pour tout générateur
  continu, car les données réelles contiennent de nombreuses égalités exactes. Mesurées par leur
  ampleur, les erreurs importantes diminuent nettement avec B et C par rapport à A.
- **Réparation** : après post-traitement, les données synthétiques respectent toutes les contraintes
  imposées ; les données réelles ne sont jamais modifiées ; C nécessite le moins de corrections.
- **Ressemblance et diversité** : comparables entre les trois modèles ; aucune copie des données
  d'entraînement.
- **Fidélité à la classe** : améliorée pour certaines classes rares seulement ; résultat mitigé pour C.

Le détail est dans `resultats/comparaison/resume.md`.

## Limites

- Un seul entraînement par modèle : les petits écarts entre B et C sont à interpréter avec prudence.
- Les relations temporelles restent le point faible des trois modèles.
- Les résultats restent provisoires tant que l'audit temporel des données n'est pas finalisé.

## Données pour la Personne 4

`comparaison.py` exporte, pour chaque modèle, dans `sorties/<modèle>/generated/` : les données
non post-traitées, arrondies, et réparées (`synthetic_repaired.parquet`, recommandé).
