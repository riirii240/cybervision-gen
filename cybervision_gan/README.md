# CyberVision Gen — Modèle A (WGAN-GP conditionnel)

Cette partie implémente uniquement le **modèle A** : un générateur MLP et un
critique MLP conditionnés par les 11 classes de CIC-IDS2017. Le générateur apprend
directement les 63 variables tabulaires normalisées ; la matrice 8×8 sert à la
visualisation et conserve sa cellule `(7, 7)` à zéro. Aucun modèle préentraîné
n'est utilisé.

## Contrat avec le pipeline de données

Le code lit sans les modifier :

- `cybervision_data/artifacts/matrix_layout.json` pour l'ordre des 63 variables ;
- `label_encoding.json` pour les 11 indices de classes ;
- `scaler.json` et `feature_schema.json` pour l'export inverse ;
- `train_scaled.parquet` pour l'entraînement ;
- `val_scaled.parquet` pour le suivi et la sélection de `best.pt`.

Le fichier `test_scaled.parquet` n'est ni chargé ni référencé dans la boucle. Il
reste réservé à l'évaluation finale menée séparément. Le tirage d'entraînement est
équilibré par poids inverses aux effectifs : il ne modifie aucune ligne ni aucune
partition, mais évite que BENIGN domine tous les mini-batches.

## Exécution dans Google Colab

1. Placer le dépôt et les Parquet privés sur Google Drive. Une organisation simple
   est `MyDrive/cybervision-gen/` avec le dépôt complet et
   `cybervision_data/data/processed/{train,val}_scaled.parquet`.
2. Importer `notebooks/02_CyberVision_WGAN_GP_Modele_A.ipynb` dans Colab.
3. Activer un accélérateur GPU, puis exécuter les cellules dans l'ordre.
4. Dans la cellule de configuration, adapter `PROJECT_ROOT` ou définir
   `CYBERVISION_PROJECT_ROOT` si le dépôt se trouve ailleurs.
5. Lancer d'abord le smoke test (cinq batches, soit un cycle `n_critic`). Ne lancer l'entraînement complet
   qu'après son succès.

Après une déconnexion, remonter Drive, réexécuter l'initialisation puis fournir
`latest.pt` à `resume_from`. Le checkpoint restaure les deux réseaux, les deux
optimiseurs, la configuration, l'historique et les états aléatoires Python/NumPy/
PyTorch/CUDA.

Les dépendances supplémentaires du notebook sont installées par une cellule `%pip`.
Pour tester localement depuis la racine du dépôt :

```bash
python3 -m pytest -q cybervision_gan/tests
```

## Organisation

```text
cybervision_gan/
├── configs/model_a.yaml
├── evaluation/generate.py
├── models/{generator,critic}.py
├── notebooks/02_CyberVision_WGAN_GP_Modele_A.ipynb
├── training/{gradient_penalty,train}.py
└── tests/test_models.py
```

`best.pt` est choisi par la plus faible distance validation entre moyennes et
écarts-types réels/synthétiques. Cette métrique est un indicateur de suivi, pas une
preuve de qualité finale. Les pertes WGAN n'ont pas l'interprétation d'une loss
supervisée et leur signe seul ne suffit pas à comparer deux expériences.

## Sorties et traçabilité

Le notebook produit sur Drive :

- `best.pt`, `latest.pt`, checkpoints périodiques et `metrics.json` ;
- exemples normalisés **non post-traités** (référence pour mesurer les violations) ;
- tables inversées brutes (`postprocess=False`) et optionnellement post-traitées ;
- configuration, seed, encodage des labels et ordre des variables.

Les checkpoints, Parquet et sorties sont ignorés par Git. Les blocs intitulés
« Exemple de sortie attendue » dans le notebook sont illustratifs et ne sont pas
des résultats expérimentaux exécutés.

## Limite temporelle — résultats provisoires

L'audit actuel de la personne 1 indique `n_unparsed = 0`, mais trois fichiers ont
une corrélation de Spearman rang/temps sous le seuil de contrôle de 0,9 (environ
0,453, 0,437 et 0,647) après la correction d'horloge heuristique. La data card
signale aussi la résolution à la minute de certains timestamps. En conséquence,
les résultats expérimentaux doivent rester **provisoires** jusqu'à résolution ou
validation explicite de ces anomalies temporelles. Ce module ne corrige pas et ne
repartitionne pas les données de la personne 1.
