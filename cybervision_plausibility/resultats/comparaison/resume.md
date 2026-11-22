# Comparaison des modèles A, B et C (Personne 3 — étape 6)

Génération : 5 000 exemples par classe cible, 1 000 par autre classe, graine 42. Référence : train réel. Le test n'est jamais utilisé.

Checkpoints (best.pt, critère de la Personne 2) : A = époque 91, B = époque 98, C = époque 85.

Contrôle de cohérence : train réel, règles fiables de la Personne 1 = 1.2 % (attendu 1,18 %).

## 1. Plausibilité — % de connexions cassant au moins une règle

| jeu                  | taux_144_regles_gardees   | taux_regles_fiables_P1   | famille_binaire   | famille_entier   | famille_ordre   | famille_positivité   | famille_relation   |
|:---------------------|:--------------------------|:-------------------------|:------------------|:-----------------|:----------------|:---------------------|:-------------------|
| Réel (train)         | 0.0 %                     | 1.2 %                    | 0.0 %             | 0.0 %            | 0.0 %           | 0.0 %                | 0.0 %              |
| A (avant arrondi)    | 100.0 %                   | 100.0 %                  | 65.3 %            | 100.0 %          | 85.0 %          | 0.0 %                | 83.1 %             |
| A (après arrondi)    | 100.0 %                   | 100.0 %                  | 0.0 %             | 0.0 %            | 78.9 %          | 0.0 %                | 79.1 %             |
| A (après réparation) | 0.0 %                     | 52.9 %                   | 0.0 %             | 0.0 %            | 0.0 %           | 0.0 %                | 0.0 %              |
| B (avant arrondi)    | 100.0 %                   | 100.0 %                  | 71.0 %            | 100.0 %          | 70.8 %          | 0.0 %                | 79.6 %             |
| B (après arrondi)    | 100.0 %                   | 100.0 %                  | 0.0 %             | 0.0 %            | 79.7 %          | 0.0 %                | 71.1 %             |
| B (après réparation) | 0.0 %                     | 33.6 %                   | 0.0 %             | 0.0 %            | 0.0 %           | 0.0 %                | 0.0 %              |
| C (avant arrondi)    | 100.0 %                   | 100.0 %                  | 58.7 %            | 100.0 %          | 73.6 %          | 0.0 %                | 94.2 %             |
| C (après arrondi)    | 100.0 %                   | 100.0 %                  | 0.0 %             | 0.0 %            | 76.9 %          | 0.0 %                | 78.4 %             |
| C (après réparation) | 0.0 %                     | 63.1 %                   | 0.0 %             | 0.0 %            | 0.0 %           | 0.0 %                | 0.0 %              |

Le taux strict sature avant réparation : un générateur continu ne produit pas d'égalités exactes (min = moyenne = max dans 58 à 91 % des vraies connexions). D'où les deux mesures suivantes.

### 1 b. Ampleur des erreurs — part des connexions avec plus de 10 % d'erreur (sorties brutes)

| jeu          | total_len_fwd=n_fwd*mean_fwd   | total_len_bwd=n_bwd*mean_bwd   | pkt_len_var=pkt_len_std^2   |
|:-------------|:-------------------------------|:-------------------------------|:----------------------------|
| Réel (train) | 0.0 %                          | 0.0 %                          | 0.0 %                       |
| A            | 39.7 %                         | 33.3 %                         | 43.3 %                      |
| B            | 23.6 %                         | 25.0 %                         | 33.6 %                      |
| C            | 28.2 %                         | 26.5 %                         | 33.2 %                      |

### 1 c. Effort de la réparation (reparation.py) — moins = générateur plus cohérent

| modele   | connexions_modifiees   | valeurs_modifiees   | variation_mediane_si_modifiee   | valeurs_modifiees_de_plus_de_10pct   |
|:---------|:-----------------------|:--------------------|:--------------------------------|:-------------------------------------|
| A        | 100.0 %                | 11.4 %              | 12.5 %                          | 6.3 %                                |
| B        | 100.0 %                | 11.1 %              | 12.2 %                          | 5.9 %                                |
| C        | 100.0 %                | 10.7 %              | 10.8 %                          | 5.5 %                                |

## 2-3. Ressemblance, copies, diversité (classes cibles, espace normalisé)

|                                     |   wasserstein_moyen |   ecart_correlations_moyen |   copies_exactes |   quasi_copies |   distance_mediane_plus_proche |   diversite_ratio_ecart_type |
|:------------------------------------|--------------------:|---------------------------:|-----------------:|---------------:|-------------------------------:|-----------------------------:|
| ('Bot', 'A', 'généré')              |              0.0228 |                     0.0456 |                0 |              0 |                         0.1574 |                       0.8355 |
| ('Bot', 'A', 'réparé')              |              0.0232 |                     0.0487 |                0 |              0 |                         0.1809 |                       0.8389 |
| ('Bot', 'B', 'généré')              |              0.0187 |                     0.0499 |                0 |              0 |                         0.1408 |                       0.8329 |
| ('Bot', 'B', 'réparé')              |              0.0187 |                     0.0526 |                0 |              0 |                         0.1503 |                       0.8359 |
| ('Bot', 'C', 'généré')              |              0.0167 |                     0.039  |                0 |              0 |                         0.1359 |                       0.8447 |
| ('Bot', 'C', 'réparé')              |              0.0169 |                     0.0481 |                0 |              0 |                         0.1367 |                       0.8447 |
| ('DoS Slowhttptest', 'A', 'généré') |              0.0568 |                     0.2012 |                0 |              0 |                         0.232  |                       0.7837 |
| ('DoS Slowhttptest', 'A', 'réparé') |              0.0585 |                     0.1919 |                0 |              0 |                         0.2588 |                       0.7811 |
| ('DoS Slowhttptest', 'B', 'généré') |              0.0582 |                     0.1065 |                0 |              0 |                         0.3222 |                       0.857  |
| ('DoS Slowhttptest', 'B', 'réparé') |              0.0587 |                     0.0969 |                0 |              0 |                         0.3353 |                       0.856  |
| ('DoS Slowhttptest', 'C', 'généré') |              0.0525 |                     0.151  |                0 |              0 |                         0.2509 |                       0.8043 |
| ('DoS Slowhttptest', 'C', 'réparé') |              0.0528 |                     0.151  |                0 |              0 |                         0.2686 |                       0.8061 |
| ('PortScan', 'A', 'généré')         |              0.0246 |                     0.0541 |                0 |              0 |                         0.1854 |                       0.9082 |
| ('PortScan', 'A', 'réparé')         |              0.0261 |                     0.0523 |                0 |              0 |                         0.2434 |                       0.917  |
| ('PortScan', 'B', 'généré')         |              0.0469 |                     0.0857 |                0 |              0 |                         0.1881 |                       0.7992 |
| ('PortScan', 'B', 'réparé')         |              0.0469 |                     0.0739 |                0 |              0 |                         0.2347 |                       0.8085 |
| ('PortScan', 'C', 'généré')         |              0.0523 |                     0.1008 |                0 |              0 |                         0.2246 |                       0.8119 |
| ('PortScan', 'C', 'réparé')         |              0.0529 |                     0.0939 |                0 |              0 |                         0.2579 |                       0.8201 |

## 4. Matrices 8×8

Voir `figures/03_matrices_8x8.png`.

## 5. Analyse des échecs — 5 règles les plus cassées (après arrondi)

**Modèle A** : `rel:pkt_len_var=pkt_len_std^2` 56.9 % ; `rel:total_len_fwd=n_fwd*mean_fwd` 55.0 % ; `order:active:mean<=max` 47.4 % ; `rel:total_len_bwd=n_bwd*mean_bwd` 43.4 % ; `order:bwd_packet_length:mean<=max` 42.1 %

**Modèle B** : `rel:pkt_len_var=pkt_len_std^2` 50.4 % ; `rel:total_len_fwd=n_fwd*mean_fwd` 47.0 % ; `rel:total_len_bwd=n_bwd*mean_bwd` 41.3 % ; `order:active:mean<=max` 39.8 % ; `order:fwd_packet_length:mean<=max` 38.4 %

**Modèle C** : `rel:pkt_len_var=pkt_len_std^2` 52.9 % ; `rel:total_len_fwd=n_fwd*mean_fwd` 51.2 % ; `rel:total_len_bwd=n_bwd*mean_bwd` 40.6 % ; `order:bwd_packet_length:mean<=max` 35.8 % ; `order:fwd_packet_length:mean<=max` 34.0 %

## 6. Fidélité à la classe (Random Forest indépendant)

Part des exemples reconnus dans leur classe. Référence « Réel (validation) » = score du même juge sur les vraies connexions de validation.

| jeu               | Bot    | PortScan   | DoS Slowhttptest   | moyenne_11_classes   |
|:------------------|:-------|:-----------|:-------------------|:---------------------|
| Réel (validation) | 99.1 % | 95.6 %     | 80.1 %             | 96.8 %               |
| A                 | 61.9 % | 93.3 %     | 84.1 %             | 85.7 %               |
| A réparé          | 60.6 % | 93.1 %     | 84.3 %             | 85.8 %               |
| B                 | 75.2 % | 91.7 %     | 67.8 %             | 85.0 %               |
| B réparé          | 73.9 % | 91.8 %     | 67.0 %             | 84.0 %               |
| C                 | 73.1 % | 86.8 %     | 60.0 %             | 77.5 %               |
| C réparé          | 72.2 % | 87.1 %     | 60.7 %             | 76.9 %               |

## 7. Export pour la Personne 4

`sorties/<modèle>/generated/` : `synthetic_normalized_unprocessed.parquet`, `synthetic_raw_inverse.parquet` (non post-traité), `synthetic_postprocessed.parquet` (arrondi), `synthetic_repaired.parquet` (arrondi + réparation de cohérence, recommandé).
