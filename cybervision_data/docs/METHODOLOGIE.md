# Personne 1 — Méthodologie, justifications et contrats d'interface

Ce document sert à trois choses : (1) justifier chaque choix dans la section « Données et protocole » de l'article,
(2) dire aux autres membres exactement quoi utiliser, (3) lister ce qui peut invalider les résultats.

---

## 1. Les 11 décisions de protocole et leur justification

| # | Décision | Pourquoi (phrase réutilisable dans l'article) |
|---|---|---|
| 1 | Version **TrafficLabelling** (avec `Timestamp`), ou WTMC-2021 corrigée | La partition temporelle exige un horodatage ; MachineLearningCVE n'en a pas. Des erreurs d'étiquetage et de calcul de variables ont été signalées (Engelen et al., 2021 ; Lanvin et al., 2022). |
| 2 | Correction de l'horloge 12 h | Dans les CSV originaux, les flux d'après-midi sont horodatés 1:00–5:59 sans AM/PM ; sans correction, la partition temporelle mélangerait matin et après-midi. Heuristique : appliquée seulement si l'heure max du fichier ≤ 12. Contrôle : corrélation de Spearman rang-ligne/temps par fichier. |
| 3 | Suppression des lignes inf/NaN et des valeurs négatives impossibles | `Flow Bytes/s` et `Flow Packets/s` sont infinis quand la durée vaut 0 ; des durées/IAT négatifs sont des artefacts d'horloge. Moins de 0,1 % des flux : suppression préférée à l'imputation (aucune information inventée). Exception : `Init_Win_bytes_* = -1` est une sentinelle légitime (fenêtre non observée). |
| 4 | Doublons exacts supprimés **avant** la partition, 1re occurrence chronologique gardée | Un vecteur identique en train et en test est une fuite directe. Les vecteurs identiques à labels contradictoires sont supprimés (inapprenables, probable erreur d'étiquetage). |
| 5 | Exclusion Heartbleed / Infiltration ; regroupement des 3 Web Attacks | Quelques dizaines d'exemples : impossible d'avoir à la fois un train exploitable et un test fiable. Brute Force et XSS sont réputés quasi indiscernables sur ces variables ; SQLi (≈ 20 flux) n'est utilisable que regroupé. |
| 6 | Partition 70/15/15 **dans chaque fenêtre (jour, label original)** | Chaque attaque n'a lieu qu'un jour : une partition par jour retirerait des classes du test. Couper chaque fenêtre dans le temps limite la fuite entre flux d'une même session. Coupure par label **original** puis fusion : sinon XSS/SQLi (plus tardifs) n'apparaîtraient qu'en val/test. |
| 7 | Coupures alignées sur un changement de timestamp | Aucun timestamp identique des deux côtés d'une coupure (horodatage à la minute dans certains fichiers). |
| 8 | Sélection des variables sur le train : identifiants, constantes, colonnes identiques, puis corrélation seulement si > 64 | IP/ports = raccourcis et données personnelles. Les colonnes identiques (`Subflow Fwd Packets = Total Fwd Packets`, `Fwd Header Length.1`) gonflent artificiellement la dimension. L'élagage par corrélation protège les variables nécessaires aux contraintes du modèle B. |
| 9 | Doublons recontrôlés **après** sélection | Retirer des colonnes crée de nouveaux doublons ; les copies en val/test d'un flux de train sont retirées. Contrôle final : chevauchement exact = 0. |
| 10 | log1p signé + min-max [-1, 1], **ajusté sur le train** | Variables à queue très lourde (octets/s jusqu'à 10⁸) ; [-1, 1] correspond à une sortie `tanh`. Inversible exactement pour ramener le synthétique en unités brutes. Val/test non écrêtés. |
| 11 | Seuil de quasi-doublon τ = 1er centile des distances val → train | Fixé sur la validation (règle commune). Interprétation : « aussi proche du train qu'un vrai flux inédit l'est rarement ». |

## 2. Contrats d'interface (ce que les autres membres utilisent)

| Pour qui | Fichier | Usage |
|---|---|---|
| **Tous** | `data/processed/splits/{train,val,test}_ids.csv.gz` + `MANIFEST.json` | Partitions **figées**. Vérifier le SHA-256 avant chaque expérience. |
| **Tous** | `artifacts/label_encoding.json` | Même encodage des classes pour G, C et les classifieurs. |
| **P2 (WGAN-GP A)** | `train_scaled.parquet`, `artifacts/matrix_layout.json` | Entrées du critique : vecteurs [-1,1] (ou matrices via `MatrixLayout.to_matrix`). Sortie du générateur : `tanh`. |
| **P2/P3** | `splits/reduced/train_frac{0.01,0.05,0.1}_ids.csv.gz` | Régime réduit : **le générateur est entraîné sur le même sous-ensemble que le classifieur** (règle d'équité). |
| **P3 (modèle B/C)** | `artifacts/feature_schema.json` | Types (binary/count/continuous), non-négativité, familles min/mean/max, relations. N'imposer que les contraintes `reliable`. |
| **P3/P4** | `cvgen_data.constraints.violation_report` | Taux de violation du synthétique, après `scaler.inverse_transform(..., postprocess=False)`. Comparer au taux réel (`reports/constraint_violations_real_train.csv`). |
| **P4 (évaluation)** | `*_raw.parquet` (RF/GBM), `*_scaled.parquet` (MLP) | Test **uniquement** sur `test_*`. |
| **P4** | `artifacts/nn_thresholds.json` + `cvgen_data.audit.near_duplicate_rate` | Taux de copies / quasi-copies du synthétique par classe. |

Exemple (Personne 4) :
```python
from cvgen_data import FlowScaler
from cvgen_data.constraints import violation_report
from cvgen_data.audit import near_duplicate_rate
import json, pandas as pd
scaler = FlowScaler.from_json("artifacts/scaler.json")
schema = json.load(open("artifacts/feature_schema.json"))
tau = json.load(open("artifacts/nn_thresholds.json"))["tau_per_class"]
train = pd.read_parquet("data/processed/train_scaled.parquet")
X_syn = ...                                   # (N, n_features) en [-1,1], classe "Bot"
raw_syn = scaler.inverse_transform(X_syn)     # unités brutes
print(violation_report(raw_syn, schema).head(10))
print(near_duplicate_rate(X_syn, train[train.label == "Bot"][scaler.features_].to_numpy(), tau["Bot"]))
```

## 3. Checklist avant de dire « données prêtes » (à cocher dans le README de l'équipe)

- [ ] `pytest -q tests/` passe (17 tests).
- [ ] Étape 1 : `n_unparsed = 0` pour chaque fichier ; `fix_12h_applied` cohérent ; Spearman rang/temps ≥ 0,9.
- [ ] Aucune classe finale à moins de ~150 flux en test (sinon regrouper/exclure et le justifier).
- [ ] `cross_split_overlap` = 0 partout.
- [ ] Figure `02_split_timeline.png` : chaque fenêtre bleue → jaune → rouge, sans mélange.
- [ ] Contraintes `unreliable` communiquées à P3.
- [ ] `MANIFEST.json`, listes d'IDs, `config.yaml`, `artifacts/*.json` committés ; **pas** de CSV brut, **pas** d'IP.
- [ ] Data card relue (section 10 « limites » complétée avec vos observations réelles).
- [ ] Partitions annoncées comme **figées** à l'équipe (date + commit).

## 4. Pièges à surveiller sur les vraies données

1. **Mémoire Colab** : ~2,8 M lignes × 80 colonnes en float64 ≈ 2 Go. Ça passe sur 12 Go ; si ça plante, charger fichier par fichier.
2. **Noms de colonnes de la version corrigée** : des alias CICFlowMeter v4 sont gérés (`io.ALIASES`). Vérifier la liste affichée à l'étape 1 ; ajouter un alias si une colonne n'est pas reconnue (sinon contraintes incomplètes).
3. **Dérive temporelle** : les 15 % finaux d'une attaque peuvent différer de son début (ex. PortScan enchaîne plusieurs types de scans). C'est voulu (réaliste) mais à discuter : une baisse de rappel en test n'est pas forcément un défaut du générateur.
4. **Très petites fenêtres** (SQLi ≈ 20 flux en 2 min) : l'alignement sur timestamp peut être impossible, val et test partagent alors la même minute. Signalé dans la data card ; impact négligeable car regroupé.
5. **Destination Port** : très prédictif mais c'est un raccourci (identifiant de service). Retiré par défaut ; si l'équipe veut le garder, en faire une ablation.
6. **Classifieur réel seul proche du plafond** : c'est pour ça que le régime réduit existe. Sans lui, aucun gain du GAN ne sera visible.

## 5. Ce que tu peux écrire dans l'article (section Données)

> Nous utilisons CIC-IDS2017 (Sharafaldin et al., 2018) dans sa version horodatée. Après suppression des valeurs infinies, manquantes et négatives impossibles (X flux), des vecteurs à étiquettes contradictoires (X) et des doublons exacts (X, soit X %), nous excluons Heartbleed et Infiltration (moins de X flux) et regroupons les trois attaques web. Chaque attaque n'apparaissant qu'un seul jour, nous partitionnons chaque fenêtre (jour, attaque) chronologiquement en 70/15/15 % (train/validation/test), coupures alignées sur les changements d'horodatage. Les N variables retenues sont sélectionnées sur le train (suppression des identifiants, des colonnes constantes et dupliquées), puis transformées par log1p signé et min-max vers [-1, 1], ajustés sur le train uniquement. Aucun vecteur n'est partagé entre partitions. Le seuil de quasi-duplication est fixé sur la validation.

(Remplacer les X par les chiffres de `reports/DATA_CARD.md`.)

## 6. Références
- Sharafaldin, Lashkari, Ghorbani (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP.
- Engelen, Rimmer, Joosen (2021). *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*. IEEE SPW, 7–12.
- Lanvin et al. (2023). *Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes*. CRiSIS 2022, Springer.
- Gulrajani et al. (2017). *Improved Training of Wasserstein GANs*. NeurIPS.
