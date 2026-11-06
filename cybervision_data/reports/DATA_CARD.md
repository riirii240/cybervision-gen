# Data card — CyberVision Gen (CIC-IDS2017, prétraité)

_Généré automatiquement par `cvgen_data` le 2026-11-06 12:35:31 — ne pas éditer à la main les chiffres._

## 1. Origine et licence
- **Source** : CIC-IDS2017, Canadian Institute for Cybersecurity (UNB), Sharafaldin et al., ICISSP 2018.
  Variante utilisée : `traffic_labelling` — 8 fichiers CSV.
- **Usage** : recherche/enseignement, citation obligatoire de la source. Les données brutes ne sont pas redistribuées.
- **Données personnelles** : adresses IP, ports et identifiants de flux sont **supprimés** de toutes les sorties.
- **Source temporelle** : `timestamp` (`timestamp` = horodatage CICFlowMeter, résolution ≤ 1 s/1 min).

## 2. Nettoyage (unités brutes)
| Étape | Lignes retirées |
|---|---|
| Lignes vides (sans label) | 288602 |
| inf / NaN | 0 |
| Valeurs négatives impossibles | 0 |
| Labels « Attempted » (politique `drop`) | 0 |
| Vecteurs identiques à labels contradictoires | 82453 |
| Doublons exacts (1re occurrence chronologique gardée) | 512202 |
| Doublons après sélection des variables | 0 |

Lignes : 2830743 chargées → 2230254 finales.

**Colonnes avec inf** : aucune — **avec NaN** : aucune
**Négatifs par colonne** : aucun

## 3. Classes
- Regroupements : {'Web Attack': ['Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - Sql Injection']}
  (partition faite **avant** fusion, par label original, pour que chaque sous-attaque soit présente dans chaque split).
- Exclues explicitement : {'Infiltration': 35, 'Heartbleed': 7}
- Exclues (< 1000 flux après dédoublonnage) : aucune

### Effectifs finaux par split
| label            |   train |    val |   test |
|:-----------------|--------:|-------:|-------:|
| BENIGN           | 1325506 | 284483 | 284276 |
| Bot              |     991 |    212 |    210 |
| DDoS             |   88443 |  19107 |  20442 |
| DoS GoldenEye    |    7491 |   2015 |    775 |
| DoS Hulk         |  121512 |  27833 |  23210 |
| DoS Slowhttptest |    3976 |    161 |   1091 |
| DoS slowloris    |    3836 |    667 |    880 |
| FTP-Patator      |    4186 |    859 |    882 |
| PortScan         |    1209 |    388 |    253 |
| SSH-Patator      |    2273 |    445 |    499 |
| Web Attack       |    1494 |    326 |    323 |
| TOTAL            | 1560917 | 336496 | 332841 |

## 4. Partition (figée)
Temporelle **dans chaque fenêtre (jour, label original)** : [0.7, 0.15, 0.15] (train/val/test),
coupures alignées sur des changements de timestamp, gap = 0.0.
Le test réel ne sert jamais à entraîner le générateur, choisir des hyperparamètres ou des seuils.

| day       | label_orig                 |      n |   train |   val |   test | start               | cut_train_val       | cut_val_test        | end                 |
|:----------|:---------------------------|-------:|--------:|------:|-------:|:--------------------|:--------------------|:--------------------|:--------------------|
| friday    | BENIGN                     | 345939 |  241719 | 51525 |  52695 | 2017-07-07 08:59:00 | 2017-07-07 15:10:00 | 2017-07-07 16:04:00 | 2017-07-07 17:02:00 |
| friday    | Bot                        |   1413 |     991 |   212 |    210 | 2017-07-07 09:34:00 | 2017-07-07 11:20:00 | 2017-07-07 12:10:00 | 2017-07-07 12:59:00 |
| friday    | DDoS                       | 127992 |   88443 | 19107 |  20442 | 2017-07-07 15:56:00 | 2017-07-07 16:10:00 | 2017-07-07 16:13:00 | 2017-07-07 16:16:00 |
| friday    | PortScan                   |   1850 |    1209 |   388 |    253 | 2017-07-07 13:05:00 | 2017-07-07 14:55:00 | 2017-07-07 15:09:00 | 2017-07-07 15:23:00 |
| monday    | BENIGN                     | 458449 |  320911 | 68748 |  68790 | 2017-07-03 08:55:58 | 2017-07-03 14:02:50 | 2017-07-03 15:42:57 | 2017-07-03 17:01:34 |
| thursday  | BENIGN                     | 331140 |  231191 | 50644 |  49305 | 2017-07-06 08:59:00 | 2017-07-06 15:03:00 | 2017-07-06 15:49:00 | 2017-07-06 17:04:00 |
| thursday  | Web Attack - Brute Force   |   1470 |    1013 |   225 |    232 | 2017-07-06 09:15:00 | 2017-07-06 09:48:00 | 2017-07-06 09:54:00 | 2017-07-06 10:00:00 |
| thursday  | Web Attack - Sql Injection |     21 |      15 |     3 |      3 | 2017-07-06 10:40:00 | 2017-07-06 10:41:00 | 2017-07-06 10:42:00 | 2017-07-06 10:42:00 |
| thursday  | Web Attack - XSS           |    652 |     466 |    98 |     88 | 2017-07-06 10:15:00 | 2017-07-06 10:30:00 | 2017-07-06 10:33:00 | 2017-07-06 10:35:00 |
| tuesday   | BENIGN                     | 379685 |  265679 | 57124 |  56882 | 2017-07-04 08:53:00 | 2017-07-04 14:39:00 | 2017-07-04 15:55:00 | 2017-07-04 17:00:00 |
| tuesday   | FTP-Patator                |   5927 |    4186 |   859 |    882 | 2017-07-04 09:17:00 | 2017-07-04 10:02:00 | 2017-07-04 10:11:00 | 2017-07-04 10:30:00 |
| tuesday   | SSH-Patator                |   3217 |    2273 |   445 |    499 | 2017-07-04 14:09:00 | 2017-07-04 14:52:00 | 2017-07-04 15:01:00 | 2017-07-04 15:11:00 |
| wednesday | BENIGN                     | 379052 |  266006 | 56442 |  56604 | 2017-07-05 08:42:00 | 2017-07-05 14:36:00 | 2017-07-05 15:30:00 | 2017-07-05 17:10:00 |
| wednesday | DoS GoldenEye              |  10281 |    7491 |  2015 |    775 | 2017-07-05 11:10:00 | 2017-07-05 11:18:00 | 2017-07-05 11:19:00 | 2017-07-05 11:19:00 |
| wednesday | DoS Hulk                   | 172555 |  121512 | 27833 |  23210 | 2017-07-05 10:43:00 | 2017-07-05 10:55:00 | 2017-07-05 10:58:00 | 2017-07-05 11:07:00 |
| wednesday | DoS Slowhttptest           |   5228 |    3976 |   161 |   1091 | 2017-07-05 10:15:00 | 2017-07-05 10:27:00 | 2017-07-05 10:33:00 | 2017-07-05 10:37:00 |
| wednesday | DoS slowloris              |   5383 |    3836 |   667 |    880 | 2017-07-05 09:01:00 | 2017-07-05 10:04:00 | 2017-07-05 10:07:00 | 2017-07-05 14:24:00 |

Chevauchement exact entre splits (espace des variables retenues) : {'train&val': 0, 'train&test': 0, 'val&test': 0} (doit être 0).

### Régime de données réduites (train, emboîté)
Même sous-ensemble pour le générateur **et** le classifieur (règle d'équité).
|                  |   frac0.01 |   frac0.05 |   frac0.1 |
|:-----------------|-----------:|-----------:|----------:|
| BENIGN           |     200000 |     200000 |    200000 |
| DoS Hulk         |       1215 |       6076 |     12151 |
| DDoS             |        884 |       4422 |      8844 |
| DoS GoldenEye    |         75 |        375 |       749 |
| FTP-Patator      |         42 |        209 |       419 |
| DoS Slowhttptest |         40 |        199 |       398 |
| DoS slowloris    |         38 |        192 |       384 |
| SSH-Patator      |         23 |        114 |       227 |
| PortScan         |         20 |         60 |       121 |
| Bot              |         20 |         50 |        99 |
| Web Attack       |         20 |         75 |       149 |

## 5. Variables (63 retenues / 77 candidates)
- Identifiants supprimés : ['destination_ip', 'destination_port', 'flow_id', 'protocol', 'source_ip', 'source_port', 'timestamp']
- Constantes (train) : ['bwd_psh_flags', 'bwd_urg_flags', 'fwd_avg_bytes_bulk', 'fwd_avg_packets_bulk', 'fwd_avg_bulk_rate', 'bwd_avg_bytes_bulk', 'bwd_avg_packets_bulk', 'bwd_avg_bulk_rate']
- Identiques à une autre colonne : {'syn_flag_count': 'fwd_psh_flags', 'cwe_flag_count': 'fwd_urg_flags', 'ece_flag_count': 'rst_flag_count', 'fwd_header_length_1': 'fwd_header_length', 'subflow_fwd_packets': 'total_fwd_packets', 'subflow_bwd_packets': 'total_backward_packets'}
- Élaguées par corrélation : aucune

Ordre (= ordre des cellules 8×8, ligne par ligne) : `flow_duration, flow_bytes_s, flow_packets_s, down_up_ratio, average_packet_size, total_fwd_packets, total_length_of_fwd_packets, fwd_packet_length_max, fwd_packet_length_min, fwd_packet_length_mean, fwd_packet_length_std, fwd_header_length, fwd_packets_s, avg_fwd_segment_size, subflow_fwd_bytes, init_win_bytes_forward, act_data_pkt_fwd, min_seg_size_forward, total_backward_packets, total_length_of_bwd_packets, bwd_packet_length_max, bwd_packet_length_min, bwd_packet_length_mean, bwd_packet_length_std, bwd_header_length, bwd_packets_s, avg_bwd_segment_size, subflow_bwd_bytes, init_win_bytes_backward, min_packet_length, max_packet_length, packet_length_mean, packet_length_std, packet_length_variance, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min, fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min, bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min, fwd_psh_flags, fwd_urg_flags, fin_flag_count, rst_flag_count, psh_flag_count, ack_flag_count, urg_flag_count, active_mean, active_std, active_max, active_min, idle_mean, idle_std, idle_max, idle_min`

## 6. Normalisation
log1p signé (56 variables) puis min-max → [-1.0, 1.0], **ajusté sur le train uniquement**.
Erreur relative max de l'inversion : 3.20e-07.
Part moyenne de valeurs hors [-1,1] : val 0.000 %, test 0.000 % (non écrêtées).

## 7. Représentation 8×8
63 cellules réelles + 1 de remplissage (valeur 0.0, masque dans `artifacts/matrix_layout.json`).
Encodage de visualisation **sans** relation spatiale physique ; l'évaluation reste tabulaire.

## 8. Contraintes de plausibilité (mesurées sur le train réel)
147 contraintes vérifiées ; 1 violées par > 1 % des flux réels
(**à ne pas imposer** au générateur) : ['rel:fwd_iat_total<=flow_duration'].
Lignes réelles violant ≥ 1 contrainte fiable : 1.179 %.
Détail : `reports/constraint_violations_real_train.csv`.

## 9. Audit des quasi-doublons (espace normalisé, distance euclidienne)
τ par classe = centile 1.0 des distances validation → train (`artifacts/nn_thresholds.json`).
|                  |         τ |   méd. val→train |   méd. test→train | quasi-doublons test (≤τ)   |
|:-----------------|----------:|-----------------:|------------------:|:---------------------------|
| BENIGN           | 0.0006994 |        0.09579   |         0.09866   | 1.14 %                     |
| Bot              | 0.002794  |        0.01636   |         0.01464   | 2.86 %                     |
| DDoS             | 0.0001403 |        0.04789   |         0.03405   | 1.60 %                     |
| DoS GoldenEye    | 0.001659  |        0.04927   |         0.06583   | 0.00 %                     |
| DoS Hulk         | 0.004857  |        0.06289   |         0.08444   | 2.30 %                     |
| DoS Slowhttptest | 1.192e-07 |        0.0006789 |         5.759e-06 | 0.00 %                     |
| DoS slowloris    | 0         |        0.003519  |         0.5867    | 0.00 %                     |
| FTP-Patator      | 0.0018    |        0.009537  |         0.01054   | 1.47 %                     |
| PortScan         | 0.0001175 |        0.01164   |         0.03893   | 0.00 %                     |
| SSH-Patator      | 0.003146  |        0.01113   |         0.01726   | 0.40 %                     |
| Web Attack       | 0.0002887 |        0.002426  |         0.002194  | 1.86 %                     |

## 10. Limites connues
- Erreurs d'étiquetage et de calcul de variables signalées dans CIC-IDS2017 (Engelen et al., 2021 ; Lanvin et al., 2022).
- Trafic simulé en laboratoire sur 5 jours : la généralisation à un réseau réel n'est pas démontrée.
- Partition intra-fenêtre : les derniers 15 % d'une attaque peuvent différer de son début (dérive volontairement non corrigée).
- Résolution temporelle des timestamps limitée ; heuristique de correction 12 h documentée dans `pipeline_report.json`.
- Une seule capture : pas de test externe sur un autre réseau.
- **PortScan** : 158 930 flux → 1 850 après nettoyage (80 689 doublons, 76 265 vecteurs identiques à des flux d'autres classes). Sans le port de destination (retiré car identifiant), les flux de scan sont indiscernables. PortScan devient une classe minoritaire ; les performances publiées sur la version non dédoublonnée sont surestimées.
- **Résolution d'horodatage à la minute** : pour les attaques à haut débit, l'alignement des coupures sur les changements de timestamp déséquilibre val/test (DoS Slowhttptest : 161 en val / 1 091 en test ; DoS GoldenEye : 2 015 / 775). Choix assumé : aucune minute n'est partagée entre deux partitions.
- **Conflits d'étiquettes** : 82 453 flux supprimés car leur vecteur apparaît avec plusieurs labels (dont 76 265 PortScan).

## 11. Contrôle de sens (validation)
Random Forest (100 arbres, 300 000 flux du train), évalué sur la validation : macro-F1 = 0,834.
Classes difficiles : PortScan (F1 0,127), Bot (0,511), DoS Slowhttptest (0,636). Aucune classe à 1,000 : pas de signe de fuite.
Classes cibles proposées pour le GAN : PortScan, Bot, DoS Slowhttptest.