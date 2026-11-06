# Data card — CyberVision Gen (CIC-IDS2017, prétraité)

_Généré automatiquement par `cvgen_data` le 2026-11-04 09:36:06 — ne pas éditer à la main les chiffres._

## 1. Origine et licence
- **Source** : CIC-IDS2017, Canadian Institute for Cybersecurity (UNB), Sharafaldin et al., ICISSP 2018.
  Variante utilisée : `traffic_labelling` — 5 fichiers CSV.
- **Usage** : recherche/enseignement, citation obligatoire de la source. Les données brutes ne sont pas redistribuées.
- **Données personnelles** : adresses IP, ports et identifiants de flux sont **supprimés** de toutes les sorties.
- **Source temporelle** : `timestamp` (`timestamp` = horodatage CICFlowMeter, résolution ≤ 1 s/1 min).

## 2. Nettoyage (unités brutes)
| Étape | Lignes retirées |
|---|---|
| Lignes vides (sans label) | 50 |
| inf / NaN | 79 |
| Valeurs négatives impossibles | 26 |
| Labels « Attempted » (politique `drop`) | 0 |
| Vecteurs identiques à labels contradictoires | 55 |
| Doublons exacts (1re occurrence chronologique gardée) | 990 |
| Doublons après sélection des variables | 0 |

Lignes : 24657 chargées → 23497 finales.

**Colonnes avec inf** : {'flow_bytes_s': 54} — **avec NaN** : {'flow_bytes_s': 79}
**Négatifs par colonne** : {'flow_iat_min': 26}

## 3. Classes
- Regroupements : {'Web Attack': ['Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - Sql Injection']}
  (partition faite **avant** fusion, par label original, pour que chaque sous-attaque soit présente dans chaque split).
- Exclues explicitement : {'Heartbleed': 10}
- Exclues (< 500 flux après dédoublonnage) : aucune

### Effectifs finaux par split
| label         |   train |   val |   test |
|:--------------|--------:|------:|-------:|
| BENIGN        |    8694 |  1863 |   1864 |
| Bot           |     770 |   164 |    164 |
| DDoS          |    1051 |   233 |    210 |
| DoS Hulk      |    1280 |   226 |    287 |
| DoS slowloris |     838 |   193 |    167 |
| FTP-Patator   |     973 |   217 |    205 |
| PortScan      |    1057 |   220 |    220 |
| SSH-Patator   |     914 |   176 |    201 |
| Web Attack    |     925 |   202 |    183 |
| TOTAL         |   16502 |  3494 |   3501 |

## 4. Partition (figée)
Temporelle **dans chaque fenêtre (jour, label original)** : [0.7, 0.15, 0.15] (train/val/test),
coupures alignées sur des changements de timestamp, gap = 0.0.
Le test réel ne sert jamais à entraîner le générateur, choisir des hyperparamètres ou des seuils.

| day       | label_orig                 |    n |   train |   val |   test | start               | cut_train_val       | cut_val_test        | end                 |
|:----------|:---------------------------|-----:|--------:|------:|-------:|:--------------------|:--------------------|:--------------------|:--------------------|
| friday    | BENIGN                     | 2486 |    1739 |   375 |    372 | 2017-07-07 09:00:00 | 2017-07-07 14:43:00 | 2017-07-07 15:55:00 | 2017-07-07 16:59:00 |
| friday    | Bot                        | 1098 |     770 |   164 |    164 | 2017-07-07 10:00:00 | 2017-07-07 10:42:00 | 2017-07-07 10:51:00 | 2017-07-07 10:59:00 |
| friday    | DDoS                       | 1494 |    1051 |   233 |    210 | 2017-07-07 15:54:00 | 2017-07-07 16:11:00 | 2017-07-07 16:15:00 | 2017-07-07 16:17:00 |
| friday    | PortScan                   | 1497 |    1057 |   220 |    220 | 2017-07-07 13:54:00 | 2017-07-07 14:24:00 | 2017-07-07 14:30:00 | 2017-07-07 14:35:00 |
| monday    | BENIGN                     | 2975 |    2087 |   442 |    446 | 2017-07-03 09:00:00 | 2017-07-03 14:29:00 | 2017-07-03 15:44:00 | 2017-07-03 16:59:00 |
| thursday  | BENIGN                     | 1986 |    1383 |   302 |    301 | 2017-07-06 09:00:00 | 2017-07-06 11:04:00 | 2017-07-06 11:31:00 | 2017-07-06 11:59:00 |
| thursday  | Web Attack - Brute Force   |  890 |     631 |   133 |    126 | 2017-07-06 09:18:00 | 2017-07-06 09:48:00 | 2017-07-06 09:54:00 | 2017-07-06 09:59:00 |
| thursday  | Web Attack - Sql Injection |   21 |      15 |     3 |      3 | 2017-07-06 10:39:00 | 2017-07-06 10:41:00 | 2017-07-06 10:41:00 | 2017-07-06 10:41:00 |
| thursday  | Web Attack - XSS           |  399 |     279 |    66 |     54 | 2017-07-06 10:15:00 | 2017-07-06 10:30:00 | 2017-07-06 10:33:00 | 2017-07-06 10:35:00 |
| tuesday   | BENIGN                     | 2489 |    1745 |   372 |    372 | 2017-07-04 09:00:00 | 2017-07-04 14:32:00 | 2017-07-04 15:45:00 | 2017-07-04 16:59:00 |
| tuesday   | FTP-Patator                | 1395 |     973 |   217 |    205 | 2017-07-04 09:18:00 | 2017-07-04 09:59:00 | 2017-07-04 10:09:00 | 2017-07-04 10:17:00 |
| tuesday   | SSH-Patator                | 1291 |     914 |   176 |    201 | 2017-07-04 14:00:00 | 2017-07-04 14:43:00 | 2017-07-04 14:51:00 | 2017-07-04 14:59:00 |
| wednesday | BENIGN                     | 2485 |    1740 |   372 |    373 | 2017-07-05 09:00:00 | 2017-07-05 14:32:00 | 2017-07-05 15:46:00 | 2017-07-05 16:59:00 |
| wednesday | DoS Hulk                   | 1793 |    1280 |   226 |    287 | 2017-07-05 10:42:00 | 2017-07-05 11:13:00 | 2017-07-05 11:18:00 | 2017-07-05 11:23:00 |
| wednesday | DoS slowloris              | 1198 |     838 |   193 |    167 | 2017-07-05 09:42:00 | 2017-07-05 10:03:00 | 2017-07-05 10:08:00 | 2017-07-05 10:11:00 |

Chevauchement exact entre splits (espace des variables retenues) : {'train&val': 0, 'train&test': 0, 'val&test': 0} (doit être 0).

### Régime de données réduites (train, emboîté)
Même sous-ensemble pour le générateur **et** le classifieur (règle d'équité).
|               |   frac0.01 |   frac0.05 |   frac0.1 |
|:--------------|-----------:|-----------:|----------:|
| BENIGN        |       3000 |       3000 |      3000 |
| Bot           |         20 |         38 |        77 |
| PortScan      |         20 |         53 |       106 |
| DDoS          |         20 |         53 |       105 |
| Web Attack    |         20 |         46 |        92 |
| FTP-Patator   |         20 |         49 |        97 |
| SSH-Patator   |         20 |         46 |        91 |
| DoS slowloris |         20 |         42 |        84 |
| DoS Hulk      |         20 |         64 |       128 |

## 5. Variables (60 retenues / 77 candidates)
- Identifiants supprimés : ['destination_ip', 'destination_port', 'flow_id', 'protocol', 'source_ip', 'source_port', 'timestamp']
- Constantes (train) : ['bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'cwe_flag_count', 'fwd_avg_bytes_bulk', 'fwd_avg_packets_bulk', 'fwd_avg_bulk_rate', 'bwd_avg_bytes_bulk', 'bwd_avg_packets_bulk', 'bwd_avg_bulk_rate']
- Identiques à une autre colonne : {'avg_fwd_segment_size': 'fwd_packet_length_mean', 'avg_bwd_segment_size': 'bwd_packet_length_mean', 'fwd_header_length_1': 'fwd_header_length', 'subflow_fwd_packets': 'total_fwd_packets', 'subflow_fwd_bytes': 'total_length_of_fwd_packets', 'subflow_bwd_packets': 'total_backward_packets', 'subflow_bwd_bytes': 'total_length_of_bwd_packets'}
- Élaguées par corrélation : aucune

Ordre (= ordre des cellules 8×8, ligne par ligne) : `flow_duration, flow_bytes_s, flow_packets_s, down_up_ratio, average_packet_size, total_fwd_packets, total_length_of_fwd_packets, fwd_packet_length_max, fwd_packet_length_min, fwd_packet_length_mean, fwd_packet_length_std, fwd_header_length, fwd_packets_s, init_win_bytes_forward, act_data_pkt_fwd, min_seg_size_forward, total_backward_packets, total_length_of_bwd_packets, bwd_packet_length_max, bwd_packet_length_min, bwd_packet_length_mean, bwd_packet_length_std, bwd_header_length, bwd_packets_s, init_win_bytes_backward, min_packet_length, max_packet_length, packet_length_mean, packet_length_std, packet_length_variance, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min, fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min, bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min, fwd_psh_flags, fin_flag_count, syn_flag_count, rst_flag_count, psh_flag_count, ack_flag_count, urg_flag_count, ece_flag_count, active_mean, active_std, active_max, active_min, idle_mean, idle_std, idle_max, idle_min`

## 6. Normalisation
log1p signé (52 variables) puis min-max → [-1.0, 1.0], **ajusté sur le train uniquement**.
Erreur relative max de l'inversion : 3.16e-07.
Part moyenne de valeurs hors [-1,1] : val 0.003 %, test 0.006 % (non écrêtées).

## 7. Représentation 8×8
60 cellules réelles + 4 de remplissage (valeur 0.0, masque dans `artifacts/matrix_layout.json`).
Encodage de visualisation **sans** relation spatiale physique ; l'évaluation reste tabulaire.

## 8. Contraintes de plausibilité (mesurées sur le train réel)
131 contraintes vérifiées ; 2 violées par > 1 % des flux réels
(**à ne pas imposer** au générateur) : ['rel:bwd_pkt_len_max<=total_len_bwd', 'rel:fwd_pkt_len_max<=total_len_fwd'].
Lignes réelles violant ≥ 1 contrainte fiable : 0.709 %.
Détail : `reports/constraint_violations_real_train.csv`.

## 9. Audit des quasi-doublons (espace normalisé, distance euclidienne)
τ par classe = centile 1.0 des distances validation → train (`artifacts/nn_thresholds.json`).
|               |      τ |   méd. val→train |   méd. test→train | quasi-doublons test (≤τ)   |
|:--------------|-------:|-----------------:|------------------:|:---------------------------|
| BENIGN        | 0.9895 |            2.103 |             2.076 | 1.07 %                     |
| Bot           | 1.025  |            2.471 |             2.47  | 1.22 %                     |
| DDoS          | 1.246  |            2.445 |             2.46  | 1.90 %                     |
| DoS Hulk      | 1.061  |            2.405 |             2.404 | 0.70 %                     |
| DoS slowloris | 1.165  |            2.45  |             2.449 | 1.20 %                     |
| FTP-Patator   | 1.312  |            2.435 |             2.388 | 1.46 %                     |
| PortScan      | 1.183  |            2.44  |             2.47  | 2.27 %                     |
| SSH-Patator   | 1.28   |            2.517 |             2.515 | 2.49 %                     |
| Web Attack    | 1.058  |            2.562 |             2.48  | 0.00 %                     |

## 10. Limites connues
- Erreurs d'étiquetage et de calcul de variables signalées dans CIC-IDS2017 (Engelen et al., 2021 ; Lanvin et al., 2022).
- Trafic simulé en laboratoire sur 5 jours : la généralisation à un réseau réel n'est pas démontrée.
- Partition intra-fenêtre : les derniers 15 % d'une attaque peuvent différer de son début (dérive volontairement non corrigée).
- Résolution temporelle des timestamps limitée ; heuristique de correction 12 h documentée dans `pipeline_report.json`.
- Une seule capture : pas de test externe sur un autre réseau.
