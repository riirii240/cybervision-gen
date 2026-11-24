# Catalogue des contraintes (Personne 3 — étape 1)

Données : train réel, 1 560 917 connexions (unités brutes). Validation et test non utilisés.
Outil : `cvgen_data.constraints.check_rows` (Personne 1). Seuil : 1.00 %.
Contrôle : écart maximal avec les taux publiés par la Personne 1 = 0.00e+00.

## Règle de décision
- **gardée** : cassée par au plus 1.00 % des vraies connexions, au total ET dans chaque classe cible ;
- **à surveiller** : fiable au total, mais cassée par plus de 1.00 % des connexions d'une classe cible (Bot, PortScan, DoS Slowhttptest) ;
- **écartée** : cassée par plus de 1.00 % des vraies connexions au total (ne pas l'imposer au générateur).

## Nombre de règles par famille et par décision

| famille    |   gardée |   à surveiller |   écartée |   total |
|:-----------|---------:|---------------:|----------:|--------:|
| binaire    |        7 |              0 |         0 |       7 |
| entier     |       32 |              0 |         0 |      32 |
| ordre      |       24 |              0 |         0 |      24 |
| positivité |       71 |              0 |         0 |      71 |
| relation   |       10 |              2 |         1 |      13 |
| total      |      144 |              2 |         1 |     147 |

## Règles non gardées (détail)

| regle                            | famille   | decision     | global   | Bot    | PortScan   | DoS Slowhttptest   |
|:---------------------------------|:----------|:-------------|:---------|:-------|:-----------|:-------------------|
| rel:bwd_iat_total<=flow_duration | relation  | à surveiller | 0.99 %   | 0.00 % | 3.39 %     | 8.65 %             |
| rel:flow_iat_max<=flow_duration  | relation  | à surveiller | 0.31 %   | 0.00 % | 3.39 %     | 8.38 %             |
| rel:fwd_iat_total<=flow_duration | relation  | écartée      | 2.49 %   | 0.00 % | 3.39 %     | 14.44 %            |

Toutes les règles et tous les taux par classe : `catalogue_contraintes.csv`.

## Décision pour le modèle B

Le modèle B impose uniquement les **144 règles gardées**. Les règles « à surveiller » et « écartée » ne sont pas imposées : les imposer pousserait le générateur à produire des connexions plus « propres » que les vraies connexions des classes cibles. Elles restent mesurées lors de la comparaison des modèles A, B et C.
