# cybervision_plausibility — Personne 3 (modèles B et C)

Contraintes de plausibilité (modèle B) et conditionnement renforcé (modèle C).
S'appuie sur `cybervision_data/` (Personne 1) et `cybervision_gan/` (Personne 2) sans les modifier.

## Étape 1 — Catalogue des contraintes

```bash
python cybervision_plausibility/catalogue_contraintes.py
```

Lit uniquement `train_raw.parquet` (train réel). Produit dans `resultats/` :
- `catalogue_contraintes.csv` : une ligne par règle (famille, décision, taux global et par classe) ;
- `catalogue_contraintes.md` : résumé lisible et justification des choix.

Décision : une règle est **gardée** si au plus 1 % des vraies connexions la cassent
(au total et dans chaque classe cible), **à surveiller** si elle est fiable au total
mais pas pour une classe cible, **écartée** sinon.
