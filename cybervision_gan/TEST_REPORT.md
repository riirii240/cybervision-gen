# Rapport de vérification — Modèle A

Date : 14 novembre 2026. Environnement local CPU macOS, Python 3.11,
PyTorch 2.2.2, NumPy 1.26.4 et pytest 9.2.0.

## Vérifications réellement exécutées

Commande :

```bash
.venv/bin/python -m pytest -q cybervision_gan/tests
```

Résultat : **5 tests réussis**.

- forme, plage `tanh` et gradients du générateur ;
- forme et absence de sigmoid du critique ;
- gradient penalty scalaire, finie, positive et différentiable ;
- génération par mini-batches en forme `(N,63)` ;
- conversion `(N,1,8,8)`, padding `(7,7)=0` et rejet des dimensions invalides.

Le notebook a aussi été parsé comme JSON, puis chaque cellule Python a été compilée
individuellement. Un smoke test d'intégration CPU sur 44 lignes artificielles
équilibrées (11 classes) a réellement exécuté une mise à jour WGAN-GP, calculé les
métriques, écrit `best.pt`/`latest.pt`, puis rechargé le checkpoint avec sa
configuration. Les nombres affichés par ce smoke test ne sont pas des résultats du
projet et ne doivent pas être publiés.

## Points non vérifiables dans ce dépôt

- Les Parquet privés `train_scaled.parquet` et `val_scaled.parquet` ne sont pas
  versionnés : le chargement complet et les effectifs réels n'ont pas été rejoués.
- Aucun entraînement complet GPU Colab n'a été lancé ; convergence, durée, mémoire
  et qualité statistique finale restent à mesurer.
- Les violations physiques du synthétique réel ne peuvent être rapportées avant un
  entraînement complet. Le notebook conserve bien l'export inverse non post-traité.
- Le split test n'a volontairement pas été chargé ou évalué.
- Trois corrélations rang/temps du rapport de la personne 1 sont sous 0,9 ; les
  résultats demeurent provisoires jusqu'à résolution ou validation de ce point.
