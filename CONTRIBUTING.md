# Comment on travaille à 4 sur ce dépôt

## La règle d'or
**Personne ne pousse directement sur `main`.** Chacun travaille sur sa branche, puis ouvre une Pull Request (PR) qu'un autre membre relit.

## Une fois au début
```bash
git clone https://github.com/<proprietaire>/cybervision-gen.git
cd cybervision-gen
git config user.name  "Prénom Nom"
git config user.email "email-du-compte-github"
```

## Chaque fois que tu travailles (le cycle)
```bash
git switch main
git pull                                   # 1. récupérer le travail des autres
git switch -c p2/critique-wgan             # 2. nouvelle branche (ou: git switch p2/critique-wgan)
#   ... tu codes ...
git add models/wgan_gp_A/                   # 3. ajouter SEULEMENT tes fichiers
git commit -m "P2: critique MLP + gradient penalty"
git push -u origin p2/critique-wgan        # 4. pousser ta branche
```
5. Sur GitHub : **Compare & pull request** → remplir le modèle → demander une relecture → **Merge**.
6. Après la fusion : `git switch main && git pull`.

## Nommage des branches
`p1/...` données · `p2/...` modèle A · `p3/...` modèles B/C · `p4/...` évaluation · `docs/...` article

## Pour éviter les conflits
- Chacun modifie surtout **son dossier**. Pour toucher au dossier d'un autre, prévenez-le.
- Faites `git pull` sur `main` **avant** de créer une branche, et souvent pendant (`git merge main` dans votre branche).
- **Notebooks** : avant chaque commit, *Clear All Outputs* (VS Code : `...` en haut du notebook). Les sorties créent des conflits illisibles.
- Petits commits fréquents, message qui commence par `P1:`, `P2:`…

## Ce qu'on ne commit JAMAIS
CSV bruts, `.parquet`, poids `.pt/.pth`, mots de passe/tokens, adresses IP. (Le `.gitignore` bloque déjà la plupart.)
Les poids et données vont sur le Drive partagé `CyberVision/`.

## Partitions figées
`cybervision_data/data/processed/splits/` et `MANIFEST.json` ne changent plus après l'annonce du gel.
Toute modification = discussion de toute l'équipe + nouvelle version notée dans le README.

## En cas de conflit
VS Code ouvre le fichier avec **Accept Current / Accept Incoming / Accept Both**. Choisir, enregistrer, puis :
```bash
git add <fichier>
git commit
```
Si vous êtes perdus : ne forcez rien (`--force` interdit sur `main`), demandez à l'équipe.
