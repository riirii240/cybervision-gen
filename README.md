# CyberVision Gen

WGAN-GP conditionnel pour le trafic réseau minoritaire (CIC-IDS2017) — IA Générative Visuelle.
Équipe : TIRAR Safia · CHENNANI Soukaina · ZITOUNI Rihab · HAKIM Taha

| Dossier | Responsable | Contenu |
|---|---|---|
| `cybervision_data/` | Personne 1 (Rihab) | données : nettoyage, partitions figées, scaler, 8×8, audit, data card |
| *(à ajouter)* | Personnes 2, 3, 4 | chacun crée son dossier à la racine, via une branche + Pull Request |

## Installation
```bash
git clone https://github.com/<proprietaire>/cybervision-gen.git
cd cybervision-gen
python -m venv .venv
# Windows : .venv\Scripts\activate   |   Mac/Linux : source .venv/bin/activate
pip install -r cybervision_data/requirements.txt
pytest -q cybervision_data/tests
```

Les données (CSV, .parquet) ne sont pas dans Git : elles sont sur le Drive partagé `CyberVision/`.
Règles de travail : [CONTRIBUTING.md](CONTRIBUTING.md).
