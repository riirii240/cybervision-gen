"""Crée le zip du code à déposer sur le Drive pour Colab.

Contient : cybervision_data (sans les données), cybervision_gan, cybervision_plausibility
(sans les checkpoints). Exclut : .venv, .git, parquets, checkpoints, caches.

Lancer depuis la racine de la repo :
    python cybervision_plausibility/creer_zip_colab.py
Puis déposer cybervision-gen.zip dans « Mon Drive / CyberVision_P3 ».
"""
from __future__ import annotations

import zipfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DOSSIERS = ["cybervision_data", "cybervision_gan", "cybervision_plausibility"]
EXCLURE_DOSSIERS = {".venv", ".git", "__pycache__", ".pytest_cache", "sorties", "raw", "interim"}
EXCLURE_EXTENSIONS = {".parquet", ".pt", ".pth", ".ckpt", ".zip", ".pyc", ".joblib"}


def a_garder(chemin: Path) -> bool:
    morceaux = set(chemin.relative_to(RACINE).parts)
    return not (morceaux & EXCLURE_DOSSIERS) and chemin.suffix not in EXCLURE_EXTENSIONS


def main() -> None:
    sortie = RACINE / "cybervision-gen.zip"
    n = 0
    with zipfile.ZipFile(sortie, "w", zipfile.ZIP_DEFLATED) as zf:
        for nom in DOSSIERS:
            for chemin in sorted((RACINE / nom).rglob("*")):
                if chemin.is_file() and a_garder(chemin):
                    zf.write(chemin, Path("cybervision-gen") / chemin.relative_to(RACINE))
                    n += 1
    print(f"{n} fichiers -> {sortie} ({sortie.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
