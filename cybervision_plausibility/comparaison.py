"""Étape 6 (Personne 3) — Comparaison des modèles A, B et C.

Pour chaque modèle : génération comme la Personne 2 (5 000 exemples par classe cible,
1 000 par autre classe), puis mesures demandées par la fiche :
  1. plausibilité : % de connexions cassant au moins une règle (outil de la Personne 1),
     global / par famille / par classe cible, AVANT et APRÈS arrondi (post-traitement) ;
  2. ressemblance : distance de Wasserstein moyenne et écart de corrélations (mesures de la Personne 2) ;
  3. copies : copies exactes et quasi-copies vis-à-vis du train (outil et seuils de la Personne 1) ;
  4. matrices 8×8 : exemples réels et synthétiques ;
  5. analyse des échecs : règles encore les plus cassées ;
  6. fidélité à la classe (Random Forest indépendant) et diversité — justification du modèle C ;
  7. export des données synthétiques pour la Personne 4 (brutes, arrondies, réparées) ;
  8. ampleur des erreurs (avant réparation) et effort de la réparation (reparation.py).
Référence = train réel. La validation sert seulement de référence au Random Forest. Le test n'est jamais lu.

Lancer depuis la racine de la repo (checkpoints dans cybervision_plausibility/sorties/) :
    python cybervision_plausibility/comparaison.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from entrainement import RACINE, construire_modeles  # noqa: E402  (ajoute aussi les chemins d'import)
from cvgen_data.audit import near_duplicate_rate  # noqa: E402            (Personne 1)
from cvgen_data.constraints import check_rows  # noqa: E402               (Personne 1)
from cvgen_data.scaling import FlowScaler  # noqa: E402                   (Personne 1)
from cybervision_gan.evaluation.generate import generate_conditioned, vectors_to_matrices  # noqa: E402  (Personne 2)
from reparation import effort_reparation, reparer  # noqa: E402                                  (étape 6)

DOSSIER = Path(__file__).resolve().parent
ARTEFACTS = RACINE / "cybervision_data" / "artifacts"
DONNEES = RACINE / "cybervision_data" / "data" / "processed"
CATALOGUE = DOSSIER / "resultats" / "catalogue_contraintes.csv"
SORTIE = DOSSIER / "resultats" / "comparaison"

MODELES = {"A": "modele_A", "B": "modele_B_lambda0.1", "C": "modele_C_lambda0.1"}
CLASSES_CIBLES = ["Bot", "PortScan", "DoS Slowhttptest"]
N_CIBLE, N_AUTRE, SEED = 5000, 1000, 42
# Couleurs (palette de référence du skill dataviz) : réel en gris neutre, A/B/C = 3 premières teintes
COULEURS = {"Réel": "#8a8984", "A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def charger_reference():
    schema = json.loads((ARTEFACTS / "feature_schema.json").read_text(encoding="utf-8"))
    layout = json.loads((ARTEFACTS / "matrix_layout.json").read_text(encoding="utf-8"))
    labels = json.loads((ARTEFACTS / "label_encoding.json").read_text(encoding="utf-8"))
    variables = layout["features"]
    cat = pd.read_csv(CATALOGUE).set_index("regle")
    train_raw = pd.read_parquet(DONNEES / "train_raw.parquet", columns=variables + ["label"])
    train_scaled = pd.read_parquet(DONNEES / "train_scaled.parquet", columns=variables + ["label"])
    val_scaled = pd.read_parquet(DONNEES / "val_scaled.parquet", columns=variables + ["label"])
    return schema, layout, labels, variables, cat, train_raw, train_scaled, val_scaled


def charger_generateur(dossier: Path):
    try:
        etat = torch.load(dossier / "best.pt", map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        etat = torch.load(dossier / "best.pt", map_location="cpu")
    generateur, _ = construire_modeles(etat["config"])
    generateur.load_state_dict(etat["generator"])
    generateur.eval()
    return generateur, etat["config"], int(etat["epoch"])


def generer(generateur, cfg, labels: dict, variables: list[str]) -> pd.DataFrame:
    """Même protocole que la Personne 2 ; graine fixe pour que A, B et C soient comparables."""
    torch.manual_seed(SEED)
    morceaux = []
    for nom, index in labels.items():
        n = N_CIBLE if nom in CLASSES_CIBLES else N_AUTRE
        x = generate_conditioned(generateur, index, n, cfg["model"]["latent_dim"], "cpu")
        df = pd.DataFrame(x, columns=variables)
        df["label"], df["label_index"] = nom, index
        morceaux.append(df)
    return pd.concat(morceaux, ignore_index=True)


# ----------------------------------------------------------------------------
# 1 et 5. Plausibilité (outil de la Personne 1) et analyse des échecs
# ----------------------------------------------------------------------------
def plausibilite(brut: pd.DataFrame, cat: pd.DataFrame, schema: dict, nom: str):
    """Taux de connexions cassant >= 1 règle : global, par famille, par classe cible + taux par règle."""
    v = check_rows(brut, schema)
    gardees = cat.index[cat["decision"] == "gardée"]
    fiables_p1 = [r for r, f in schema["reliability"].items() if f == "reliable"]
    ligne = {"jeu": nom, "n": len(brut),
             "taux_144_regles_gardees": v[gardees].any(axis=1).mean(),
             "taux_regles_fiables_P1": v[fiables_p1].any(axis=1).mean()}
    for famille in sorted(cat["famille"].unique()):
        regles = [r for r in gardees if cat.loc[r, "famille"] == famille]
        if regles:
            ligne[f"famille_{famille}"] = v[regles].any(axis=1).mean()
    for classe in CLASSES_CIBLES:
        masque = (brut["label"] == classe).to_numpy()
        ligne[f"classe_{classe}"] = v.loc[masque, gardees].any(axis=1).mean()
    return ligne, v[gardees].mean().rename(nom)


def ampleur_erreurs(brut: pd.DataFrame, schema: dict, nom: str) -> dict:
    """Part des connexions dont l'erreur relative dépasse 10 % sur les relations d'égalité.

    Le taux strict « >= 1 règle cassée » sature à 100 % : un générateur continu ne produit
    jamais d'égalités exactes (min = moyenne = max dans 58 à 91 % des vraies connexions).
    Cette mesure compare donc les modèles sur l'AMPLEUR des écarts."""
    ligne = {"jeu": nom}
    for r in schema["relations"]:
        if r["kind"] not in ("prod", "sq"):
            continue
        c = [brut[x].to_numpy(float) for x in r["cols"]]
        cible = c[1] * c[2] if r["kind"] == "prod" else c[1] ** 2
        ligne[r["name"]] = float((np.abs(c[0] - cible) / (np.abs(cible) + 1.0) > 0.10).mean())
    return ligne


# ----------------------------------------------------------------------------
# 2, 3 et 6. Ressemblance, copies, diversité (par classe cible, espace normalisé)
# ----------------------------------------------------------------------------
def par_classe_cible(faux: pd.DataFrame, train_scaled: pd.DataFrame, variables, tau: dict, nom: str,
                     variante: str = "généré"):
    lignes = []
    for classe in CLASSES_CIBLES:
        reel = train_scaled.loc[train_scaled["label"] == classe, variables].to_numpy(np.float32)
        synt = faux.loc[faux["label"] == classe, variables].to_numpy(np.float32)
        wass = np.mean([wasserstein_distance(reel[:, j], synt[:, j]) for j in range(len(variables))])
        c_reel, c_synt = np.corrcoef(reel, rowvar=False), np.corrcoef(synt, rowvar=False)
        haut = np.triu(np.ones_like(c_reel, dtype=bool), k=1) & np.isfinite(c_reel) & np.isfinite(c_synt)
        copies = near_duplicate_rate(synt, reel, tau[classe])
        ecart_reel = reel.std(axis=0)
        utiles = ecart_reel > 1e-6
        lignes.append({
            "modele": nom, "variante": variante, "classe": classe,
            "wasserstein_moyen": wass,
            "ecart_correlations_moyen": float(np.abs(c_reel[haut] - c_synt[haut]).mean()),
            "copies_exactes": copies["exact_copy_rate"],
            "quasi_copies": copies["near_dup_rate"],
            "distance_mediane_plus_proche": copies["median_nn_dist"],
            "diversite_ratio_ecart_type": float((synt.std(axis=0)[utiles] / ecart_reel[utiles]).mean()),
        })
    return lignes


# ----------------------------------------------------------------------------
# 6. Fidélité à la classe : Random Forest indépendant, entraîné sur le train réel
# ----------------------------------------------------------------------------
def entrainer_juge(train_scaled: pd.DataFrame, variables, n_par_classe: int = 3000):
    echantillon = pd.concat([g.sample(n=min(len(g), n_par_classe), random_state=SEED)
                             for _, g in train_scaled.groupby("label")])
    juge = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    juge.fit(echantillon[variables].to_numpy(np.float32), echantillon["label"])
    return juge


def taux_reconnaissance(juge, df: pd.DataFrame, variables, nom: str) -> dict:
    predit = juge.predict(df[variables].to_numpy(np.float32))
    ligne = {"jeu": nom}
    for classe in CLASSES_CIBLES:
        masque = (df["label"] == classe).to_numpy()
        ligne[classe] = float((predit[masque] == classe).mean())
    ligne["moyenne_11_classes"] = float(pd.Series(predit == df["label"].to_numpy()).groupby(
        df["label"].to_numpy()).mean().mean())
    return ligne


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _style(ax):
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.spines["left"].set_color("#b4b2a9")
    ax.spines["bottom"].set_color("#b4b2a9")
    ax.tick_params(colors="#52514e")
    ax.yaxis.grid(True, color="#e6e4dd", linewidth=0.8)
    ax.set_axisbelow(True)


def figure_barres(tableau: pd.DataFrame, colonnes: list[str], noms: list[str], titre, fichier, pct=True):
    """Barres groupées : une couleur par jeu (Réel, A, B, C), valeurs écrites au-dessus."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    jeux = list(tableau.index)
    largeur = 0.8 / len(jeux)
    x = np.arange(len(colonnes))
    for i, jeu in enumerate(jeux):
        valeurs = tableau.loc[jeu, colonnes].to_numpy(float) * (100 if pct else 1)
        barres = ax.bar(x + (i - (len(jeux) - 1) / 2) * largeur, valeurs, largeur * 0.92,
                        color=COULEURS.get(jeu.split(" ")[0], "#8a8984"), label=jeu)
        for b, val in zip(barres, valeurs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{val:.0f}" if pct else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks(x, noms)
    ax.set_ylabel("%" if pct else "")
    ax.set_title(titre, loc="left", fontsize=12, color="#0b0b0b", pad=28)
    ax.legend(frameon=False, ncol=len(jeux), loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    ax.set_ylim(0, max(1e-9, ax.get_ylim()[1]) * 1.08)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fichier, dpi=160)
    plt.close(fig)


def figure_matrices(reels: dict, faux: dict, layout: dict, fichier):
    """Exemples 8×8 : une ligne par classe cible, colonnes Réel / A / B / C (case (7,7) = remplissage)."""
    colonnes = ["Réel"] + list(faux)
    fig, axes = plt.subplots(len(CLASSES_CIBLES), len(colonnes), figsize=(2.3 * len(colonnes), 2.4 * len(CLASSES_CIBLES)))
    for i, classe in enumerate(CLASSES_CIBLES):
        vecteurs = [reels[classe]] + [faux[m][classe] for m in faux]
        for j, (nom, v) in enumerate(zip(colonnes, vecteurs)):
            ax = axes[i, j]
            image = ax.imshow(vectors_to_matrices(v[None], layout)[0, 0], cmap="Blues", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(nom, fontsize=11)
            if j == 0:
                ax.set_ylabel(classe, fontsize=10)
    fig.colorbar(image, ax=axes, shrink=0.6, label="valeur normalisée [-1, 1]")
    fig.savefig(fichier, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare les modèles A, B et C (étape 6).")
    parser.add_argument("--sorties", type=Path, default=DOSSIER / "sorties",
                        help="dossier contenant modele_A/, modele_B_lambda0.1/, modele_C_lambda0.1/")
    parser.add_argument("--modeles", nargs="*", default=None,
                        help="optionnel : NOM=dossier (ex. A=modele_A B=modele_B_lambda0.1)")
    args = parser.parse_args()
    modeles = dict(m.split("=") for m in args.modeles) if args.modeles else MODELES
    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / "figures").mkdir(exist_ok=True)

    print("Chargement des données réelles (train, validation)…")
    schema, layout, labels, variables, cat, train_raw, train_scaled, val_scaled = charger_reference()
    scaler = FlowScaler.from_json(str(ARTEFACTS / "scaler.json"))
    tau = json.loads((ARTEFACTS / "nn_thresholds.json").read_text(encoding="utf-8"))["tau_per_class"]

    print("Plausibilité du train réel (référence)…")
    ref, regles_reel = plausibilite(train_raw, cat, schema, "Réel (train)")
    lignes_plaus, par_regle = [ref], [regles_reel]
    print(f"  Contrôle : taux réel sur les règles fiables de la Personne 1 = {100 * ref['taux_regles_fiables_P1']:.2f} %"
          " (attendu : 1,18 %)")

    print("Juge indépendant (Random Forest sur le train réel)…")
    juge = entrainer_juge(train_scaled, variables)
    lignes_fid = [taux_reconnaissance(juge, val_scaled, variables, "Réel (validation)")]

    lignes_classe, faux_normalises, epoques = [], {}, {}
    lignes_ampleur = [ampleur_erreurs(train_raw, schema, "Réel (train)")]
    lignes_effort = []
    for nom, sous_dossier in modeles.items():
        dossier = args.sorties / sous_dossier
        print(f"Modèle {nom} ({dossier.name})…")
        generateur, cfg, epoques[nom] = charger_generateur(dossier)
        normalise = generer(generateur, cfg, labels, variables)
        brut = scaler.inverse_transform(normalise[variables], postprocess=False)
        post = scaler.inverse_transform(normalise[variables], postprocess=True, schema=schema)
        repare = reparer(post[variables], schema)
        for df in (brut, post, repare):
            df["label"], df["label_index"] = normalise["label"].to_numpy(), normalise["label_index"].to_numpy()
        repare_normalise = scaler.transform(repare[variables]).astype(np.float32)
        repare_normalise["label"] = normalise["label"].to_numpy()

        export = dossier / "generated"                                  # 7. export pour la Personne 4
        export.mkdir(exist_ok=True)
        normalise.to_parquet(export / "synthetic_normalized_unprocessed.parquet", index=False)
        brut.to_parquet(export / "synthetic_raw_inverse.parquet", index=False)
        post.to_parquet(export / "synthetic_postprocessed.parquet", index=False)
        repare.to_parquet(export / "synthetic_repaired.parquet", index=False)

        lignes_ampleur.append(ampleur_erreurs(brut, schema, nom))
        lignes_effort.append({"modele": nom, **effort_reparation(post, repare, variables)})
        for variante, df in (("avant arrondi", brut), ("après arrondi", post), ("après réparation", repare)):
            ligne, regles = plausibilite(df, cat, schema, f"{nom} ({variante})")
            lignes_plaus.append(ligne)
            par_regle.append(regles)
        lignes_classe += par_classe_cible(normalise, train_scaled, variables, tau, nom)
        lignes_classe += par_classe_cible(repare_normalise, train_scaled, variables, tau, nom, "réparé")
        lignes_fid.append(taux_reconnaissance(juge, normalise, variables, nom))
        lignes_fid.append(taux_reconnaissance(juge, repare_normalise, variables, f"{nom} réparé"))
        faux_normalises[nom] = normalise

    # --- Tableaux ---
    plaus = pd.DataFrame(lignes_plaus).set_index("jeu")
    echecs = pd.concat(par_regle, axis=1)
    echecs.insert(0, "famille", cat.loc[echecs.index, "famille"])
    classe = pd.DataFrame(lignes_classe)
    fid = pd.DataFrame(lignes_fid).set_index("jeu")
    ampleur = pd.DataFrame(lignes_ampleur).set_index("jeu")
    effort = pd.DataFrame(lignes_effort).set_index("modele")
    ampleur.to_csv(SORTIE / "ampleur_erreurs.csv", float_format="%.6f")
    effort.to_csv(SORTIE / "effort_reparation.csv", float_format="%.6f")
    plaus.to_csv(SORTIE / "plausibilite.csv", float_format="%.6f")
    echecs.to_csv(SORTIE / "violations_par_regle.csv", float_format="%.6f")
    classe.to_csv(SORTIE / "ressemblance_copies_diversite.csv", index=False, float_format="%.6f")
    fid.to_csv(SORTIE / "fidelite_classe.csv", float_format="%.6f")

    # --- Figures ---
    familles = [c for c in plaus.columns if c.startswith("famille_")]
    relations_eq = list(ampleur.columns)
    figure_barres(ampleur.rename(index={"Réel (train)": "Réel"}), relations_eq,
                  ["variance = écart-type²" if "var" in r else ("total aller" if "fwd" in r else "total retour")
                   for r in relations_eq],
                  "Connexions dont l'erreur dépasse 10 % (sorties brutes du générateur)",
                  SORTIE / "figures" / "01_ampleur_erreurs.png")
    figure_barres(fid.loc[["Réel (validation)"] + list(modeles)].rename(index={"Réel (validation)": "Réel"}),
                  CLASSES_CIBLES, CLASSES_CIBLES,
                  "Fidélité à la classe : % reconnus par un Random Forest entraîné sur le train réel",
                  SORTIE / "figures" / "02_fidelite.png")
    rng = np.random.default_rng(SEED)
    reels = {c: train_scaled.loc[train_scaled["label"] == c, variables].to_numpy(np.float32)[
        rng.integers((train_scaled["label"] == c).sum())] for c in CLASSES_CIBLES}
    faux = {m: {c: df.loc[df["label"] == c, variables].to_numpy(np.float32)[0] for c in CLASSES_CIBLES}
            for m, df in faux_normalises.items()}
    figure_matrices(reels, faux, layout, SORTIE / "figures" / "03_matrices_8x8.png")

    # --- Résumé ---
    pct = lambda v: f"{100 * v:.1f} %"
    tab_plaus = plaus[["taux_144_regles_gardees", "taux_regles_fiables_P1"] + familles].map(pct)
    pires = {m: echecs[f"{m} (après arrondi)"].sort_values(ascending=False).head(5) for m in modeles}
    texte = [
        "# Comparaison des modèles A, B et C (Personne 3 — étape 6)", "",
        "Génération : 5 000 exemples par classe cible, 1 000 par autre classe, graine 42. "
        "Référence : train réel. Le test n'est jamais utilisé.", "",
        "Checkpoints (best.pt, critère de la Personne 2) : "
        + ", ".join(f"{m} = époque {e}" for m, e in epoques.items()) + ".", "",
        f"Contrôle de cohérence : train réel, règles fiables de la Personne 1 = {pct(ref['taux_regles_fiables_P1'])} (attendu 1,18 %).", "",
        "## 1. Plausibilité — % de connexions cassant au moins une règle", "",
        tab_plaus.to_markdown(), "",
        "Le taux strict sature avant réparation : un générateur continu ne produit pas d'égalités exactes "
        "(min = moyenne = max dans 58 à 91 % des vraies connexions). D'où les deux mesures suivantes.", "",
        "### 1 b. Ampleur des erreurs — part des connexions avec plus de 10 % d'erreur (sorties brutes)", "",
        ampleur.map(pct).to_markdown(), "",
        "### 1 c. Effort de la réparation (reparation.py) — moins = générateur plus cohérent", "",
        effort.map(pct).to_markdown(), "",
        "## 2-3. Ressemblance, copies, diversité (classes cibles, espace normalisé)", "",
        classe.set_index(["classe", "modele", "variante"]).sort_index().round(4).to_markdown(), "",
        "## 4. Matrices 8×8", "", "Voir `figures/03_matrices_8x8.png`.", "",
        "## 5. Analyse des échecs — 5 règles les plus cassées (après arrondi)", "",
    ]
    for m, s in pires.items():
        texte += [f"**Modèle {m}** : " + " ; ".join(f"`{r}` {pct(v)}" for r, v in s.items()), ""]
    texte += ["## 6. Fidélité à la classe (Random Forest indépendant)", "",
              "Part des exemples reconnus dans leur classe. Référence « Réel (validation) » = "
              "score du même juge sur les vraies connexions de validation.", "",
              fid.map(pct).to_markdown(), "",
              "## 7. Export pour la Personne 4", "",
              "`sorties/<modèle>/generated/` : `synthetic_normalized_unprocessed.parquet`, "
              "`synthetic_raw_inverse.parquet` (non post-traité), `synthetic_postprocessed.parquet` (arrondi), "
              "`synthetic_repaired.parquet` (arrondi + réparation de cohérence, recommandé)."]
    (SORTIE / "resume.md").write_text("\n".join(texte) + "\n", encoding="utf-8")
    print(f"\nTerminé. Résultats dans {SORTIE}")
    print(tab_plaus.to_string())


if __name__ == "__main__":
    main()
