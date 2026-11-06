"""Figures d'analyse exploratoire et de contrôle (matplotlib uniquement)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLIT_COLORS = {"train": "#2a6fdb", "val": "#e0a400", "test": "#d1495b", "drop": "#999999"}
KEY_FEATURES = ["flow_duration", "total_fwd_packets", "total_length_of_fwd_packets",
                "fwd_packet_length_mean", "flow_iat_mean", "flow_bytes_s",
                "init_win_bytes_forward", "packet_length_std"]


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def class_distribution(before: dict, after: dict, path: str):
    labs = sorted(set(before) | set(after), key=lambda l: -before.get(l, 0))
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.2, [before.get(l, 0) for l in labs], 0.4, label="brut", color="#b0b8c4")
    ax.bar(x + 0.2, [after.get(l, 0) for l in labs], 0.4, label="après nettoyage + dédoublonnage", color="#2a6fdb")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("nombre de flux (log)"); ax.legend(frameon=False)
    ax.set_title("Distribution des classes (label original)")
    return _save(fig, path)


def split_timeline(df: pd.DataFrame, path: str, max_points: int = 60000, seed: int = 0):
    """Une ligne par fenêtre (jour, label), points colorés par split : montre les coupures."""
    use_ts = df["ts"].notna().all()
    d = df.sample(min(len(df), max_points), random_state=seed)
    windows = sorted(d.groupby(["day", "label_orig"]).groups.keys(),
                     key=lambda k: (["monday", "tuesday", "wednesday", "thursday", "friday", "unknown"].index(k[0])
                                    if k[0] in ["monday", "tuesday", "wednesday", "thursday", "friday"] else 9, k[1]))
    fig, ax = plt.subplots(figsize=(12, 0.35 * len(windows) + 1.5))
    for i, w in enumerate(windows):
        g = d[(d["day"] == w[0]) & (d["label_orig"] == w[1])]
        if use_ts:
            t = (g["ts"] - g["ts"].dt.normalize()).dt.total_seconds() / 3600
        else:
            t = g["row_in_file"]
        ax.scatter(t, np.full(len(g), i) + np.random.default_rng(i).uniform(-0.25, 0.25, len(g)),
                   s=2, c=g["split"].map(SPLIT_COLORS), alpha=0.6, linewidths=0)
    ax.set_yticks(range(len(windows))); ax.set_yticklabels([f"{a[:3]} · {b}" for a, b in windows], fontsize=7)
    ax.set_xlabel("heure de la journée" if use_ts else "rang dans le fichier")
    for s, c in SPLIT_COLORS.items():
        if s in d["split"].unique():
            ax.scatter([], [], c=c, s=20, label=s)
    ax.legend(frameon=False, loc="upper right", ncol=4)
    ax.set_title("Partition temporelle dans chaque fenêtre d'attaque")
    return _save(fig, path)


def feature_histograms(df: pd.DataFrame, feats: list[str], path: str, max_classes: int = 8):
    feats = [f for f in KEY_FEATURES if f in feats][:8] or feats[:8]
    top = df["label"].value_counts().index[:max_classes]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, f in zip(axes.flat, feats):
        for c in top:
            x = df.loc[df["label"] == c, f]
            x = np.sign(x) * np.log1p(np.abs(x))
            ax.hist(x, bins=50, histtype="step", density=True, label=c)
        ax.set_title(f, fontsize=9); ax.set_xlabel("log1p signé", fontsize=7)
    axes.flat[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Distributions (train) des variables clés par classe")
    return _save(fig, path)


def correlation_heatmap(df: pd.DataFrame, feats: list[str], path: str, n: int = 100000, seed: int = 0):
    s = df[feats].sample(min(len(df), n), random_state=seed)
    corr = s.rank().corr().fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=5)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman")
    ax.set_title("Corrélations (train, variables retenues)")
    return _save(fig, path)


def class_matrices(Xs: pd.DataFrame, labels: pd.Series, layout, path: str, n_examples: int = 3, seed: int = 0):
    """Matrice moyenne + exemples réels par classe ; cases de remplissage hachurées."""
    classes = sorted(labels.unique())
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), n_examples + 1, figsize=(1.8 * (n_examples + 1), 1.8 * len(classes)))
    axes = np.atleast_2d(axes)
    pad = layout.mask == 0
    for i, c in enumerate(classes):
        X = Xs[labels == c].to_numpy()
        mats = [X.mean(0)] + [X[j] for j in rng.choice(len(X), min(n_examples, len(X)), replace=False)]
        for j, ax in enumerate(axes[i]):
            ax.axis("off")
            if j >= len(mats):
                continue
            M = layout.to_matrix(mats[j][None])[0, 0]
            M = np.ma.array(M, mask=pad)
            ax.imshow(M, cmap="viridis", vmin=-1, vmax=1)
            ys, xs = np.where(pad)
            ax.scatter(xs, ys, marker="x", s=12, c="#888888")
            if j == 0:
                ax.set_title(f"{c}\n(moyenne)", fontsize=7)
            else:
                ax.set_title("exemple réel", fontsize=6)
    fig.suptitle("Encodage 8×8 (× = case de remplissage)", fontsize=9)
    return _save(fig, path)


def nn_histograms(nn_report_dists: dict, path: str):
    """nn_report_dists : {classe: {'val': array, 'test': array}}"""
    classes = sorted(nn_report_dists)
    cols = 4
    rows = int(np.ceil(len(classes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.8 * rows))
    for ax, c in zip(np.atleast_1d(axes).flat, classes):
        for k, col in (("val", "#e0a400"), ("test", "#d1495b")):
            d = nn_report_dists[c].get(k)
            if d is not None and len(d):
                ax.hist(np.log10(d + 1e-6), bins=40, histtype="step", color=col, label=k, density=True)
        ax.axvline(np.log10(nn_report_dists[c]["tau"] + 1e-6), color="k", ls="--", lw=1, label="τ")
        ax.set_title(c, fontsize=8); ax.set_xlabel("log10 distance au PPV train", fontsize=7)
    np.atleast_1d(axes).flat[0].legend(fontsize=6, frameon=False)
    for ax in list(np.atleast_1d(axes).flat)[len(classes):]:
        ax.axis("off")
    return _save(fig, path)
