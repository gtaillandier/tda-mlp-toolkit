from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def project_2d(features: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("scikit-learn is required for plotting (pip install tda-selector[plot])") from exc

    n_components = min(2, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(features)
    ratio = pca.explained_variance_ratio_
    if n_components < 2:
        coords = np.column_stack([coords, np.zeros((len(coords), 2 - n_components))])
        ratio = np.concatenate([ratio, np.zeros(2 - n_components)])
    return coords, ratio


def nearest_pick_labels(features: np.ndarray, picks: Sequence[int]) -> np.ndarray:
    labels = np.empty(len(features), dtype=int)
    centers = features[np.asarray(picks, dtype=int)]
    for start in range(0, len(features), 4096):
        chunk = features[start : start + 4096]
        d = np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=2)
        labels[start : start + len(chunk)] = np.argmin(d, axis=1)
    return labels


def _cluster_colors(labels: np.ndarray):
    import matplotlib.pyplot as plt

    uniq, inv = np.unique(labels, return_inverse=True)
    cmap = plt.get_cmap("tab20" if len(uniq) <= 20 else "turbo", max(len(uniq), 1))
    return cmap(inv), dict(zip(uniq.tolist(), cmap(np.arange(len(uniq)))))


OKABE_ITO_BLUE = "#0072B2"
OKABE_ITO_VERMILLION = "#D55E00"
DATASET_GREY = "#C8C8C8"
FIGURE_SIZES = {"single": (3.5, 3.0), "double": (7.0, 4.6)}

PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
}


def plot_selection_comparison(
    features: np.ndarray,
    fps_indices: Sequence[int],
    kmeans_indices: Sequence[int],
    output_path: str | Path,
    title: Optional[str] = None,
    seed: int = 42,
    cluster_labels: Optional[np.ndarray] = None,
    figure_width: str = "single",
    show_fps_start: bool = False,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting (pip install tda-selector[plot])") from exc

    coords, ratio = project_2d(features, seed=seed)
    fps = np.asarray(fps_indices, dtype=int)
    km = np.asarray(kmeans_indices, dtype=int)
    scale = 1.0 if figure_width == "single" else 1.5

    with plt.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=FIGURE_SIZES[figure_width])

        if cluster_labels is None:
            ax.scatter(
                coords[:, 0], coords[:, 1], s=3 * scale, c=DATASET_GREY, linewidths=0,
                label=f"All structures ($n$ = {len(coords)})", zorder=1, rasterized=True,
            )
            km_style = dict(facecolors="none", edgecolors=OKABE_ITO_BLUE)
        else:
            labels = np.asarray(cluster_labels)
            point_colors, by_label = _cluster_colors(labels)
            ax.scatter(
                coords[:, 0], coords[:, 1], s=3 * scale, c=point_colors, alpha=0.5, linewidths=0,
                label=f"All structures ($n$ = {len(coords)})", zorder=1, rasterized=True,
            )
            km_style = dict(facecolors=[by_label[l] for l in labels[km].tolist()], edgecolors="black")

        ax.scatter(
            coords[km, 0], coords[km, 1], s=20 * scale, marker="s", linewidths=0.8,
            label=f"K-means ($n$ = {len(km)})", zorder=2, **km_style,
        )
        ax.scatter(
            coords[fps, 0], coords[fps, 1], s=9 * scale, marker="o", c=OKABE_ITO_VERMILLION,
            linewidths=0, label=f"FPS ($n$ = {len(fps)})", zorder=3,
        )
        if show_fps_start and len(fps):
            ax.scatter(
                coords[fps[0], 0], coords[fps[0], 1], s=28 * scale, marker="o", facecolors="none",
                edgecolors="black", linewidths=0.8, label="FPS start", zorder=4,
            )

        ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")
        if title:
            ax.set_title(title)
        n_entries = 4 if show_fps_start else 3
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n_entries, handletextpad=0.3,
            columnspacing=1.0, borderaxespad=0.3, markerscale=1.3,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return out
