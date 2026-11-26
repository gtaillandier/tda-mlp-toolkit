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


def plot_selection_comparison(
    features: np.ndarray,
    fps_indices: Sequence[int],
    kmeans_indices: Sequence[int],
    output_path: str | Path,
    title: Optional[str] = None,
    seed: int = 42,
    cluster_labels: Optional[np.ndarray] = None,
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
    n_common = len(np.intersect1d(fps, km))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    if cluster_labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=8, c="lightgray", alpha=0.6, label=f"All structures ({len(coords)})", zorder=1)
        ax.scatter(
            coords[km, 0], coords[km, 1], s=70, marker="s", facecolors="none", edgecolors="tab:blue",
            linewidths=1.4, label=f"K-means ({len(km)})", zorder=2,
        )
        fps_style = dict(c="tab:red", edgecolors="black", linewidths=0.4)
    else:
        labels = np.asarray(cluster_labels)
        point_colors, by_label = _cluster_colors(labels)
        n_clusters = len(by_label)
        ax.scatter(
            coords[:, 0], coords[:, 1], s=10, c=point_colors, alpha=0.45,
            label=f"All structures ({len(coords)}, {n_clusters} clusters)", zorder=1,
        )
        ax.scatter(
            coords[km, 0], coords[km, 1], s=80, marker="s", c=[by_label[l] for l in labels[km].tolist()],
            edgecolors="black", linewidths=1.2, label=f"K-means ({len(km)})", zorder=2,
        )
        fps_style = dict(c="white", edgecolors="black", linewidths=1.2)
    ax.scatter(coords[fps, 0], coords[fps, 1], s=40, marker="o", label=f"FPS ({len(fps)})", zorder=3, **fps_style)
    if len(fps):
        ax.scatter(
            coords[fps[0], 0], coords[fps[0], 1], s=220, marker="*", c="gold", edgecolors="black",
            linewidths=0.8, label="FPS start", zorder=4,
        )

    ax.set_xlabel(f"PC1 ({ratio[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%} variance)")
    ax.set_title(title or f"FPS vs K-means selection (common: {n_common})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
