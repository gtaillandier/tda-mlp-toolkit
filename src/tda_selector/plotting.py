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


def plot_selection_comparison(
    features: np.ndarray,
    fps_indices: Sequence[int],
    kmeans_indices: Sequence[int],
    output_path: str | Path,
    title: Optional[str] = None,
    seed: int = 42,
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

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(coords[:, 0], coords[:, 1], s=8, c="lightgray", alpha=0.6, label=f"All structures ({len(coords)})", zorder=1)
    ax.scatter(
        coords[km, 0], coords[km, 1], s=70, marker="s", facecolors="none", edgecolors="tab:blue",
        linewidths=1.4, label=f"K-means ({len(km)})", zorder=2,
    )
    ax.scatter(
        coords[fps, 0], coords[fps, 1], s=35, marker="o", c="tab:red", edgecolors="black",
        linewidths=0.4, label=f"FPS ({len(fps)})", zorder=3,
    )
    if len(fps):
        ax.scatter(
            coords[fps[0], 0], coords[fps[0], 1], s=220, marker="*", c="gold", edgecolors="black",
            linewidths=0.8, label="FPS start", zorder=4,
        )

    ax.set_xlabel(f"PC1 ({ratio[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%} variance)")
    ax.set_title(title or f"FPS vs K-means selection (common: {n_common})")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
