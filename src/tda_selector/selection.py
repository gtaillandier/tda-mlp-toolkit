from typing import Optional

import numpy as np
from tqdm import tqdm


def _start_index(features: np.ndarray, mode: str) -> int:
    if mode == "centroid":
        centroid = features.mean(axis=0)
        return int(np.argmin(np.linalg.norm(features - centroid, axis=1)))
    if mode == "random":
        return int(np.random.randint(len(features)))
    return int(mode)


def farthest_point_sampling(
    features: np.ndarray,
    n_select: Optional[int] = None,
    min_distance: Optional[float] = None,
    start: str = "centroid",
) -> tuple[list[int], list[float]]:
    if n_select is None and min_distance is None:
        raise ValueError("Specify at least one of --n-select or --min-distance")

    n = len(features)
    s0 = _start_index(features, start)
    selected = [s0]
    min_dists = np.linalg.norm(features - features[s0], axis=1).astype(np.float64)
    min_dists[s0] = -np.inf
    history: list[float] = []

    cap = n_select if n_select is not None else n
    pbar = tqdm(total=cap, desc="Selecting structures")
    pbar.update(1)

    while len(selected) < n:
        next_idx = int(np.argmax(min_dists))
        next_dist = float(min_dists[next_idx])
        if min_distance is not None and next_dist < min_distance:
            break
        if n_select is not None and len(selected) >= n_select:
            break

        selected.append(next_idx)
        history.append(next_dist)
        pbar.update(1)

        new_d = np.linalg.norm(features - features[next_idx], axis=1)
        np.minimum(min_dists, new_d, out=min_dists)
        min_dists[next_idx] = -np.inf

    pbar.close()
    return selected, history


def kmeans_selection(
    features: np.ndarray,
    n_select: int,
    seed: int = 42,
    fill_empty: bool = True,
) -> tuple[list[int], dict]:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise ImportError("scikit-learn is required for kmeans mode") from exc

    n = len(features)
    if n_select > n:
        raise ValueError("n_select cannot exceed number of structures")

    km = MiniBatchKMeans(n_clusters=n_select, random_state=seed, n_init=3, batch_size=min(10000, n), max_iter=300)
    labels = km.fit_predict(features)
    centers = km.cluster_centers_

    selected: list[int] = []
    empty = 0
    for c in range(n_select):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            empty += 1
            continue
        d = np.linalg.norm(features[idx] - centers[c], axis=1)
        selected.append(int(idx[np.argmin(d)]))

    if empty and fill_empty:
        remaining = [i for i in range(n) if i not in set(selected)]
        if not selected:
            selected.append(remaining.pop(0))
        while len(selected) < n_select and remaining:
            pool = features[np.array(remaining)]
            sel = features[np.array(selected)]
            d = np.linalg.norm(pool[:, None, :] - sel[None, :, :], axis=2).min(axis=1)
            pick = int(np.argmax(d))
            selected.append(remaining.pop(pick))

    return selected, {
        "inertia": float(km.inertia_),
        "mean_dist_to_centroid": float(np.mean(np.linalg.norm(features[selected] - centers[labels[selected]], axis=1))),
        "empty_clusters": empty,
    }
