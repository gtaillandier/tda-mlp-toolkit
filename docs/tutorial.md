# Tutorial

1. Prepare an ASE-readable multi-frame trajectory file.
2. Run selection:

```bash
python tda_selector.py your_traj.xyz --n-select 20 --output-dir run1
```

3. Inspect:
- run1/selected_indices.txt
- run1/selected_structures.xyz
- run1/summary.txt

4. Reuse cached features:

```bash
python tda_selector.py your_traj.xyz --n-select 20 --features-cache feats.npy --output-dir run2
```

5. Compare FPS and K-means on the same plot (needs `pip install tda-selector[plot]`):

```bash
python plot_selection_comparison.py your_traj.xyz --n-select 20 --features-cache feats.npy --output compare.png
```

The plot shows the full dataset (grey) in the first two PCA components of the TDA features, with FPS picks (red, start point as a star) and K-means picks (squares) overlaid. Structures and K-means picks are coloured by K-means cluster; with `--kmeans-indices` each structure takes the colour of its nearest K-means pick. Use `--no-cluster-colors` for the plain grey view. To plot existing runs instead of re-selecting, pass `--fps-indices run1/selected_indices.txt --kmeans-indices run_km/selected_indices.txt`.
