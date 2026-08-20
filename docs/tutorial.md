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
