"""
Parcellate ADNI 4D volumes with Schaefer 200 atlas.

Input:  all_4d_downsampled.pt  (812, 45, 54, 45, 140)
Output: adni_parcellated_schaefer200.pt  (812, 140, 200)

Indices aligned with index_to_name.json (subject 0 = row 0).

Usage:
    python3 scripts/parcellate_adni.py
"""

import torch
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_img
from pathlib import Path

DATA_ROOT = Path("/sci/nosnap/arieljaffe/sagi.nathan/shared_fmri_data")
LAB_DIR = Path("/sci/labs/arieljaffe/dan.abergel1")
OUTPUT_PATH = LAB_DIR / "data" / "adni_parcellated_schaefer200.pt"


def main():
    # 1. Load 4D volumes
    pt_path = DATA_ROOT / "all_4d_downsampled.pt"
    print(f"Loading {pt_path} ...")
    data = torch.load(pt_path, weights_only=True)
    N, X, Y, Z, T = data.shape
    print(f"  Shape: ({N}, {X}, {Y}, {Z}, {T})")

    # 2. Data affine — standard MNI152 at 4mm isotropic
    data_affine = np.array([
        [-4.,  0.,  0.,  90.],
        [ 0.,  4.,  0., -126.],
        [ 0.,  0.,  4.,  -72.],
        [ 0.,  0.,  0.,    1.]
    ])

    # 3. Fetch Schaefer 200 atlas and resample to data grid
    print("Fetching Schaefer 200 atlas ...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)
    atlas_img = nib.load(atlas.maps)
    atlas_resampled = resample_img(
        atlas_img,
        target_affine=data_affine,
        target_shape=(X, Y, Z),
        interpolation="nearest",
    )
    atlas_labels = atlas_resampled.get_fdata().astype(int)

    regions_found = np.unique(atlas_labels)
    regions_found = regions_found[regions_found > 0]
    print(f"  Resampled atlas: {atlas_labels.shape}, {len(regions_found)}/200 regions")

    if len(regions_found) < 190:
        print("  WARNING: many regions missing — affine may be wrong!")

    # 4. Precompute region masks
    masks = {}
    for r in range(1, 201):
        mask = atlas_labels == r
        if mask.sum() > 0:
            masks[r] = torch.from_numpy(mask)

    # 5. Parcellate
    print(f"Parcellating {N} subjects ...")
    result = torch.zeros(N, T, len(masks), dtype=torch.float32)
    region_order = sorted(masks.keys())

    for i in range(N):
        if i % 100 == 0:
            print(f"  {i}/{N}")
        vol = data[i].float()  # (X, Y, Z, T)
        for j, r in enumerate(region_order):
            voxels = vol[masks[r]]  # (n_voxels, T)
            result[i, :, j] = voxels.mean(dim=0)

    # 6. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Shape: {result.shape}  (N={N}, T={T}, R={len(masks)})")

    if len(masks) < 200:
        print(f"WARNING: only {len(masks)}/200 regions had voxels. "
              f"Missing: {sorted(set(range(1, 201)) - set(region_order))}")


if __name__ == "__main__":
    main()
