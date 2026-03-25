"""
Verify ADNI data structure before running BNT.

Usage:
    python3 scripts/verify_adni_data.py
"""

import torch
import json

DATA_ROOT = "/sci/nosnap/arieljaffe/sagi.nathan/shared_fmri_data"

# --- Tensor ---
LAB_DIR = "/sci/labs/arieljaffe/dan.abergel1"
pt_path = f"{LAB_DIR}/data/adni_parcellated_schaefer200.pt"
t = torch.load(pt_path, weights_only=True)
print(f"Tensor shape: {t.shape}, dtype: {t.dtype}, NaN count: {t.isnan().sum().item()}")

# --- Labels ---
with open(f"{DATA_ROOT}/index_to_name.json") as f:
    idx = json.load(f)
with open(f"{DATA_ROOT}/imageID_to_labels.json") as f:
    lab = json.load(f)

print(f"\nindex_to_name: {len(idx)} entries, keys sample: {list(idx.keys())[:3]}")
print(f"First entry: {idx['0']}")
print(f"imageID_to_labels: {len(lab)} entries")

# First matching label
first_id = idx["0"]["image_id"]
print(f"\nLabels for {first_id}: {lab.get(first_id, 'NOT FOUND')}")

# Count valid subjects per label
print("\nValid subjects per label:")
COLUMNS = [
    "Sex_Binary", "Age", "MMSE Total Score",
    "CDR_Binary", "degradation_binary_1year", "degradation_binary_3years",
]
for col in COLUMNS:
    valid = sum(
        1 for i in sorted(idx.keys(), key=int)
        if idx[i]["image_id"] in lab
        and col in lab[idx[i]["image_id"]]
        and lab[idx[i]["image_id"]][col] is not None
    )
    print(f"  {col}: {valid}/{len(idx)} valid")
