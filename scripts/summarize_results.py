"""
Summarize BNT ADNI results from training_process.npy files.

Reads result/<label>/*/training_process.npy — one per run.
Extracts last epoch metrics and computes mean ± std over 5 runs.

No modifications to original BNT code required.

Usage:
    python3 scripts/summarize_results.py
"""

import sys
import numpy as np
from pathlib import Path


LABELS = [
    "Sex_Binary",
    "CDR_Binary",
    "degradation_binary_1year",
    "degradation_binary_2years",
    "degradation_binary_3years",
]

# Metrics from training_process.npy (val and test only)
METRICS = [
    ("Test Accuracy", "Test Acc"),
    ("micro precision", "Test Prec"),
    ("micro recall", "Test Recall"),
    ("micro F1", "Test F1"),
    ("Test AUC", "Test AUROC"),
    ("Test Sensitivity", "Test Sens"),
    ("Test Specificity", "Test Spec"),
    ("Val AUC", "Val AUROC"),
]


def fmt(vals):
    if len(vals) > 1:
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    elif len(vals) == 1:
        return f"{vals[0]:.4f}"
    return "N/A"


def main():
    result_dir = Path("result")
    if not result_dir.exists():
        print("No result/ directory found. Run training first.")
        sys.exit(1)

    print("=" * 130)
    print("BNT on ADNI — Results Summary (mean ± std over runs)")
    print("=" * 130)

    # Header
    header = f"  {'Label':<30} {'N':>3}"
    for _, display in METRICS:
        header += f"  {display:>16}"
    print(header)
    print(f"  {'-' * 125}")

    for label in LABELS:
        label_dir = result_dir / label
        if not label_dir.exists():
            print(f"  {label:<30}   - (no results)")
            continue

        npy_files = sorted(label_dir.glob("*/training_process.npy"))
        if not npy_files:
            print(f"  {label:<30}   - (no .npy files)")
            continue

        # Collect last epoch from each run
        runs = []
        for npy_path in npy_files:
            data = np.load(npy_path, allow_pickle=True)
            if len(data) > 0:
                runs.append(data[-1])

        row = f"  {label:<30} {len(runs):>3}"
        for key, _ in METRICS:
            vals = [float(r[key]) for r in runs if key in r]
            row += f"  {fmt(vals):>16}"
        print(row)

    print(f"\n{'=' * 130}")


if __name__ == "__main__":
    main()
