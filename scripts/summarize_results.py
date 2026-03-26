"""
Summarize BNT ADNI training results from log files.

Parses the original log format (no code changes to training.py).
Extracts val/test metrics from the last epoch of each of 5 runs.

Original log format:
  Epoch[199/200] | Train Loss:... | Train Accuracy:...% | Test Loss:... |
  Test Accuracy:...% | Val AUC:... | Test AUC:... | Test Sen:... | LR:...

Metrics available in logs: Test Accuracy, Val AUC, Test AUC, Test Sensitivity
Additional metrics (precision, recall, F1) saved in training_process.npy

Usage:
    python3 scripts/summarize_results.py logs/train_adni_*.out
"""

import re
import sys
import glob
from collections import defaultdict
import numpy as np
from pathlib import Path


def parse_runs_from_log(filepath):
    """Parse all 5 runs from a single .out file using original log format.

    Detects run boundaries via Epoch[0/ lines.
    Returns: (label, list of metric dicts)
    """
    label = None
    runs = []
    current_last_epoch = None

    with open(filepath) as f:
        for line in f:
            if "Label:" in line:
                m = re.search(r'Label:\s*(\S+)', line)
                if m:
                    label = m.group(1)

            if "Epoch[" in line:
                if "Epoch[0/" in line and current_last_epoch is not None:
                    metrics = parse_original_epoch(current_last_epoch)
                    if metrics:
                        runs.append(metrics)
                current_last_epoch = line

    # Capture last run
    if current_last_epoch is not None:
        metrics = parse_original_epoch(current_last_epoch)
        if metrics:
            runs.append(metrics)

    return label, runs


def parse_original_epoch(line):
    """Extract metrics from the original epoch log format."""
    metrics = {}

    m = re.search(r'Test Accuracy:\s*([\d.]+)%', line)
    if m:
        metrics['Test_Accuracy'] = float(m.group(1))

    m = re.search(r'Val AUC:([\d.]+)', line)
    if m:
        metrics['Val_AUC'] = float(m.group(1))

    m = re.search(r'Test AUC:([\d.]+)', line)
    if m:
        metrics['Test_AUC'] = float(m.group(1))

    m = re.search(r'Test Sen:([\d.]+)', line)
    if m:
        metrics['Test_Sensitivity'] = float(m.group(1))

    return metrics if metrics else None


def load_npy_results(result_dir):
    """Load training_process.npy files from result/ directory.

    Each run saves to result/<unique_id>/training_process.npy.
    Returns list of metric dicts (last epoch of each run).
    """
    runs = []
    npy_files = sorted(Path(result_dir).glob("*/training_process.npy"))
    for npy_path in npy_files:
        data = np.load(npy_path, allow_pickle=True)
        if len(data) > 0:
            last = data[-1]  # last epoch
            runs.append(last)
    return runs


def fmt(vals):
    """Format mean ± std."""
    if len(vals) > 1:
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    elif len(vals) == 1:
        return f"{vals[0]:.4f}"
    return "N/A"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/summarize_results.py logs/train_adni_*.out")
        sys.exit(1)

    results = defaultdict(list)
    for filepath in sys.argv[1:]:
        label, runs = parse_runs_from_log(filepath)
        if label and runs:
            results[label].extend(runs)

    if not results:
        print("No results found.")
        sys.exit(1)

    # Metrics from the original log format
    metric_keys = [
        ('Test_Accuracy', 'Test Accuracy'),
        ('Test_AUC', 'Test AUC'),
        ('Val_AUC', 'Val AUC'),
        ('Test_Sensitivity', 'Test Sensitivity'),
    ]

    print("=" * 100)
    print("BNT on ADNI — Results Summary (mean ± std over runs)")
    print("=" * 100)

    header = f"  {'Label':<30} {'N':>3}"
    for _, display in metric_keys:
        header += f"  {display:>20}"
    print(header)
    print(f"  {'-' * 95}")

    for label in sorted(results.keys()):
        runs = results[label]
        row = f"  {label:<30} {len(runs):>3}"
        for key, _ in metric_keys:
            vals = [r[key] for r in runs if key in r]
            row += f"  {fmt(vals):>20}"
        print(row)

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
