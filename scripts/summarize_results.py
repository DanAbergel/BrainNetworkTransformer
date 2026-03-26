"""
Summarize BNT ADNI training results from log files.

Parses the last epoch of each run (5 runs per label) and computes
mean ± std for each metric.

Usage:
    python3 scripts/summarize_results.py logs/bnt_adni_*.log
    python3 scripts/summarize_results.py logs/train_adni_*.out
"""

import re
import sys
from collections import defaultdict
import numpy as np


def parse_last_epoch(filepath):
    """Parse the last Epoch line from a log file, return dict of metrics."""
    last_line = None
    label = None
    with open(filepath) as f:
        for line in f:
            if "label=" in line.lower() or "Label:" in line:
                m = re.search(r'label[=:]\s*(\S+)', line, re.IGNORECASE)
                if m:
                    label = m.group(1).strip(',')
            if "Epoch[" in line:
                last_line = line
    if last_line is None:
        return None, None

    metrics = {}
    for pattern, key in [
        (r'Test Accuracy:\s*([\d.]+)', 'Accuracy'),
        (r'Test AUC:([\d.]+)', 'AUC'),
        (r'Test Precision:([\d.]+)', 'Precision'),
        (r'Test Recall:([\d.]+)', 'Recall'),
        (r'Test F1:([\d.]+)', 'F1'),
        (r'Test Sen:([\d.]+)', 'Sensitivity'),
    ]:
        m = re.search(pattern, last_line)
        if m:
            metrics[key] = float(m.group(1))
    return label, metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/summarize_results.py logs/train_adni_*.out")
        sys.exit(1)

    # Collect results: label -> list of metric dicts (one per run)
    results = defaultdict(list)

    for filepath in sys.argv[1:]:
        label, metrics = parse_last_epoch(filepath)
        if label and metrics:
            results[label].append(metrics)

    if not results:
        print("No results found. Check that log files contain Epoch lines.")
        sys.exit(1)

    # Determine all metric keys
    all_keys = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'Sensitivity']
    present_keys = [k for k in all_keys if any(k in m for runs in results.values() for m in runs)]

    # Print table
    header = f"{'Label':<30} {'Runs':>4}"
    for k in present_keys:
        header += f" | {k:>18}"
    print("=" * len(header))
    print("BNT on ADNI — Results Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label in sorted(results.keys()):
        runs = results[label]
        row = f"{label:<30} {len(runs):>4}"
        for k in present_keys:
            vals = [r[k] for r in runs if k in r]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                if len(vals) > 1:
                    row += f" | {mean:>7.4f} ± {std:.4f}"
                else:
                    row += f" | {mean:>7.4f}          "
            else:
                row += f" | {'N/A':>18}"
        print(row)

    print("=" * len(header))
    print(f"\nTotal runs parsed: {sum(len(v) for v in results.values())}")
    print(f"Labels: {sorted(results.keys())}")


if __name__ == "__main__":
    main()
