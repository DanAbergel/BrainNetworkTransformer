"""
Summarize BNT ADNI training results from log files.

Parses the last epoch of each run (5 runs per label) and computes
mean ± std for each metric across train/val/test.

Usage:
    python3 scripts/summarize_results.py logs/train_adni_*.out
"""

import re
import sys
from collections import defaultdict
import numpy as np


def parse_last_epoch(filepath):
    """Parse the last Epoch line, return label and metrics dict."""
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

    # Parse: Train Acc:XX% AUC:XX P:XX R:XX F1:XX
    for split in ['Train', 'Val', 'Test']:
        pattern = rf'{split} Acc:([\d.]+)% AUC:([\d.]+) P:([\d.]+) R:([\d.]+) F1:([\d.]+)'
        m = re.search(pattern, last_line)
        if m:
            metrics[f'{split}_Accuracy'] = float(m.group(1))
            metrics[f'{split}_AUC'] = float(m.group(2))
            metrics[f'{split}_Precision'] = float(m.group(3))
            metrics[f'{split}_Recall'] = float(m.group(4))
            metrics[f'{split}_F1'] = float(m.group(5))

    return label, metrics


def print_table(results):
    """Print a formatted results table."""
    splits = ['Train', 'Val', 'Test']
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']

    print("=" * 120)
    print("BNT on ADNI — Results Summary (mean ± std over runs)")
    print("=" * 120)

    for label in sorted(results.keys()):
        runs = results[label]
        n = len(runs)
        print(f"\n{'─' * 120}")
        print(f"  {label}  ({n} runs)")
        print(f"{'─' * 120}")

        # Header
        header = f"  {'':>12}"
        for m in metrics:
            header += f"  {m:>18}"
        print(header)
        print(f"  {'':>12}  {'─'*18}  {'─'*18}  {'─'*18}  {'─'*18}  {'─'*18}")

        for split in splits:
            row = f"  {split:>12}"
            for m in metrics:
                key = f'{split}_{m}'
                vals = [r[key] for r in runs if key in r]
                if vals and len(vals) > 1:
                    row += f"  {np.mean(vals):>7.4f} ± {np.std(vals):.4f}"
                elif vals:
                    row += f"  {vals[0]:>7.4f}          "
                else:
                    row += f"  {'N/A':>18}"
            print(row)

    print(f"\n{'=' * 120}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/summarize_results.py logs/train_adni_*.out")
        sys.exit(1)

    results = defaultdict(list)
    for filepath in sys.argv[1:]:
        label, metrics = parse_last_epoch(filepath)
        if label and metrics:
            results[label].append(metrics)

    if not results:
        print("No results found.")
        sys.exit(1)

    print_table(results)


if __name__ == "__main__":
    main()
