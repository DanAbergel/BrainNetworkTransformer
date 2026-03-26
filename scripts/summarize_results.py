"""
Summarize BNT ADNI training results from log files.

Each .out file contains 5 runs for the same label. Runs are detected
by Epoch[0/200] lines (start of each run). The last epoch of each run
is used for metrics. Reports mean ± std across runs.

Usage:
    python3 scripts/summarize_results.py logs/train_adni_*.out
"""

import re
import sys
from collections import defaultdict
import numpy as np


def parse_runs(filepath):
    """Parse all runs from a single log file."""
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
                # Detect start of a new run (Epoch[0/...)
                if "Epoch[0/" in line and current_last_epoch is not None:
                    metrics = parse_epoch_line(current_last_epoch)
                    if metrics:
                        runs.append(metrics)
                current_last_epoch = line

    # Capture the last run
    if current_last_epoch is not None:
        metrics = parse_epoch_line(current_last_epoch)
        if metrics:
            runs.append(metrics)

    return label, runs


def parse_epoch_line(line):
    """Extract metrics from an epoch log line."""
    metrics = {}
    for split in ['Train', 'Val', 'Test']:
        pattern = rf'{split} Acc:([\d.]+)% P:([\d.]+) R:([\d.]+) F1:([\d.]+)'
        m = re.search(pattern, line)
        if m:
            metrics[f'{split}_Accuracy'] = float(m.group(1))
            metrics[f'{split}_Precision'] = float(m.group(2))
            metrics[f'{split}_Recall'] = float(m.group(3))
            metrics[f'{split}_F1'] = float(m.group(4))
    return metrics if metrics else None


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
        label, runs = parse_runs(filepath)
        if label and runs:
            results[label].extend(runs)

    if not results:
        print("No results found.")
        sys.exit(1)

    splits = ['Train', 'Val', 'Test']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    print("=" * 100)
    print("BNT on ADNI — Results Summary (mean ± std over 5 runs)")
    print("=" * 100)

    for label in sorted(results.keys()):
        runs = results[label]
        print(f"\n  {label}  ({len(runs)} runs)")
        print(f"  {'-' * 96}")

        header = f"  {'':>10}"
        for m in metrics:
            header += f"  {m:>20}"
        print(header)

        for split in splits:
            row = f"  {split:>10}"
            for m in metrics:
                key = f'{split}_{m}'
                vals = [r[key] for r in runs if key in r]
                row += f"  {fmt(vals):>20}"
            print(row)

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
