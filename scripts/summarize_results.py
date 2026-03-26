"""
Summarize BNT ADNI training results from log files.

Each .out file contains 5 runs for the same label. This script parses
the last epoch of EACH run (detected by "ADNI loaded" separator lines)
and computes mean ± std across all 5 runs.

Usage:
    python3 scripts/summarize_results.py logs/train_adni_*.out
"""

import re
import sys
from collections import defaultdict
import numpy as np


def parse_runs(filepath):
    """Parse all runs from a single log file.

    Returns: (label, list of metric dicts — one per run)
    """
    label = None
    runs = []
    current_last_epoch = None

    with open(filepath) as f:
        for line in f:
            # Detect label
            if "Label:" in line:
                m = re.search(r'Label:\s*(\S+)', line)
                if m:
                    label = m.group(1)

            # Track last epoch line
            if "Epoch[" in line:
                current_last_epoch = line

            # "ADNI loaded" marks the start of a new run
            # so the previous last_epoch belongs to the previous run
            if "ADNI loaded" in line and current_last_epoch is not None:
                metrics = parse_epoch_line(current_last_epoch)
                if metrics:
                    runs.append(metrics)
                current_last_epoch = None

    # If file ends without "ADNI loaded" after last run (shouldn't happen
    # but handle it), capture the last epoch
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
        return f"{np.mean(vals):>7.4f} ± {np.std(vals):.4f}"
    elif len(vals) == 1:
        return f"{vals[0]:>7.4f}          "
    return f"{'N/A':>18}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/summarize_results.py logs/train_adni_*.out")
        sys.exit(1)

    # Collect: label -> list of metric dicts across all files
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

    print("=" * 110)
    print("BNT on ADNI — Results Summary (mean ± std over 5 runs)")
    print("=" * 110)

    for label in sorted(results.keys()):
        runs = results[label]
        print(f"\n{'─' * 110}")
        print(f"  {label}  ({len(runs)} runs)")
        print(f"{'─' * 110}")

        header = f"  {'':>10}"
        for m in metrics:
            header += f"  {m:>22}"
        print(header)

        for split in splits:
            row = f"  {split:>10}"
            for m in metrics:
                key = f'{split}_{m}'
                vals = [r[key] for r in runs if key in r]
                row += f"  {fmt(vals):>22}"
            print(row)

    print(f"\n{'=' * 110}")


if __name__ == "__main__":
    main()
