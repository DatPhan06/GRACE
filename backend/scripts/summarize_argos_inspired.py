"""
Summarize Recall@K results from ARGOS INSPIRED output TSV files.

Reads all TSV files in output/agros/INSPIRED/ and prints a summary table.
Also compares against GRACE baseline in output/INSPIRED/ if available.

Usage (from project root or backend/):
  python backend/scripts/summarize_argos_inspired.py
  python backend/scripts/summarize_argos_inspired.py --output-dir output/agros/INSPIRED
  python backend/scripts/summarize_argos_inspired.py --compare
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def find_tsv_files(directory: Path) -> Dict[str, Dict[int, Path]]:
    """
    Scan a directory for recall TSV files.
    Returns {model_label: {top_k: path}} mapping.
    """
    result: Dict[str, Dict[int, Path]] = {}
    pattern = re.compile(r"^(.+)_recall@(\d+)_(\d+)sample\.tsv$")
    for f in sorted(directory.glob("*.tsv")):
        m = pattern.match(f.name)
        if not m:
            continue
        model = m.group(1)
        k = int(m.group(2))
        result.setdefault(model, {})[k] = f
    return result


def load_recall_stats(path: Path) -> Tuple[float, int]:
    """Return (avg_recall, row_count) for a TSV file."""
    try:
        df = pd.read_csv(path, sep="\t")
        if "recall" not in df.columns or len(df) == 0:
            return 0.0, 0
        return float(df["recall"].mean()), len(df)
    except Exception as exc:
        print(f"  Warning: could not read {path.name}: {exc}", file=sys.stderr)
        return 0.0, 0


def print_table(title: str, model_results: Dict[str, Dict[int, Tuple[float, int]]]) -> None:
    if not model_results:
        print(f"\n{title}\n  (no files found)")
        return

    all_ks = sorted({k for v in model_results.values() for k in v})
    col_w = max(len(m) for m in model_results) + 2

    header = f"{'Model':<{col_w}}" + "".join(f"  R@{k:<4}" for k in all_ks) + "  Rows"
    print(f"\n{title}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for model, ks in sorted(model_results.items()):
        row_str = f"{model:<{col_w}}"
        n_rows = 0
        for k in all_ks:
            if k in ks:
                avg, n = ks[k]
                row_str += f"  {avg:.4f}"
                n_rows = n  # assume same across K files
            else:
                row_str += "  ------"
        row_str += f"  {n_rows}"
        print(row_str)
    print("-" * len(header))


def main(argos_dir: Path, grace_dir: Optional[Path], compare: bool) -> None:
    # ---- ARGOS results ----
    argos_files = find_tsv_files(argos_dir)
    argos_results: Dict[str, Dict[int, Tuple[float, int]]] = {}
    for model, k_paths in argos_files.items():
        argos_results[model] = {k: load_recall_stats(p) for k, p in k_paths.items()}

    print_table("ARGOS INSPIRED results", argos_results)

    if not compare or grace_dir is None:
        return

    # ---- GRACE baseline comparison ----
    grace_files = find_tsv_files(grace_dir)
    grace_results: Dict[str, Dict[int, Tuple[float, int]]] = {}
    for model, k_paths in grace_files.items():
        grace_results[model] = {k: load_recall_stats(p) for k, p in k_paths.items()}

    print_table("GRACE (baseline) INSPIRED results", grace_results)

    # ---- Delta ----
    common_models = set(argos_results) & set(grace_results)
    if not common_models:
        return

    delta_results: Dict[str, Dict[int, Tuple[float, int]]] = {}
    for model in sorted(common_models):
        a_ks = argos_results[model]
        g_ks = grace_results[model]
        common_ks = set(a_ks) & set(g_ks)
        if common_ks:
            delta_results[f"Δ {model}"] = {
                k: (a_ks[k][0] - g_ks[k][0], a_ks[k][1])
                for k in common_ks
            }

    print_table("ARGOS − GRACE delta (positive = ARGOS better)", delta_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize ARGOS INSPIRED Recall@K results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "output/agros/INSPIRED",
        help="ARGOS output directory (default: output/agros/INSPIRED)",
    )
    parser.add_argument(
        "--grace-dir", type=Path,
        default=PROJECT_ROOT / "output/INSPIRED",
        help="GRACE baseline directory for comparison (default: output/INSPIRED)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also show GRACE baseline and delta table",
    )
    args = parser.parse_args()

    main(
        argos_dir=args.output_dir,
        grace_dir=args.grace_dir if args.compare else None,
        compare=args.compare,
    )
