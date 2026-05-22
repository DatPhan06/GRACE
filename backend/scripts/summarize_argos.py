"""
Summarize Recall@K results from ARGOS output TSV files.

Scans output/agros/REDIAL/ and output/agros/INSPIRED/ by default.
Optionally compares against GRACE baseline in output/REDIAL/ and output/INSPIRED/.

Usage (from backend/):
  python scripts/summarize_argos.py
  python scripts/summarize_argos.py --compare
  python scripts/summarize_argos.py --dataset redial
  python scripts/summarize_argos.py --dataset inspired --compare
  python scripts/summarize_argos.py --dirs output/agros/REDIAL output/agros/INSPIRED
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIRS = {
    "redial":  (PROJECT_ROOT / "output/agros/REDIAL",  PROJECT_ROOT / "output/grace/REDIAL"),
    "inspired": (PROJECT_ROOT / "output/agros/INSPIRED", PROJECT_ROOT / "output/grace/INSPIRED"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tsv_files(directory: Path) -> Dict[str, Dict[int, Path]]:
    """Returns {model_label: {top_k: path}}."""
    result: Dict[str, Dict[int, Path]] = {}
    pattern = re.compile(r"^(.+)_recall@(\d+)_(\d+)sample\.tsv$")
    if not directory.exists():
        return result
    for f in sorted(directory.glob("*.tsv")):
        m = pattern.match(f.name)
        if not m:
            continue
        model = m.group(1)
        k = int(m.group(2))
        result.setdefault(model, {})[k] = f
    return result


def load_recall_stats(path: Path) -> Tuple[float, int]:
    try:
        df = pd.read_csv(path, sep="\t")
        if "recall" not in df.columns or len(df) == 0:
            return 0.0, 0
        return float(df["recall"].mean()), len(df)
    except Exception as exc:
        print(f"  Warning: could not read {path.name}: {exc}", file=sys.stderr)
        return 0.0, 0


def build_results(directory: Path) -> Dict[str, Dict[int, Tuple[float, int]]]:
    files = find_tsv_files(directory)
    return {model: {k: load_recall_stats(p) for k, p in k_paths.items()}
            for model, k_paths in files.items()}


def print_table(title: str, results: Dict[str, Dict[int, Tuple[float, int]]]) -> None:
    if not results:
        print(f"\n{title}\n  (no files found in directory)")
        return

    all_ks = sorted({k for v in results.values() for k in v})
    col_w = max(len(m) for m in results) + 2

    header = f"{'Model':<{col_w}}" + "".join(f"  R@{k:<5}" for k in all_ks) + "  Rows"
    sep = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)

    for model, ks in sorted(results.items()):
        row = f"{model:<{col_w}}"
        n_rows = 0
        for k in all_ks:
            if k in ks:
                avg, n = ks[k]
                row += f"  {avg:.4f}"
                n_rows = n
            else:
                row += "  ------"
        row += f"  {n_rows}"
        print(row)
    print(sep)


def summarize_one(label: str, argos_dir: Path, grace_dir: Optional[Path]) -> None:
    argos = build_results(argos_dir)
    print_table(f"ARGOS — {label}", argos)

    if grace_dir is None:
        return

    grace = build_results(grace_dir)
    print_table(f"GRACE baseline — {label}", grace)

    common = set(argos) & set(grace)
    if not common:
        return

    delta: Dict[str, Dict[int, Tuple[float, int]]] = {}
    for model in sorted(common):
        common_ks = set(argos[model]) & set(grace[model])
        if common_ks:
            delta[f"Δ {model}"] = {
                k: (argos[model][k][0] - grace[model][k][0], argos[model][k][1])
                for k in common_ks
            }
    print_table(f"ARGOS − GRACE delta — {label}  (positive = ARGOS better)", delta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    datasets: List[str],
    compare: bool,
    custom_dirs: Optional[List[Path]],
) -> None:
    if custom_dirs:
        for d in custom_dirs:
            summarize_one(d.name, d, None)
        return

    for ds in datasets:
        argos_dir, grace_dir = DATASET_DIRS[ds]
        summarize_one(ds.upper(), argos_dir, grace_dir if compare else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize ARGOS Recall@K results for ReDial and/or INSPIRED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", choices=["redial", "inspired", "both"], default="both",
        help="Dataset to summarize (default: both)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also show GRACE baseline and delta tables",
    )
    parser.add_argument(
        "--dirs", type=Path, nargs="+", metavar="DIR",
        help="Custom output directories to scan (overrides --dataset)",
    )
    args = parser.parse_args()

    datasets = (
        ["redial", "inspired"] if args.dataset == "both"
        else [args.dataset]
    )

    main(
        datasets=datasets,
        compare=args.compare,
        custom_dirs=args.dirs,
    )
