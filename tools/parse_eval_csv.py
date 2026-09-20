#!/usr/bin/env python3
"""Parse the durable evaluation CSVs written by ``Trainer._flush_eval_csv``.

The trainer appends one row per successfully evaluated batch (``partial=1``)
and a final row once the loop finishes (``partial=0``). All metric values are
stored as SUMS over ``num_trajs`` trajectories, so the true average is
``value / num_trajs``.

Usage:
    python tools/parse_eval_csv.py results_sdd/cor_fm/_SDD_hodeathCircle_agent
    python tools/parse_eval_csv.py results_sdd/cor_fm/_SDD_hocoupa_agent --status test
    python tools/parse_eval_csv.py path/to/eval_test_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

# Metric families reported by the evaluator, each stored per horizon (1..4).
METRICS: tuple[str, ...] = (
    "ADE_min",
    "FDE_min",
    "JADE_min",
    "JFDE_min",
    "A_var",
    "F_var",
    "MASD",
)
NUM_HORIZONS = 4

# Horizon seconds for the standard ETH/UCY + SDD stride (freq=3 @ 2.5 fps).
DEFAULT_HORIZON_SECONDS: tuple[float, ...] = (1.2, 2.4, 3.6, 4.8)


def _resolve_csv(path: str, status: str) -> str:
    """Accept either a CSV file or a scene results directory."""
    if os.path.isfile(path):
        return path

    candidates = [
        os.path.join(path, f"eval_{status}_metrics.csv"),
        os.path.join(path, "log", f"eval_{status}_metrics.csv"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(
        "No eval CSV found. Looked for:\n  " + "\n  ".join(candidates)
    )


def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("num_trajs") is not None]
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return rows


def _pick_rows(rows: List[Dict[str, str]]) -> tuple[Dict[str, str], List[Dict[str, str]], bool]:
    """Return (chosen_row, partial_rows, used_partial_fallback).

    Prefers the last COMPLETE row (``partial == 0``); if the run crashed before
    the final flush, falls back to the last partial row (marked partial).
    """
    partial_rows = [r for r in rows if str(r.get("partial", "1")).strip() == "1"]
    complete_rows = [r for r in rows if str(r.get("partial", "1")).strip() == "0"]
    if complete_rows:
        return complete_rows[-1], partial_rows, False
    return rows[-1], partial_rows, True


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def report(csv_path: str, horizon_seconds: tuple[float, ...], chosen: Dict[str, str],
           partial_rows: List[Dict[str, str]], used_fallback: bool) -> None:
    num_trajs = int(float(chosen["num_trajs"]))
    failed_batches = int(float(chosen.get("failed_batches", 0) or 0))
    scene = chosen.get("held_out_scene", "") or chosen.get("dataset", "")
    status = chosen.get("status", "")

    print("=" * 78)
    print(f"Evaluation metrics: {csv_path}")
    print(f"  dataset/scene : {chosen.get('dataset', '?')} / {scene or '-'}")
    print(f"  status        : {status}  (partial={chosen.get('partial')})")
    print(f"  trajectories  : {num_trajs}")
    print(f"  batches flushed (partial rows): {len(partial_rows)}")
    print(f"  failed/skipped batches        : {failed_batches}"
          + ("  <-- robust loop skipped these" if failed_batches else "  (none)"))
    if used_fallback:
        print("  NOTE: no complete row found - run was interrupted; using the last")
        print("        partial flush, which is a valid partial average.")
    print("=" * 78)

    if num_trajs <= 0:
        print("No valid trajectories recorded; cannot compute averages.")
        return

    # Column widths: metric names + one column per horizon.
    name_w = max(len(m) for m in METRICS) + 2
    col_w = 12
    header = "metric".ljust(name_w) + "".join(
        f"{f'{s:g}s':>{col_w}}" for s in horizon_seconds[:NUM_HORIZONS]
    )
    print(header)
    print("-" * len(header))

    for metric in METRICS:
        cells = []
        for t in range(1, NUM_HORIZONS + 1):
            col = f"{metric}_{t}s"
            raw = chosen.get(col)
            if raw in (None, ""):
                cells.append(f"{'n/a':>{col_w}}")
                continue
            cells.append(f"{_fmt(float(raw) / num_trajs):>{col_w}}")
        print(metric.ljust(name_w) + "".join(cells))
    print("-" * len(header))
    print("Values are MEAN metrics (accumulated sum / num_trajs).")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="scene results directory or eval CSV file")
    parser.add_argument("--status", default="test", choices=["test", "train", "val"],
                        help="which eval CSV to read (default: test)")
    parser.add_argument("--horizon-seconds", default=",".join(str(s) for s in DEFAULT_HORIZON_SECONDS),
                        help="comma-separated horizon labels (default: 1.2,2.4,3.6,4.8 for SDD/ETH/UCY)")
    args = parser.parse_args(argv)

    horizon_seconds = tuple(float(x) for x in args.horizon_seconds.split(","))
    if len(horizon_seconds) != NUM_HORIZONS:
        parser.error(f"--horizon-seconds must provide {NUM_HORIZONS} values")

    try:
        csv_path = _resolve_csv(args.path, args.status)
        rows = _read_rows(csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    chosen, partial_rows, used_fallback = _pick_rows(rows)
    report(csv_path, horizon_seconds, chosen, partial_rows, used_fallback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
