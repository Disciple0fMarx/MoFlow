"""Aggregate ETH/UCY cor_fm LOSO experiment metrics and validate checkpoints.

Discovers structured run directories under results_eth_ucy/cor_fm matching the
5 scenes x 3 video modes, parses log/eval_val_metrics.csv for best-epoch
minADE/minFDE/avgADE/avgFDE, validates models/checkpoint_best.pt, and writes:
- results_eth_ucy/aggregated_benchmark_metrics.csv
- results_eth_ucy/aggregated_benchmark_metrics.md
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
MODES = ["off", "static", "full"]
VM_MAP = {"off": "vmoff", "static": "vmstatic", "full": "vmfull"}

METRIC_ALIASES = {
    "minADE": ["minADE", "min_ade", "ADE_min", "ade_min", "best_ade"],
    "minFDE": ["minFDE", "min_fde", "FDE_min", "fde_min", "best_fde"],
    "avgADE": ["avgADE", "avg_ade", "ADE_avg", "ade_avg", "mean_ade"],
    "avgFDE": ["avgFDE", "avg_fde", "FDE_avg", "fde_avg", "mean_fde"],
}


def _find_col(fieldnames: Iterable[str], aliases: List[str]) -> Optional[str]:
    by_norm = {str(f).strip().lower(): f for f in fieldnames if f}
    for a in aliases:
        if a.lower() in by_norm:
            return by_norm[a.lower()]
    # fuzzy contains
    for a in aliases:
        al = a.lower()
        for norm, orig in by_norm.items():
            if al in norm or norm in al:
                return orig
    return None


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def discover_run_dir(cor_fm_root: Path, scene: str, mode: str) -> Optional[Path]:
    vm = VM_MAP[mode]
    # Exact nomenclature pattern (user spec) plus tolerant fallbacks.
    patterns = [
        f"*_{scene}_{vm}_min_max_LR0.0001_WD0.01_CLS_1.0_BS32_EP64",
        f"*_{scene}_{vm}_*",
        f"*_{scene}_*{vm}*",
        f"*_{scene}_vm{mode}_*",
        f"*_{scene}_{mode}_*",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        for p in cor_fm_root.glob(pat):
            if p.is_dir():
                candidates.append(p)
    # Prefer best/most specific match (longest name containing exact vm token)
    if not candidates:
        # recursive fallback
        for p in cor_fm_root.rglob(f"*{scene}*{vm}*"):
            if p.is_dir():
                candidates.append(p)
    uniq = []
    seen = set()
    for c in candidates:
        rp = c.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(c)
    if not uniq:
        return None

    def score(p: Path) -> Tuple[int, int, float]:
        n = p.name
        s = 0
        if f"_{scene}_{vm}_" in n or n.endswith(f"_{scene}_{vm}"):
            s += 100
        if "min_max_LR0.0001_WD0.01_CLS_1.0_BS32_EP64" in n:
            s += 50
        if "_FM_S10_" in n:
            s += 10
        has_best = (p / "models" / "checkpoint_best.pt").exists()
        has_csv = (p / "log" / "eval_val_metrics.csv").exists()
        return (s + (5 if has_best else 0) + (5 if has_csv else 0), len(n), p.stat().st_mtime if p.exists() else 0)

    uniq.sort(key=score, reverse=True)
    return uniq[0]


HORIZON_METRIC_PREFIXES = [
    "ADE_min", "ADE_avg", "FDE_min", "FDE_avg",
    "JADE_min", "JADE_avg", "JFDE_min", "JFDE_avg",
    "MASD", "AVar", "FVar",
]


def _norm_metric_col(col: str) -> str:
    c = str(col).strip()
    # normalize common variants
    c = c.replace("ade_min", "ADE_min").replace("fde_min", "FDE_min")
    c = c.replace("ade_avg", "ADE_avg").replace("fde_avg", "FDE_avg")
    c = c.replace("jade_min", "JADE_min").replace("jfde_min", "JFDE_min")
    c = c.replace("jade_avg", "JADE_avg").replace("jfde_avg", "JFDE_avg")
    c = c.replace("masd", "MASD").replace("avar", "AVar").replace("fvar", "FVar")
    c = c.replace("-", "_")
    return c


def _is_horizon_metric_col(col: str) -> bool:
    c = _norm_metric_col(col)
    for pfx in HORIZON_METRIC_PREFIXES:
        if c == pfx:
            return True
        if c.startswith(pfx + "_") or c.startswith(pfx + "@") or c.startswith(pfx + "/"):
            return True
        # e.g. ADE_min_1.2, ADE_min1.2
        if re.match(rf"^{re.escape(pfx)}[_/@]?\d", c):
            return True
    return False


def _extract_horizon_metrics(row: Dict[str, object], fieldnames: List[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for col in fieldnames:
        if not col:
            continue
        if _is_horizon_metric_col(col):
            key = _norm_metric_col(col)
            # keep horizon suffix numeric-friendly
            key = re.sub(r"[^A-Za-z0-9_.]+", "_", key)
            out[key] = _to_float(row.get(col))
    return out


def _best_epoch_key(row_vals: Dict[str, object], i: int):
    # primary: overall min ADE (ADE_min final if present, else any ADE_min*, else val/loss)
    candidates = []
    for k, v in row_vals.items():
        if v is None:
            continue
        nk = k.lower()
        if nk in ("ade_min", "minade", "ade_min_final"):
            candidates.append((0, float(v)))
        elif nk.startswith("ade_min"):
            candidates.append((1, float(v)))
    if candidates:
        candidates.sort()
        return (candidates[0][1], i)
    for lk in ["val_loss", "loss", "eval_loss"]:
        if lk in row_vals and row_vals[lk] is not None:
            return (float(row_vals[lk]), i)
    return (float("inf"), i)


def parse_best_metrics(csv_path: Path) -> Dict[str, Optional[object]]:
    out: Dict[str, Optional[object]] = {"best_epoch": None, "n_rows": 0}
    if not csv_path.exists():
        return out
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or (rows[0].keys() if rows else []))
    out["n_rows"] = len(rows)
    if not rows:
        return out
    # also keep legacy aliases for compatibility
    legacy_cols = {k: _find_col(fieldnames, v) for k, v in METRIC_ALIASES.items()}
    best = None
    best_key = None
    for i, row in enumerate(rows):
        all_num = { _norm_metric_col(k): _to_float(v) for k, v in row.items() if k }
        key = _best_epoch_key(all_num, i)
        if best_key is None or key < best_key:
            best_key = key
            best = (i, row)
    assert best is not None
    i, row = best
    epoch_col = _find_col(fieldnames, ["epoch", "ep", "step"])
    out["best_epoch"] = _to_float(row.get(epoch_col)) if epoch_col else i
    for k, col in legacy_cols.items():
        out[k] = _to_float(row.get(col)) if col else None
    out.update(_extract_horizon_metrics(row, fieldnames))
    return out


def validate_artifacts(run_dir: Path) -> Dict[str, object]:
    ckpt_best = run_dir / "models" / "checkpoint_best.pt"
    ckpt_last = run_dir / "models" / "checkpoint_last.pt"
    eval_csv = run_dir / "log" / "eval_val_metrics.csv"
    cfgs = [run_dir / "cor_fm.yml", run_dir / "cor_fm_updated.yml", run_dir / "code_backup" / "cor_fm.yml"]
    return {
        "checkpoint_best_exists": ckpt_best.exists(),
        "checkpoint_best_path": str(ckpt_best) if ckpt_best.exists() else "",
        "checkpoint_last_exists": ckpt_last.exists(),
        "eval_val_metrics_exists": eval_csv.exists(),
        "eval_val_metrics_path": str(eval_csv),
        "config_backup_exists": any(p.exists() for p in cfgs),
    }


def aggregate(cor_fm_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scene in SCENES:
        for mode in MODES:
            run_dir = discover_run_dir(cor_fm_root, scene, mode)
            rec: Dict[str, object] = {
                "scene": scene,
                "video_mode": mode,
                "vm_tag": VM_MAP[mode],
                "run_dir": str(run_dir) if run_dir else "",
                "status": "ok" if run_dir else "missing_run_dir",
            }
            if not run_dir:
                rec["best_epoch"] = ""
                rec["n_rows"] = 0
                rec.update({
                    "checkpoint_best_exists": False,
                    "checkpoint_best_path": "",
                    "checkpoint_last_exists": False,
                    "eval_val_metrics_exists": False,
                    "eval_val_metrics_path": "",
                    "config_backup_exists": False,
                })
                rec["status"] = "missing_run_dir"
                rows.append(rec)
                continue
            art = validate_artifacts(run_dir)
            rec.update(art)
            metrics = parse_best_metrics(run_dir / "log" / "eval_val_metrics.csv")
            rec.update({k: ("" if v is None else v) for k, v in metrics.items()})
            if not art["checkpoint_best_exists"]:
                rec["status"] = "missing_checkpoint_best"
                # do not mislabel: checkpoint path is explicit
            elif not art["eval_val_metrics_exists"]:
                rec["status"] = "missing_eval_metrics"
            else:
                rec["status"] = "ok"
            rows.append(rec)
    return rows


def _dynamic_metric_fields(rows: List[Dict[str, object]]) -> List[str]:
    base = {
        "scene","video_mode","vm_tag","status","run_dir",
        "checkpoint_best_exists","checkpoint_best_path","checkpoint_last_exists",
        "eval_val_metrics_exists","eval_val_metrics_path","config_backup_exists",
        "best_epoch","n_rows","minADE","minFDE","avgADE","avgFDE",
    }
    keys = set()
    for r in rows:
        for k in r.keys():
            if k not in base:
                keys.add(k)
    def sort_key(k: str):
        m = re.match(r"([A-Za-z]+(?:_[A-Za-z]+)*)[_@]?([0-9]+(?:\.[0-9]+)?)?$", k)
        if m:
            return (m.group(1), float(m.group(2)) if m.group(2) else -1, k)
        return (k, -1, k)
    return sorted(keys, key=sort_key)


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> List[str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = _dynamic_metric_fields(rows)
    fields = [
        "scene", "video_mode", "vm_tag", "status", "run_dir",
        "checkpoint_best_exists", "checkpoint_best_path", "checkpoint_last_exists",
        "eval_val_metrics_exists", "eval_val_metrics_path", "config_backup_exists",
        "best_epoch", "n_rows", "minADE", "minFDE", "avgADE", "avgFDE",
    ] + metric_fields
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return metric_fields


def _fmt(v) -> str:
    if v == "" or v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_markdown(rows: List[Dict[str, object]], out_path: Path, cor_fm_root: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# ETH/UCY LOSO Aggregated Benchmark Metrics")
    lines.append("")
    lines.append(f"- Source: `{cor_fm_root}`")
    lines.append(f"- Runs discovered: {sum(1 for r in rows if r.get('run_dir'))}/15")
    lines.append(f"- Checkpoint best present: {sum(1 for r in rows if r.get('checkpoint_best_exists'))}/15")
    lines.append("")
    lines.append("## Comparison Table")
    lines.append("")
    header = ["Scene", "Mode", "Status", "Best Ckpt", "Best Epoch", "minADE", "minFDE", "avgADE", "avgFDE", "Run Dir"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join([
            _fmt(r.get("scene")), _fmt(r.get("video_mode")), _fmt(r.get("status")),
            "yes" if r.get("checkpoint_best_exists") else "no",
            _fmt(r.get("best_epoch")), _fmt(r.get("minADE")), _fmt(r.get("minFDE")),
            _fmt(r.get("avgADE")), _fmt(r.get("avgFDE")),
            f"`{Path(str(r.get('run_dir'))).name}`" if r.get("run_dir") else "-",
        ]) + " |")
    lines.append("")
    # Per-scene pivot (minADE/minFDE by mode)
    lines.append("## minADE / minFDE by Mode")
    lines.append("")
    lines.append("| Scene | off (ADE/FDE) | static (ADE/FDE) | full (ADE/FDE) |")
    lines.append("|---|---:|---:|---:|")
    idx = {(r["scene"], r["video_mode"]): r for r in rows}
    for scene in SCENES:
        cells = []
        for mode in MODES:
            r = idx.get((scene, mode), {})
            cells.append(f"{_fmt(r.get('minADE'))} / {_fmt(r.get('minFDE'))}")
        lines.append(f"| {scene} | " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description="Aggregate cor_fm ETH/UCY LOSO metrics and validate checkpoints.")
    p.add_argument("--cor-fm-root", default="results_eth_ucy/cor_fm")
    p.add_argument("--out-csv", default="results_eth_ucy/aggregated_benchmark_metrics.csv")
    p.add_argument("--out-md", default="results_eth_ucy/aggregated_benchmark_metrics.md")
    p.add_argument("--out-comprehensive-csv", default="results_eth_ucy/comprehensive_horizon_metrics.csv")
    args = p.parse_args(argv)

    cor_fm_root = Path(args.cor_fm_root)
    if not cor_fm_root.exists():
        print(f"[WARN] cor_fm root not found: {cor_fm_root}. Writing empty discovery report.")
    rows = aggregate(cor_fm_root)
    metric_fields = write_csv(rows, Path(args.out_csv))
    # comprehensive wide-format: one row per (scene, video_mode) with all horizon metric-horizon pairs
    write_csv(rows, Path(args.out_comprehensive_csv))
    write_markdown(rows, Path(args.out_md), cor_fm_root)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_ckpt = sum(1 for r in rows if r.get("checkpoint_best_exists"))
    print(f"[aggregate] wrote {args.out_csv}, {args.out_comprehensive_csv}, {args.out_md} ({n_ok}/15 ok, {n_ckpt}/15 checkpoint_best.pt found)")
    # clean stdout summary
    show = ["scene","video_mode","status","best_epoch","minADE","minFDE","avgADE","avgFDE"] + metric_fields[:12]
    print(" | ".join(show))
    for r in rows:
        print(" | ".join(_fmt(r.get(k)) for k in show))
    for r in rows:
        if r.get("status") != "ok":
            print(f"  [!] {r['scene']}/{r['video_mode']}: {r['status']} run_dir={r.get('run_dir') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
