"""CLI: python -m video_encoder {build-splits, encode, encode-video}."""
from __future__ import annotations

import argparse
from pathlib import Path

from .scenes import CANONICAL_SCENES
from .splits import build_loso_splits


def _cmd_build_splits(args):
    paths = build_loso_splits(
        data_root=args.data_root,
        leave_out=args.leave_out,
        out_dir=args.out,
        snap_policy=args.snap_policy,
        ratio=args.ratio,
    )
    print("Wrote splits:")
    for k, p in paths.items():
        print(f"  {k:>11s}  {p}")


def _cmd_encode(args):
    from .encoder import GlobalVideoEncoder

    enc = GlobalVideoEncoder(
        backbone=args.backbone,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    splits_dir = Path(args.splits)
    targets = args.splits_to_encode or ["train", "val", "test_train", "test_val"]
    written = {}
    for name in targets:
        parquet = splits_dir / f"{name}.parquet"
        if not parquet.exists():
            print(f"  skip {name}: {parquet} missing")
            continue
        print(f"-- encoding split: {name}")
        w = enc.encode_split_to_cache(parquet, args.out, limit=args.limit)
        written[name] = w
    print("Done. Per-split scene features written under", Path(args.out) / args.backbone)


def _cmd_encode_video(args):
    from .custom_video import encode_custom_video

    res = encode_custom_video(
        video=args.video,
        out_dir=args.out,
        backbone=args.backbone,
        stride=args.stride,
        device=args.device,
        batch_size=args.batch_size,
    )
    print("Wrote:")
    for k, v in res.items():
        print(f"  {k:>10s}  {v}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="video_encoder")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("build-splits", help="Build LOSO splits.")
    sp.add_argument("--data-root", required=True)
    sp.add_argument("--leave-out", required=True, choices=CANONICAL_SCENES)
    sp.add_argument("--out", required=True)
    sp.add_argument("--snap-policy", default="nearest", choices=["nearest", "floor", "drop"])
    sp.add_argument("--ratio", type=float, default=0.8)
    sp.set_defaults(func=_cmd_build_splits)

    se = sub.add_parser("encode", help="Encode frames in a split into features.")
    se.add_argument("--splits", required=True, help="Directory containing *.parquet splits.")
    se.add_argument("--out", required=True)
    se.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    se.add_argument("--device", default=None)
    se.add_argument("--batch-size", type=int, default=64)
    se.add_argument("--num-workers", type=int, default=2)
    se.add_argument("--limit", type=int, default=None)
    se.add_argument("--splits-to-encode", nargs="*", default=None)
    se.set_defaults(func=_cmd_encode)

    sv = sub.add_parser("encode-video", help="Encode any video file.")
    sv.add_argument("--video", required=True)
    sv.add_argument("--out", required=True)
    sv.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    sv.add_argument("--stride", type=int, default=10)
    sv.add_argument("--device", default=None)
    sv.add_argument("--batch-size", type=int, default=32)
    sv.set_defaults(func=_cmd_encode_video)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
