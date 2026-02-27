from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/val/test split files from tokenized tracks."
    )
    parser.add_argument(
        "--tokens-root",
        type=Path,
        default=Path("data/tokens"),
        help="Path with token files (*.pt or *.npz).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/splits"),
        help="Path to write split txt files.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="Glob for token files.",
    )
    return parser.parse_args()


def normalize_track_id(stem: str) -> str:
    return re.sub(r"__chunk\d{4}$", "", stem)


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be non-negative.")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")
    if not args.tokens_root.exists():
        raise FileNotFoundError(f"Tokens folder does not exist: {args.tokens_root}")

    token_files = sorted(args.tokens_root.rglob(args.pattern))
    if not token_files:
        raise RuntimeError(f"No token files found with pattern '{args.pattern}'.")

    track_ids = sorted({normalize_track_id(path.stem) for path in token_files})
    if len(track_ids) < 3:
        raise RuntimeError("Need at least 3 unique tracks to build train/val/test.")

    rnd = random.Random(args.seed)
    rnd.shuffle(track_ids)

    total = len(track_ids)
    val_count = max(1, int(total * args.val_ratio))
    test_count = max(1, int(total * args.test_ratio))
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise RuntimeError("Split produced empty train set. Add more tracks or change ratios.")

    train_ids = sorted(track_ids[:train_count])
    val_ids = sorted(track_ids[train_count : train_count + val_count])
    test_ids = sorted(track_ids[train_count + val_count :])

    write_lines(args.output_root / "train.txt", train_ids)
    write_lines(args.output_root / "val.txt", val_ids)
    write_lines(args.output_root / "test.txt", test_ids)

    summary = {
        "total_tracks": total,
        "train_tracks": len(train_ids),
        "val_tracks": len(val_ids),
        "test_tracks": len(test_ids),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "pattern": args.pattern,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print("Split files saved:")
    print(f"  train: {len(train_ids)} tracks")
    print(f"  val:   {len(val_ids)} tracks")
    print(f"  test:  {len(test_ids)} tracks")
    print(f"  output: {args.output_root}")


if __name__ == "__main__":
    main()

