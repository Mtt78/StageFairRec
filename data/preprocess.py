#!/usr/bin/env python3
"""Prepare StageFairRec data from canonical UTF-8 CSV files.

Run from the repository root:
    python data/preprocess.py
    python data/preprocess.py --raw data/raw --out data/prepared
    python data/preprocess.py --attribute age --age-bins 20 30 40 --out data/age

Default paths are relative to this script, independent of the working directory.
Required input files:
    courses.csv: course_id plus course text fields
    users.csv: user_id plus gender (explicit 0/1) or age (numeric years)
    interactions.csv: user_id, course_id, timestamp (numeric)

Age boundaries in the example are illustrative, not recovered paper settings.
The shared implementation filters users, orders events, splits each user's
history 8:1:1, and caches one target plus 99 fixed unseen negative candidates.
Outputs: metadata.json, rows.npy, candidates.npy, manifest.json.
No real dataset or demographic information is bundled with this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


DATA_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DATA_DIR.parent
# Allows direct execution before an editable install, after dependencies exist.
sys.path.insert(0, str(REPOSITORY_ROOT))


def check_inputs(raw: Path, attribute: str) -> None:
    required = {
        "courses.csv": {"course_id"},
        "users.csv": {"user_id", attribute},
        "interactions.csv": {"user_id", "course_id", "timestamp"},
    }
    for filename, columns in required.items():
        path = raw / filename
        if not path.is_file():
            raise ValueError(f"Missing input file: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            header = csv.DictReader(stream).fieldnames or []
        missing = columns - set(header)
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DATA_DIR / "raw")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "prepared")
    parser.add_argument("--attribute", choices=("gender", "age"), default="gender")
    parser.add_argument("--age-bins", type=float, nargs="+")
    parser.add_argument("--min-interactions", type=int, default=10)
    parser.add_argument("--candidate-seed", type=int, default=2026)
    parser.add_argument("--eval-negatives", type=int, default=99,
                        help="Paper setting: 99; smaller values are non-paper experiments.")
    args = parser.parse_args(argv)
    if args.min_interactions < 10:
        parser.error("--min-interactions must be at least 10 for this paper protocol")
    if args.eval_negatives < 1:
        parser.error("--eval-negatives must be positive")
    if args.candidate_seed < 0:
        parser.error("--candidate-seed must be nonnegative")
    if args.attribute == "age":
        if not args.age_bins:
            parser.error("Age experiments require explicit --age-bins")
        if (any(not math.isfinite(x) or not 0 < x < 120 for x in args.age_bins)
                or args.age_bins != sorted(set(args.age_bins))):
            parser.error("--age-bins must be finite, strictly increasing boundaries within (0,120)")
    elif args.age_bins is not None:
        parser.error("--age-bins is only valid with --attribute age")
    try:
        check_inputs(args.raw, args.attribute)
        from stagefairrec.data import prepare
        prepare(raw=args.raw, out=args.out, attribute=args.attribute,
                min_interactions=args.min_interactions,
                candidate_seed=args.candidate_seed,
                eval_negatives=args.eval_negatives, age_bins=args.age_bins)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Preprocessing failed: {exc}\n")
    manifest = json.loads((args.out / "manifest.json").read_text())
    print(json.dumps({"output": str(args.out.resolve()), **manifest},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
