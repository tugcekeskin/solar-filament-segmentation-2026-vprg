#!/usr/bin/env python
"""Fuse several raw prediction caches (with soft crops) into one raw cache."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.ensemble import ensemble_caches  # noqa: E402
from filseg.infer import load_cache, save_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-votes", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    dfs = [load_cache(p) for p in a.caches]
    print("instances in:", {p: len(d) for p, d in zip(a.caches, dfs)})
    out = ensemble_caches(dfs, a.iou, a.min_votes, a.workers)
    save_cache(out, a.out)
    print(f"instances out: {len(out)} ({out['stem'].nunique()} stems) -> {a.out}  [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
