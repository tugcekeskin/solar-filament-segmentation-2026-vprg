"""Average the soft masks of two refined caches (same raw source), matched on (stem, score).

Rows present in only one cache are kept as-is. Output has the same schema.
Usage: blend_caches.py A.pkl B.pkl OUT.pkl [w_a=0.5]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filseg.infer import SOFT_COLUMN, load_cache, pack_soft, save_cache, unpack_soft  # noqa: E402
from filseg.rle import H, W, encode_mask  # noqa: E402


def full(t):
    y0, x0, c = unpack_soft(t)
    f = np.zeros((H, W), np.float32)
    f[y0 : y0 + c.shape[0], x0 : x0 + c.shape[1]] = c
    return f, (y0, x0, y0 + c.shape[0], x0 + c.shape[1])


def main():
    a, b, out = sys.argv[1:4]
    w = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    A, B = load_cache(a), load_cache(b)
    A["k"] = A["stem"] + "|" + A["score"].round(6).astype(str)
    B["k"] = B["stem"] + "|" + B["score"].round(6).astype(str)
    bm = dict(zip(B["k"], B[SOFT_COLUMN]))
    rows, used = [], set()
    for st, sc, k, t in zip(A["stem"], A["score"], A["k"], A[SOFT_COLUMN]):
        if k in bm:
            used.add(k)
            fa, ba = full(t)
            fb, bb = full(bm[k])
            y0, x0 = min(ba[0], bb[0]), min(ba[1], bb[1])
            y1, x1 = max(ba[2], bb[2]), max(ba[3], bb[3])
            c = (w * fa[y0:y1, x0:x1] + (1 - w) * fb[y0:y1, x0:x1]).round().astype(np.uint8)
            t = pack_soft(y0, x0, c)
        y0, x0, c = unpack_soft(t)
        m = np.zeros((H, W), np.uint8)
        m[y0 : y0 + c.shape[0], x0 : x0 + c.shape[1]] = c >= 128
        if m.sum() == 0:
            continue
        rows.append((st, float(sc), encode_mask(m), int(m.sum()), t))
    for st, sc, k, t, r, ar in zip(B["stem"], B["score"], B["k"], B[SOFT_COLUMN], B["rle"], B["area"]):
        if k not in used:
            rows.append((st, float(sc), r, int(ar), t))
    df = pd.DataFrame(rows, columns=["stem", "score", "rle", "area", SOFT_COLUMN])
    save_cache(df, out)
    print(f"A {len(A)} B {len(B)} matched {len(used)} -> {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
