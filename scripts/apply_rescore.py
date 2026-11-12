"""Replace cache scores with rescorer scores (joined on stem|round(score,6)); drop unmatched rows.
Usage: apply_rescore.py CACHE.pkl SCORES.pkl OUT.pkl [blend_w]  (score = w*rescore + (1-w)*score)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from filseg.infer import load_cache, save_cache  # noqa: E402

c, s, out = sys.argv[1:4]
w = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
d = load_cache(c)
d["key"] = d["stem"] + "|" + d["score"].round(6).astype(str)
m = dict(zip(*pd.read_pickle(s)[["key", "rescore"]].values.T))
n0 = len(d)
d = d[d["key"].isin(m)].copy()
d["score"] = [w * m[k] + (1 - w) * sc for k, sc in zip(d["key"], d["score"])]
save_cache(d.drop(columns="key").reset_index(drop=True), out)
print(f"{n0} -> {len(d)} rows rescored -> {out}")
