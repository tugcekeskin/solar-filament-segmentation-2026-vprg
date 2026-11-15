"""Grouped (by stem) K-fold OOF rescoring of candidates. Prints ranking metrics, writes OOF scores.

Usage: rescore_cv.py FEAT.pkl OUT.pkl SET [--seeds N]; SET = own | flips | all
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

OWN = ["score", "area", "rank", "n_cand25", "soft_mean", "soft_hi", "soft_lo", "n_parts", "elong", "length", "fill",
       "r_rel", "contrast", "inside_rel", "inside_std", "max_iou_higher", "n_overlap_higher"]
FLIPS = [f"agree_{n}_{k}" for n in ("id", "h", "v", "hv") for k in ("iou", "score")]
MODELS = [f"agree_{n}_{k}" for n in ("s1", "m1792", "flat") for k in ("iou", "score")]
SETS = {"own": OWN, "flips": OWN + FLIPS, "all": OWN + FLIPS + MODELS, "score": ["score"]}


def model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)


def main():
    feat, out, which = sys.argv[1:4]
    d = pd.read_pickle(feat)
    cols = SETS[which]
    X, y, gr = d[cols].values, d["y_frac"].values, d["stem"].values
    oof = np.zeros(len(d))
    for seed in range(3):
        rng = np.random.RandomState(seed)
        us = np.unique(gr); perm = dict(zip(us, rng.permutation(len(us))))
        g2 = np.array([perm[s] for s in gr])
        for tr, te in GroupKFold(5).split(X, y, g2):
            m = model().fit(X[tr], y[tr])
            oof[te] += m.predict(X[te]) / 3
    pos = y >= 0.5
    for name, s in (("yolo score", d["score"].values), (f"rescore[{which}]", oof)):
        print(f"{name:18s} AUC(y>=.5) {roc_auc_score(pos, s):.4f}  AP {average_precision_score(pos, s):.4f}  "
              f"corr {np.corrcoef(s, y)[0, 1]:.4f}")
    d[["key", "stem", "score"]].assign(rescore=oof).to_pickle(out)


if __name__ == "__main__":
    main()
