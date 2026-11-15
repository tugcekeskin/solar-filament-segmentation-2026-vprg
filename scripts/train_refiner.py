#!/usr/bin/env python
"""Build refiner crops from train-split proposals + GT and train the U-Net refiner."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from filseg.infer import load_cache  # noqa: E402
from filseg.refine import build_dataset, make_input, make_model  # noqa: E402


def iou_per_sample(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    i = (pred & tgt).flatten(1).sum(1).float()
    u = (pred | tgt).flatten(1).sum(1).float()
    return i / u.clamp_min(1)


def loss_fn(logit, tgt):
    bce = F.binary_cross_entropy_with_logits(logit, tgt)
    p = torch.sigmoid(logit)
    inter = (p * tgt).flatten(1).sum(1)
    dice = 1 - (2 * inter + 1) / (p.flatten(1).sum(1) + tgt.flatten(1).sum(1) + 1)
    return bce + dice.mean()


def _erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)


def _dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def skeleton(t: torch.Tensor, iters: int = 15) -> torch.Tensor:
    """Morphological skeleton (clDice soft-skeleton applied to a binary target), tubed by a 3x3 dilation."""
    img, sk = t, F.relu(t - _dilate(_erode(t)))
    for _ in range(iters):
        img = _erode(img)
        d = F.relu(img - _dilate(_erode(img)))
        sk = sk + F.relu(d - sk * d)
    return _dilate((sk > 0).float()) * t


def skel_recall_loss(logit, skel):
    """Skeleton Recall loss (Kirchhoff et al., ECCV 2024): 1 - soft recall of the tubed GT skeleton."""
    p = torch.sigmoid(logit)
    return 1 - ((p * skel).flatten(1).sum(1) / skel.flatten(1).sum(1).clamp_min(1)).mean()


def augment(x: torch.Tensor, t: torch.Tensor):
    """x (B,2,S,S), t (B,1,S,S): per-sample flips/rot90 + intensity jitter on the image channel."""
    xs, ts = [], []
    for i in range(x.shape[0]):
        xi, ti = x[i], t[i]
        if np.random.rand() < 0.5:
            xi, ti = xi.flip(-1), ti.flip(-1)
        if np.random.rand() < 0.5:
            xi, ti = xi.flip(-2), ti.flip(-2)
        k = np.random.randint(4)
        xi, ti = torch.rot90(xi, k, (-2, -1)), torch.rot90(ti, k, (-2, -1))
        xs.append(xi), ts.append(ti)
    x, t = torch.stack(xs), torch.stack(ts)
    B = x.shape[0]
    c = torch.empty(B, 1, 1, device=x.device).uniform_(0.8, 1.25)
    b = torch.empty(B, 1, 1, device=x.device).uniform_(-0.3, 0.3)
    x = torch.cat([(x[:, 0] * c + b).unsqueeze(1), x[:, 1:]], 1)
    return x, t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="artifacts/refiner/train_raw.pkl")
    p.add_argument("--split", default="data/splits/split_v1.json")
    p.add_argument("--data", default="artifacts/refiner/crops.npz")
    p.add_argument("--out", default="artifacts/refiner/refiner.pt")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--arch", choices=["unet", "sam"], default="unet", help="sam = SAM ViT-B + LoRA (filseg.sam_refine)")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--win-mode", choices=["union", "prior"], default="union", help="training crop window from prior|GT union or prior only")
    p.add_argument("--frac-lo", type=float, default=0.25, help="window padding fraction range (prior mode)")
    p.add_argument("--frac-hi", type=float, default=0.25)
    p.add_argument("--frag-p", type=float, default=0.0, help="prob. a synthetic prior is cut to a fragment")
    p.add_argument("--skel-w", type=float, default=0.0, help="weight of the Skeleton Recall loss (0 = off)")
    p.add_argument("--max-train", type=int, default=0, help="subsample train crops (0 = all), for quick checks")
    a = p.parse_args()
    torch.manual_seed(a.seed), np.random.seed(a.seed)
    stems = json.load(open(a.split))["train"]
    if a.rebuild or not Path(a.data).exists():
        t = time.time()
        d = build_dataset(load_cache(a.cache), stems, a.data, seed=a.seed, win_mode=a.win_mode,
                          frac_lo=a.frac_lo, frac_hi=a.frac_hi, frag_p=a.frag_p)
        print(f"built {len(d['syn'])} samples ({int(d['syn'].sum())} synthetic) in {time.time() - t:.0f}s", flush=True)
    else:
        d = dict(np.load(a.data))
    rng = np.random.default_rng(a.seed)
    us = np.array(sorted(set(d["stem"].tolist())))
    hold = set(rng.choice(us, size=max(1, int(round(0.1 * len(us)))), replace=False).tolist())
    is_h = np.array([s in hold for s in d["stem"]])
    tr, ho = np.flatnonzero(~is_h), np.flatnonzero(is_h)
    if a.max_train:
        tr = rng.choice(tr, size=min(a.max_train, len(tr)), replace=False)
    print(f"train {len(tr)} crops / {len(us) - len(hold)} stems; holdout {len(ho)} crops / {len(hold)} stems", flush=True)
    img, prior, tgt = (torch.from_numpy(d[k]) for k in ("img", "prior", "tgt"))
    syn = d["syn"]
    dev = "cuda"
    if a.arch == "sam":
        from filseg.sam_refine import SamRefiner

        model = SamRefiner(rank=a.lora_rank).to(dev)
        save_sd = model.trainable_state_dict
    else:
        model = make_model().to(dev)
        save_sd = model.state_dict
    params = [q for q in model.parameters() if q.requires_grad]
    print(f"trainable params {sum(q.numel() for q in params) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    eb = 64 if a.arch == "unet" else 4
    amp = torch.float16 if a.arch == "unet" else torch.bfloat16
    steps = a.epochs * (len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler()

    def evaluate():
        model.eval()
        pi, ri = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            for i in range(0, len(ho), eb):
                j = ho[i : i + eb]
                x = make_input(img[j].to(dev), prior[j].to(dev))
                tg = tgt[j].to(dev).bool()
                r = torch.sigmoid(model(x).float())[:, 0] >= 0.5
                pr = x[:, 1] >= 0.5
                pi.append(iou_per_sample(pr, tg).cpu()), ri.append(iou_per_sample(r, tg).cpu())
        pi, ri = torch.cat(pi).numpy(), torch.cat(ri).numpy()
        s = syn[ho] == 1
        model.train()
        return {"all": (pi.mean(), ri.mean()), "real": (pi[~s].mean(), ri[~s].mean()), "syn": (pi[s].mean(), ri[s].mean())}

    best, t0 = -1.0, time.time()
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(tr)
        tot, n = 0.0, 0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            j = perm[i : i + a.batch]
            x = make_input(img[j].to(dev), prior[j].to(dev))
            tg = (tgt[j].to(dev).float().unsqueeze(1))
            x, tg = augment(x, tg)
            with torch.autocast("cuda", dtype=amp):
                lg = model(x)
            loss = loss_fn(lg.float(), tg)
            if a.skel_w > 0:
                loss = loss + a.skel_w * skel_recall_loss(lg.float(), skeleton(tg))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt), scaler.update(), sched.step()
            tot += loss.item()
            n += 1
        m = evaluate()
        print(f"ep {ep + 1:2d} loss {tot / n:.4f} | holdout IoU prior/refined: all {m['all'][0]:.4f}/{m['all'][1]:.4f} "
              f"real {m['real'][0]:.4f}/{m['real'][1]:.4f} syn {m['syn'][0]:.4f}/{m['syn'][1]:.4f} | {time.time() - t0:.0f}s", flush=True)
        if m["real"][1] > best:
            best = m["real"][1]
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(save_sd(), a.out)
    print(f"best holdout real refined IoU {best:.4f} (prior {m['real'][0]:.4f}); saved {a.out}")


if __name__ == "__main__":
    main()
