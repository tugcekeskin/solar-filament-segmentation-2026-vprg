"""Run a YOLO-seg model on images and cache raw full-resolution instances.

Each instance is stored as (stem, score, rle, area) with a 2048x2048 compressed
RLE (see ``filseg.rle``). Inference uses a very low confidence so that
thresholds can be swept offline (``filseg.postprocess``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from filseg.rle import H, W, encode_mask

RAW_COLUMNS = ["stem", "score", "rle", "area"]
SOFT_COLUMN = "soft"
SOFT_PAD = 8  # px of padding around the bbox in stored soft crops


def pack_soft(y0: int, x0: int, crop: np.ndarray) -> tuple:
    """(y0, x0, h, w, zlib bytes of uint8 crop)."""
    import zlib

    return (int(y0), int(x0), int(crop.shape[0]), int(crop.shape[1]), zlib.compress(np.ascontiguousarray(crop).tobytes(), 3))


def unpack_soft(t: tuple) -> tuple[int, int, np.ndarray]:
    import zlib

    y0, x0, h, w, b = t
    return y0, x0, np.frombuffer(zlib.decompress(b), dtype=np.uint8).reshape(h, w)


def soft_masks_native(protos, masks_in, bboxes, shape, pad: int = SOFT_PAD) -> list:
    """Per-instance sigmoid mask probability at full resolution, cropped to the padded bbox, uint8 0-255.

    Mirrors ``ultralytics.utils.ops.process_mask_native`` (logits are bilinearly
    upsampled and letterbox-cropped) but returns sigmoid(logit) instead of
    ``logit > 0``. Pixels outside the (unpadded) bbox are zeroed like ultralytics'
    ``crop_mask``, so ``soft >= 0.5`` reproduces the binary retina masks.
    """
    import math

    import torch
    import torch.nn.functional as F
    from ultralytics.utils.ops import scale_masks

    c, mh, mw = protos.shape
    h, w = shape
    if masks_in.shape[0] == 0:
        return []
    coeffs = masks_in @ protos.float().view(c, -1)
    gain = min(4 * mh / h, 4 * mw / w)
    lp = (4 * mh - max(round(h * gain), 1)) | (4 * mw - max(round(w * gain), 1))
    f = 4 // math.gcd(4, lp | lp // 2)
    n = max(1, 16_000_000 // (h * w + f * f * mh * mw))
    out: list = []
    bb = bboxes.float().cpu().numpy()
    k = 0
    for ch in coeffs.view(1, -1, mh, mw).split(n, 1):
        lg = scale_masks(F.interpolate(ch, scale_factor=f, mode="bilinear") if f > 1 else ch, shape)[0]
        q = torch.round(torch.sigmoid(lg) * 255).clamp_(0, 255).to(torch.uint8)
        for j in range(q.shape[0]):
            x1, y1, x2, y2 = bb[k]
            k += 1
            # ultralytics crop_mask keeps pixels with x1 <= idx < x2
            bx0, bx1 = max(int(math.ceil(x1)), 0), min(int(math.ceil(x2)), w)
            by0, by1 = max(int(math.ceil(y1)), 0), min(int(math.ceil(y2)), h)
            if bx1 <= bx0 or by1 <= by0:
                out.append(None)
                continue
            y0, yy1 = max(by0 - pad, 0), min(by1 + pad, h)
            x0, xx1 = max(bx0 - pad, 0), min(bx1 + pad, w)
            crop = torch.zeros((yy1 - y0, xx1 - x0), dtype=torch.uint8, device=q.device)
            crop[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0] = q[j, by0:by1, bx0:bx1]
            out.append(pack_soft(y0, x0, crop.cpu().numpy()))
    return out


def _to_full_res(m: np.ndarray) -> np.ndarray:
    """Binary uint8 mask at 2048x2048 (nearest resize if the model returned another size)."""
    if m.shape != (H, W):
        import cv2

        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 0.5).astype(np.uint8)


def predict_raw(
    weights: str,
    image_paths: list[str],
    imgsz: int = 1024,
    conf: float = 0.01,
    iou: float = 0.7,
    max_det: int = 100,
    batch: int = 4,
    half: bool = True,
    tta: bool = False,
    device: str | int | None = None,
    soft: bool = False,
    flip: str = "",
    with_cls: bool = False,
) -> pd.DataFrame:
    """Predict every image; return DataFrame[stem, score, rle, area] (one row per instance).

    Deterministic given weights/params. Images with no detections yield no rows.
    With ``soft=True`` an extra ``soft`` column holds packed uint8 mask probabilities
    (see ``soft_masks_native``) aligned with the binary masks.
    ``flip`` in {"", "h", "v", "hv"} predicts on the flipped image and maps masks back
    to the original frame (manual flip TTA: ultralytics ignores augment=True for seg).
    ``with_cls=True`` appends a ``cls`` column (predicted class index, for multi-class models).
    """
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops

    model = YOLO(weights)
    rows: list[tuple] = []
    orig_pmn = ops.process_mask_native
    orig_cr = SegmentationPredictor.construct_result
    if soft:
        captured: list = []

        def _pmn(protos, masks_in, bboxes, shape):
            binm = orig_pmn(protos, masks_in, bboxes, shape)
            sm = soft_masks_native(protos, masks_in, bboxes, shape)
            keep = (binm.amax((-2, -1)) > 0).tolist()  # same filter ultralytics applies afterwards
            captured.append([s_ for s_, k_ in zip(sm, keep) if k_])
            return binm

        def _cr(self, *a, **k):
            captured.clear()
            r = orig_cr(self, *a, **k)
            r._soft = captured[0] if captured else None
            return r

        ops.process_mask_native = _pmn
        SegmentationPredictor.construct_result = _cr
    try:
        return _run(model, rows, soft, image_paths, imgsz, conf, iou, max_det, batch, half, tta, device, flip, with_cls)
    finally:
        ops.process_mask_native = orig_pmn
        SegmentationPredictor.construct_result = orig_cr


def flip_array(a: np.ndarray, flip: str) -> np.ndarray:
    if "h" in flip:
        a = a[:, ::-1]
    if "v" in flip:
        a = a[::-1]
    return np.ascontiguousarray(a)


def unflip_soft(t: tuple, flip: str) -> tuple:
    """Map a packed soft crop predicted on a flipped image back to the original frame."""
    y0, x0, crop = unpack_soft(t)
    h, w = crop.shape
    if "h" in flip:
        x0 = W - (x0 + w)
    if "v" in flip:
        y0 = H - (y0 + h)
    return pack_soft(y0, x0, flip_array(crop, flip))


def _run(model, rows, soft, image_paths, imgsz, conf, iou, max_det, batch, half, tta, device, flip="", with_cls=False) -> pd.DataFrame:
    import cv2

    paths = sorted(image_paths)
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        src = [flip_array(cv2.imread(p_), flip) for p_ in chunk] if flip else chunk
        results = model.predict(
            src, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, retina_masks=True,
            verbose=False, half=half, augment=tta, device=device, agnostic_nms=with_cls,
        )
        for p, r in zip(chunk, results):
            if r.masks is None or len(r.masks) == 0:
                continue
            stem = Path(p).stem
            masks = r.masks.data.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            softs = r._soft if soft else [None] * len(masks)
            assert len(softs) == len(masks)
            for m, s, sf, c in zip(masks, scores, softs, clss):
                mm = _to_full_res(m)
                if flip:
                    mm = flip_array(mm, flip)
                    sf = unflip_soft(sf, flip) if sf is not None else None
                area = int(mm.sum())
                if area == 0:
                    continue
                rows.append((stem, float(s), encode_mask(mm), area) + ((sf,) if soft else ()) + ((int(c),) if with_cls else ()))
    return pd.DataFrame(rows, columns=RAW_COLUMNS + ([SOFT_COLUMN] if soft else []) + (["cls"] if with_cls else []))


def save_cache(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def load_cache(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)
