"""Inference + postprocessing: conf filter, mask NMS, min-area, thinning (erosion), optional flip TTA."""
import numpy as np, cv2, torch

KERNELS = {"none": None, "3": cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
           "5": cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
           "7": cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))}


def raw_predict(model, path, imgsz, conf=0.05, iou=0.5):
    """Returns (masks[N,2048,2048] uint8, scores[N]) at low conf; filtering done later."""
    r = model.predict(path, imgsz=imgsz, conf=conf, iou=iou, retina_masks=True, verbose=False, max_det=300)[0]
    if r.masks is None:
        return np.zeros((0, 2048, 2048), np.uint8), np.zeros(0)
    return (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8), r.boxes.conf.cpu().numpy()


def mask_nms(masks, scores, thr):
    order = np.argsort(-scores); keep = []
    for i in order:
        if all(np.logical_and(masks[i], masks[j]).sum() / max(1, np.logical_or(masks[i], masks[j]).sum()) <= thr
               for j in keep):
            keep.append(i)
    return keep


def post(masks, scores, conf=0.3, nms=0.3, min_area=20, thin="5"):
    out = []
    idx = [i for i in np.argsort(-scores) if scores[i] >= conf]
    if not idx:
        return out
    m, s = masks[idx], scores[idx]
    for i in mask_nms(m, s, nms):
        x = m[i]
        if KERNELS[thin] is not None:
            x = cv2.erode(x, KERNELS[thin])
        if x.sum() >= min_area:
            out.append(x)
    return out
