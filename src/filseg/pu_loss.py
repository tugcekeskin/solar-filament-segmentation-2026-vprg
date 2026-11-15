"""Positive-unlabeled segmentation loss for ultralytics YOLO-seg.

Datasets built by ``scripts/make_pu_dataset.py`` have two classes: 0 = filament, 1 = ignore (a likely
filament the single annotator did not mark). The task-aligned assigner runs as usual over both, so an
ignore instance "claims" its anchors. Those anchors then get
  * no classification loss on the filament channel (neither positive nor negative),
  * no box / DFL / mask loss (they are removed from the foreground mask).
The ignore channel itself is always trained towards 0, so at inference only the filament channel fires.
Anchors not assigned to anything are ordinary negatives, exactly as in the stock loss.
"""
from __future__ import annotations

from typing import Any

import torch
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors

IGNORE_CLS = 1


class PUSegmentationLoss(v8SegmentationLoss):
    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any], **kw) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            **kw,
        )
        assigned_cls = gt_labels.squeeze(-1).long().gather(1, target_gt_idx)  # (b, anchors)
        ignore = fg_mask & (assigned_cls == IGNORE_CLS)
        fg_mask = fg_mask & ~ignore
        target_scores = target_scores.clone()
        target_scores[..., IGNORE_CLS] = 0.0  # ignore channel: always a negative
        weight = torch.ones_like(target_scores)
        weight[..., 0][ignore] = 0.0  # filament channel: no loss on ignore-claimed anchors
        target_scores_sum = target_scores.sum().clamp_(min=1)
        bce = self.bce(pred_scores, target_scores.to(dtype)) * weight
        loss[1] = bce.sum() / target_scores_sum
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor, target_scores,
            target_scores_sum, fg_mask, imgsz, stride_tensor,
        )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


def install() -> None:
    """Make every ultralytics SegmentationModel use the PU loss (call before model.train)."""
    from ultralytics.nn.tasks import SegmentationModel

    SegmentationModel.init_criterion = lambda self: PUSegmentationLoss(self)
