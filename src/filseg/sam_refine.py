"""SAM (ViT-B) + LoRA crop refiner: a drop-in replacement for the U-Net in ``filseg.refine``.

Same interface as the U-Net: ``forward(x)`` with x = ``refine.make_input`` output (B,2,256,256):
channel 0 the per-crop standardised image, channel 1 the prior probability. Returns mask logits
(B,1,256,256) in the same 256 crop frame.

How the crop is fed to SAM:
  * image: standardised z -> grey 128 + 40 z (a per-crop contrast stretch), replicated to RGB,
    bilinearly upsampled 256 -> 1024 (SAM's fixed input), SAM pixel normalisation;
  * box prompt: bbox of prior >= 0.5 (whole crop if empty), in 1024 coordinates;
  * mask prompt: prior as logits (B,1,256,256); SAM's low-res mask frame is input/4, i.e. exactly
    our 256 crop frame, so the prior needs no resampling and the output logits map 1:1 back.
Trainable: LoRA (rank r) on every ViT attention qkv projection, plus the full prompt encoder and
mask decoder. The ViT weights themselves stay frozen.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SAM_ID = "facebook/sam-vit-base"
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


class SamRefiner(nn.Module):
    def __init__(self, rank: int = 16, pretrained: bool = True, grad_ckpt: bool = True):
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import SamConfig, SamModel

        self.sam = SamModel.from_pretrained(SAM_ID) if pretrained else SamModel(SamConfig.from_pretrained(SAM_ID))
        for p in self.sam.parameters():
            p.requires_grad_(False)
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=["qkv"])
        self.sam.vision_encoder = inject_adapter_in_model(cfg, self.sam.vision_encoder)
        for n, p in self.sam.named_parameters():
            if "lora_" in n or n.startswith("prompt_encoder") or n.startswith("mask_decoder"):
                p.requires_grad_(True)
        if grad_ckpt:
            self.sam.vision_encoder.gradient_checkpointing = True
            from functools import partial

            import torch.utils.checkpoint as cp

            self.sam.vision_encoder._gradient_checkpointing_func = partial(cp.checkpoint, use_reentrant=False)
        self.register_buffer("mean", MEAN.clone(), persistent=False)
        self.register_buffer("std", STD.clone(), persistent=False)

    def trainable_state_dict(self) -> dict:
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in keep}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, s, _ = x.shape
        grey = (128.0 + 40.0 * x[:, :1]).clamp(0, 255)
        pix = F.interpolate(grey.expand(b, 3, s, s), size=(1024, 1024), mode="bilinear", align_corners=False)
        pix = (pix - self.mean) / self.std
        prior = x[:, 1:2].float()
        m = prior >= 0.5
        boxes = torch.zeros(b, 1, 4, device=x.device)
        scale = 1024.0 / s
        for i in range(b):
            ys, xs = torch.nonzero(m[i, 0], as_tuple=True)
            if len(ys):
                boxes[i, 0] = torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float() * scale
            else:
                boxes[i, 0] = torch.tensor([0.0, 0.0, 1024.0, 1024.0], device=x.device)
        if s != 256:
            prior = F.interpolate(prior, size=(256, 256), mode="bilinear", align_corners=False)
        mask_logit = torch.logit(prior.clamp(1e-3, 1 - 1e-3))
        out = self.sam(pixel_values=pix, input_boxes=boxes, input_masks=mask_logit, multimask_output=False)
        lg = out.pred_masks[:, 0, :1]  # (B,1,256,256)
        if s != 256:
            lg = F.interpolate(lg, size=(s, s), mode="bilinear", align_corners=False)
        return lg


def load_sam_refiner(weights: str, device: str = "cuda") -> SamRefiner:
    m = SamRefiner(pretrained=True, grad_ckpt=False)
    missing = m.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return m.to(device).eval()
