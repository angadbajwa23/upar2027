"""Attribute model: timm backbone -> dropout -> linear head. Shipped inside the submission."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from scoring import BINARY, GROUPS

# Head layout for the grouped variant: each group's classes (+none), then the binaries.
GROUP_SIZES = [len(idx) + int(none) for idx, none in GROUPS]
N_GROUPED = sum(GROUP_SIZES) + len(BINARY)


class AttrNet(nn.Module):
    def __init__(self, backbone: str, img_hw=(256, 128), head="sigmoid", drop=0.1, pretrained=True):
        super().__init__()
        kw = {}
        if any(k in backbone for k in ("vit", "eva", "deit")):
            kw["img_size"] = tuple(img_hw)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        self.head_type = head
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(self.backbone.num_features, 40 if head == "sigmoid" else N_GROUPED)
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        """x: float [B,3,H,W] in [0,1]. Returns (logits, pooled features)."""
        f = self.backbone((x - self.mean) / self.std)
        return self.head(self.drop(f).to(self.head.weight.dtype)), f

    def probs(self, logits):
        """Returns (marginals [B,40], group-prob vector [B, N_GROUPED])."""
        if self.head_type == "sigmoid":
            from scoring import sigmoid_to_group_probs
            p = torch.sigmoid(logits.float())
            return p, sigmoid_to_group_probs(p)
        return grouped_logits_to_probs(logits.float())


def grouped_logits_to_probs(logits):
    p = torch.zeros(logits.shape[0], 40, device=logits.device)
    parts, o = [], 0
    for (idx, none), n in zip(GROUPS, GROUP_SIZES):
        sm = F.softmax(logits[:, o:o + n], 1)
        p[:, idx] = sm[:, :len(idx)]
        parts.append(sm)
        o += n
    b = torch.sigmoid(logits[:, o:])
    p[:, BINARY] = b
    parts += [b, 1 - b]
    return p, torch.cat(parts, 1)


def resize_for_model(x_uint8_nhwc, hw):
    """uint8 [B,H,W,3] (cached 384x192) -> float [B,3,h,w] in [0,1], antialiased resize."""
    x = x_uint8_nhwc.permute(0, 3, 1, 2).float().div_(255)
    if tuple(x.shape[-2:]) != tuple(hw):
        x = F.interpolate(x, size=tuple(hw), mode="bilinear", antialias=True, align_corners=False)
    return x
