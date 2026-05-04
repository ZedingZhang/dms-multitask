"""
Anchor 生成器 —— 为每个 FPN 层生成先验框
"""

import torch
import numpy as np
from itertools import product


class AnchorGenerator:
    """基于特征图尺寸与步长, 生成 anchor 中心点及宽高."""

    def __init__(self,
                 scales: list = None,
                 steps: list = None,
                 img_size: int = 640):
        self.scales = scales or [[16, 32], [64, 128], [256, 512]]
        self.steps = steps or [8, 16, 32]
        self.img_size = img_size

    def __call__(self, feature_maps: list = None):
        """
        Args:
            feature_maps: list of (H, W) for each FPN level; 默认根据 img_size 推算
        Returns:
            anchors: Tensor (N, 4) 格式 [cx, cy, w, h], 归一化到 [0, 1]
        """
        if feature_maps is None:
            feature_maps = [
                (self.img_size // s, self.img_size // s)
                for s in self.steps
            ]

        anchors = []
        for k, (fh, fw) in enumerate(feature_maps):
            step = self.steps[k]
            for i, j in product(range(fh), range(fw)):
                cx = (j + 0.5) * step / self.img_size
                cy = (i + 0.5) * step / self.img_size
                for s in self.scales[k]:
                    w = s / self.img_size
                    h = s / self.img_size
                    anchors.append([cx, cy, w, h])

        return torch.tensor(anchors, dtype=torch.float32)

    @staticmethod
    def decode_boxes(anchors: torch.Tensor, preds: torch.Tensor):
        """
        将网络预测的偏移量解码为实际边框坐标 [x1, y1, x2, y2].
        anchors: (N, 4) [cx, cy, w, h]
        preds:   (N, 4) [dx, dy, dw, dh]
        """
        cx = anchors[:, 0] + preds[:, 0] * anchors[:, 2]
        cy = anchors[:, 1] + preds[:, 1] * anchors[:, 3]
        w = anchors[:, 2] * torch.exp(preds[:, 2])
        h = anchors[:, 3] * torch.exp(preds[:, 3])

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    @staticmethod
    def decode_landmarks(anchors: torch.Tensor, preds: torch.Tensor,
                         num_landmarks: int = 14):
        """
        解码关键点预测.
        anchors: (N, 4) [cx, cy, w, h]
        preds:   (N, num_landmarks*2)
        返回:    (N, num_landmarks, 2)
        """
        landmarks = preds.reshape(-1, num_landmarks, 2)
        anchor_cx = anchors[:, 0].unsqueeze(1)
        anchor_cy = anchors[:, 1].unsqueeze(1)
        anchor_w = anchors[:, 2].unsqueeze(1)
        anchor_h = anchors[:, 3].unsqueeze(1)

        landmarks[:, :, 0] = anchor_cx + landmarks[:, :, 0] * anchor_w
        landmarks[:, :, 1] = anchor_cy + landmarks[:, :, 1] * anchor_h
        return landmarks
