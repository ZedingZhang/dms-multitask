"""
Detection Head —— 目标检测头
负责三类目标的分类 (face / phone / cigarette) 与边框回归
"""

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """
    轻量级 Anchor-based 检测头 — 共享卷积 + 双分支输出.

    FPN 特征先经过共享子网提取通用特征, 再分别进入两个 1×1 输出层:
      shared_subnet (共享)
           │
      ┌────┴────┐
      ↓         ↓
    cls_output  bbox_output
    (分类)      (边框回归)

    相比独立 cls/bbox 子网, 参数减少 ~30%, 且共享特征有助于多任务泛化.
    """

    def __init__(self, in_channels: int = 64, num_classes: int = 3,
                 num_anchors: int = 2, num_conv: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # 共享特征提取
        shared = []
        for _ in range(num_conv):
            shared += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
            ]
        self.shared_subnet = nn.Sequential(*shared)

        # 任务特定输出层 (仅 1×1 Conv, 极小参数量)
        self.cls_output = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.bbox_output = nn.Conv2d(in_channels, num_anchors * 4, 1)

        self._init_weights()

    def _init_weights(self):
        import math
        # 分类输出使用 focal loss 的 bias 初始化
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_output.bias, bias_value)
        nn.init.normal_(self.bbox_output.weight, std=0.01)
        nn.init.constant_(self.bbox_output.bias, 0)

    def forward_single(self, feat: torch.Tensor):
        """对单层 FPN 特征做检测."""
        x = self.shared_subnet(feat)
        cls_score = self.cls_output(x)
        bbox_pred = self.bbox_output(x)
        return cls_score, bbox_pred

    def forward(self, features: tuple):
        """
        Args:
            features: (P3, P4, P5) FPN 特征
        Returns:
            cls_scores:  list of (B, num_anchors*num_classes, Hi, Wi)
            bbox_preds:  list of (B, num_anchors*4, Hi, Wi)
        """
        cls_scores = []
        bbox_preds = []
        for feat in features:
            cls_score, bbox_pred = self.forward_single(feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds
