"""
Teacher 模型定义 —— 用于知识蒸馏

支持两种高精度 Teacher:
  1. ResNet50 + FPN + 多任务 Head  (默认)
  2. YOLOv8-x Wrapper              (需安装 ultralytics)

Teacher 与 Student 共享相同的 Head 结构, 仅 Backbone 不同,
这样可以直接对 FPN 特征 / Head logits 做蒸馏.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from models.neck import LiteFPN
from models.heads import DetectionHead, LandmarkHead


# ────────────────────────────────────────────────────────────────
#  ResNet50 Backbone — 高精度 Teacher
# ────────────────────────────────────────────────────────────────

class ResNet50Backbone(nn.Module):
    """提取 ResNet50 的 C3/C4/C5 多尺度特征."""

    OUT_CHANNELS = (512, 1024, 2048)   # layer2 / layer3 / layer4

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,        # stride=4, 256ch
        )
        self.layer2 = resnet.layer2   # stride=8,  512ch  → C3
        self.layer3 = resnet.layer3   # stride=16, 1024ch → C4
        self.layer4 = resnet.layer4   # stride=32, 2048ch → C5

    def forward(self, x):
        x = self.stem(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class ResNet50Teacher(nn.Module):
    """
    完整的 ResNet50 Teacher 模型.
    结构: ResNet50 Backbone → FPN → DetHead + LmkHead
    与 Student DMSNet 保持相同的 Head, 方便 logit-level 蒸馏.
    """

    def __init__(self, num_classes: int = 3, num_landmarks: int = 14,
                 num_anchors: int = 2, fpn_channels: int = 128,
                 pretrained: bool = True):
        super().__init__()

        self.backbone = ResNet50Backbone(pretrained=pretrained)
        in_channels = ResNet50Backbone.OUT_CHANNELS   # (512, 1024, 2048)

        self.fpn = LiteFPN(in_channels=in_channels, fpn_channels=fpn_channels)

        self.det_head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )
        self.lmk_head = LandmarkHead(
            in_channels=fpn_channels,
            num_landmarks=num_landmarks,
            num_anchors=num_anchors,
        )

        self.fpn_channels = fpn_channels

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        features = self.fpn(c3, c4, c5)
        cls_scores, bbox_preds = self.det_head(features)
        lmk_preds = self.lmk_head(features)
        return cls_scores, bbox_preds, lmk_preds

    def get_fpn_features(self, x):
        c3, c4, c5 = self.backbone(x)
        return self.fpn(c3, c4, c5)
