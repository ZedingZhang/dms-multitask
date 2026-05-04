"""
DMSNet —— 多任务轻量级驾驶员监控网络

整体架构:
  ┌────────────────┐
  │  Input Image   │
  └───────┬────────┘
          │
  ┌───────▼────────┐
  │   MobileNetV3  │  ← 轻量级 Backbone (共享特征提取)
  │   (Small)      │
  └──┬────┬────┬───┘
     C3   C4   C5       ← 多尺度特征 (stride 8/16/32)
  ┌──▼────▼────▼───┐
  │    Lite-FPN     │  ← 特征融合
  └──┬────┬────┬───┘
     P3   P4   P5       ← 融合后特征 (统一通道数)
     │    │    │
  ┌──▼────▼────▼───┐  ┌──▼────▼────▼───┐
  │ Detection Head │  │ Landmark Head  │
  │ (cls + bbox)   │  │ (关键点回归)     │
  └────────────────┘  └────────────────┘
  face/phone/cigarette   EAR → PERCLOS
                         MAR → 打哈欠
"""

import torch
import torch.nn as nn

from .backbone import (MobileNetV3Backbone, MobileNetV3LargeBackbone,
                        TimmBackbone)
from .neck import LiteFPN, BiFPN
from .heads import DetectionHead, LandmarkHead


class DMSNet(nn.Module):
    """多任务 DMS 网络: 共享 Backbone + FPN, 双 Head 并行输出."""

    def __init__(self,
                 num_classes: int = 3,
                 num_landmarks: int = 14,
                 num_anchors: int = 2,
                 fpn_channels: int = 64,
                 backbone_type: str = "small",
                 neck_type: str = "lite_fpn",
                 bifpn_layers: int = 1,
                 pretrained: bool = True,
                 timm_model_name: str = "mobilenetv4_conv_small"):
        super().__init__()

        self.num_classes = num_classes
        self.num_landmarks = num_landmarks
        self.num_anchors = num_anchors

        # ---- Backbone ----
        if backbone_type == "timm":
            self.backbone = TimmBackbone(model_name=timm_model_name,
                                         pretrained=pretrained)
            in_channels = self.backbone.OUT_CHANNELS
        elif backbone_type == "large":
            self.backbone = MobileNetV3LargeBackbone(pretrained=pretrained)
            in_channels = MobileNetV3LargeBackbone.OUT_CHANNELS
        else:
            self.backbone = MobileNetV3Backbone(pretrained=pretrained)
            in_channels = MobileNetV3Backbone.OUT_CHANNELS

        # ---- Neck ----
        if neck_type == "bifpn":
            self.fpn = BiFPN(in_channels=in_channels, fpn_channels=fpn_channels,
                             num_layers=bifpn_layers)
        else:
            self.fpn = LiteFPN(in_channels=in_channels, fpn_channels=fpn_channels)

        # ---- Head 1: 目标检测 (face / phone / cigarette) ----
        self.det_head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

        # ---- Head 2: 面部关键点回归 ----
        self.lmk_head = LandmarkHead(
            in_channels=fpn_channels,
            num_landmarks=num_landmarks,
            num_anchors=num_anchors,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) 输入图像
        Returns:
            cls_scores:    list[Tensor], 每层 FPN 的分类得分
            bbox_preds:    list[Tensor], 每层 FPN 的边框偏移
            landmark_preds: list[Tensor], 每层 FPN 的关键点偏移
        """
        # 共享特征提取
        c3, c4, c5 = self.backbone(x)
        features = self.fpn(c3, c4, c5)  # (P3, P4, P5)

        # 两个 Head 并行
        cls_scores, bbox_preds = self.det_head(features)
        landmark_preds = self.lmk_head(features)

        return cls_scores, bbox_preds, landmark_preds

    def get_fpn_features(self, x: torch.Tensor):
        """导出 FPN 中间特征, 用于知识蒸馏."""
        c3, c4, c5 = self.backbone(x)
        return self.fpn(c3, c4, c5)


def build_model(cfg: dict) -> DMSNet:
    """从配置文件构建模型."""
    model_cfg = cfg["model"]
    backbone_map = {
        "mobilenet_v3_small": "small",
        "mobilenet_v3_large": "large",
    }
    backbone_type = backbone_map.get(model_cfg["backbone"], model_cfg["backbone"])
    return DMSNet(
        num_classes=model_cfg["num_classes"],
        num_landmarks=model_cfg["num_landmarks"],
        num_anchors=model_cfg["num_anchors"],
        fpn_channels=model_cfg["fpn_channels"],
        backbone_type=backbone_type,
        neck_type=model_cfg.get("neck", "lite_fpn"),
        bifpn_layers=model_cfg.get("bifpn_layers", 1),
        pretrained=model_cfg.get("pretrained", True),
        timm_model_name=model_cfg.get("timm_model_name",
                                       "mobilenetv4_conv_small"),
    )
