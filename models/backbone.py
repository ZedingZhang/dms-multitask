"""
MobileNetV3-Small Backbone —— 提取多尺度特征用于 FPN
输出三个尺度的特征图: C3(stride=8), C4(stride=16), C5(stride=32)
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Backbone(nn.Module):
    """轻量级 Backbone: 基于 MobileNetV3-Small, 输出多尺度特征."""

    # MobileNetV3-Small 各 stage 对应的 features 索引与输出通道数
    STAGE_INDICES = {
        "C3": (0, 4),    # stride=8,  out_channels=24
        "C4": (4, 9),    # stride=16, out_channels=48
        "C5": (9, 13),   # stride=32, out_channels=576 (含最后 1x1 Conv)
    }
    OUT_CHANNELS = (24, 48, 576)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        features = list(backbone.features)

        # 拆分为三段, 各自产出一个尺度的特征
        s = self.STAGE_INDICES
        self.stage1 = nn.Sequential(*features[s["C3"][0]:s["C3"][1]])  # -> C3
        self.stage2 = nn.Sequential(*features[s["C4"][0]:s["C4"][1]])  # -> C4
        self.stage3 = nn.Sequential(*features[s["C5"][0]:s["C5"][1]])  # -> C5

    def forward(self, x: torch.Tensor):
        """返回 (C3, C4, C5) 三个尺度的特征图."""
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return c3, c4, c5


class MobileNetV3LargeBackbone(nn.Module):
    """Teacher 网络使用的大 Backbone: MobileNetV3-Large."""

    STAGE_INDICES = {
        "C3": (0, 7),    # stride=8,  out_channels=40
        "C4": (7, 13),   # stride=16, out_channels=112
        "C5": (13, 17),  # stride=32, out_channels=960
    }
    OUT_CHANNELS = (40, 112, 960)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        features = list(backbone.features)

        s = self.STAGE_INDICES
        self.stage1 = nn.Sequential(*features[s["C3"][0]:s["C3"][1]])
        self.stage2 = nn.Sequential(*features[s["C4"][0]:s["C4"][1]])
        self.stage3 = nn.Sequential(*features[s["C5"][0]:s["C5"][1]])

    def forward(self, x: torch.Tensor):
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return c3, c4, c5


class TimmBackbone(nn.Module):
    """通用 timm backbone 包装器 — 输出 stride 8/16/32 多尺度特征.

    通过 `timm` 库的 features_only 模式提取中间层特征,
    自动匹配最接近 [8, 16, 32] 倍下采样的 stage.

    推荐模型 (按效率排序):
      - mobilenetv4_conv_small / mobilenetv4_conv_medium  (MobileNetV4)
      - repvit_m1_1 / repvit_m2_3                          (RepViT)
      - fastvit_t8 / fastvit_t12                            (FastViT)
      - efficientformerv2_s0 / efficientformerv2_s1         (EfficientFormerV2)
      - mobileone_s0 / mobileone_s1                         (MobileOne)

    用法:
        backbone = TimmBackbone("repvit_m1_1", pretrained=True)
        c3, c4, c5 = backbone(x)  # stride 8, 16, 32
        print(backbone.OUT_CHANNELS)  # e.g. (48, 96, 192)
    """

    def __init__(self, model_name: str = "mobilenetv4_conv_small",
                 pretrained: bool = True, target_strides: tuple = (8, 16, 32)):
        super().__init__()
        import timm
        self.model = timm.create_model(
            model_name, pretrained=pretrained, features_only=True)

        reductions = self.model.feature_info.reduction()
        channels = self.model.feature_info.channels()

        # 自动匹配最接近 target strides 的 stage
        self.out_indices = self._match_strides(reductions, target_strides)
        self.OUT_CHANNELS = tuple(channels[i] for i in self.out_indices)

    @staticmethod
    def _match_strides(reductions: list, targets: tuple) -> list:
        indices = []
        used = set()
        for t in targets:
            best = min(
                (i for i, r in enumerate(reductions) if i not in used),
                key=lambda i: abs(reductions[i] - t),
            )
            indices.append(best)
            used.add(best)
        return indices

    def forward(self, x: torch.Tensor):
        feats = self.model(x)  # list of feature maps
        return (feats[self.out_indices[0]],
                feats[self.out_indices[1]],
                feats[self.out_indices[2]])
