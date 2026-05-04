"""
Landmark Head —— 面部关键点回归头
对检测到的人脸区域预测关键点坐标, 用于 PERCLOS / 打哈欠检测.

关键点定义 (默认 14 个点):
  - 左眼  6 个点 (p0-p5): 用于计算左眼 EAR
  - 右眼  6 个点 (p6-p11): 用于计算右眼 EAR
  - 嘴唇  2 个点 (p12-p13): 上唇中心 / 下唇中心, 用于计算 MAR (嘴巴张开程度)
"""

import torch
import torch.nn as nn


class LandmarkHead(nn.Module):
    """
    轻量级关键点回归头.
    与 DetectionHead 共享 FPN 特征, 对每个 anchor 位置回归关键点偏移.
    """

    def __init__(self, in_channels: int = 64, num_landmarks: int = 14,
                 num_anchors: int = 2, num_conv: int = 2):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_anchors = num_anchors

        layers = []
        for _ in range(num_conv):
            layers += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
            ]
        self.subnet = nn.Sequential(*layers)

        # 每个 anchor 输出 num_landmarks*2 个坐标值 (x, y)
        self.output = nn.Conv2d(in_channels, num_anchors * num_landmarks * 2, 1)

        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(self.output.bias, 0)

    def forward_single(self, feat: torch.Tensor):
        return self.output(self.subnet(feat))

    def forward(self, features: tuple):
        """
        Args:
            features: (P3, P4, P5) FPN 特征
        Returns:
            landmark_preds: list of (B, num_anchors*num_landmarks*2, Hi, Wi)
        """
        return [self.forward_single(f) for f in features]
