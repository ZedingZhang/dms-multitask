"""
Lite-FPN (Feature Pyramid Network) —— 轻量级特征融合
将 Backbone 输出的 C3/C4/C5 融合为统一通道数的 P3/P4/P5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积, 用于降低 FPN 计算量."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, padding=kernel // 2,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class LiteFPN(nn.Module):
    """轻量级 FPN: 将多尺度特征统一到 fpn_channels 通道并自顶向下融合."""

    def __init__(self, in_channels: tuple = (24, 48, 576), fpn_channels: int = 64):
        super().__init__()
        # 1x1 横向连接, 统一通道数
        self.lateral5 = nn.Conv2d(in_channels[2], fpn_channels, 1)
        self.lateral4 = nn.Conv2d(in_channels[1], fpn_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels[0], fpn_channels, 1)

        # 融合后的平滑卷积 (深度可分离)
        self.smooth5 = DepthwiseSeparableConv(fpn_channels, fpn_channels)
        self.smooth4 = DepthwiseSeparableConv(fpn_channels, fpn_channels)
        self.smooth3 = DepthwiseSeparableConv(fpn_channels, fpn_channels)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor):
        # 横向连接
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3)

        # 自顶向下融合
        p4 = p4 + F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode="nearest")

        # 平滑
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)

        return p3, p4, p5


class BiFPNBlock(nn.Module):
    """单层 BiFPN: 双向跨尺度连接 + 快速归一化加权融合.

    Top-down:     P5 → P4 → P3   (语义信息向下传递)
    Bottom-up:    P3 → P4 → P5   (位置信息向上传递)

    每个融合节点用 learnable weights 做加权求和而非简单加法:
        out = Σ ReLU(w_i) / (Σ ReLU(w_j) + ε) × feat_i
    """

    def __init__(self, channels: int):
        super().__init__()
        # Top-down 平滑卷积
        self.td_p5 = DepthwiseSeparableConv(channels, channels)
        self.td_p4 = DepthwiseSeparableConv(channels, channels)
        self.td_p3 = DepthwiseSeparableConv(channels, channels)

        # Bottom-up 平滑卷积
        self.out_p3 = DepthwiseSeparableConv(channels, channels)
        self.out_p4 = DepthwiseSeparableConv(channels, channels)
        self.out_p5 = DepthwiseSeparableConv(channels, channels)

        # 下采样 (stride-2 DW Conv)
        self.down_p3 = nn.Conv2d(channels, channels, 3, stride=2,
                                  padding=1, groups=channels, bias=False)
        self.down_p4 = nn.Conv2d(channels, channels, 3, stride=2,
                                  padding=1, groups=channels, bias=False)

        # 快速归一化融合权重
        self.w4_td = nn.Parameter(torch.ones(2))   # [P4_in, P5_td_up]
        self.w3_td = nn.Parameter(torch.ones(2))   # [P3_in, P4_td_up]
        self.w3_out = nn.Parameter(torch.ones(2))  # [P3_in, P3_td]
        self.w4_out = nn.Parameter(torch.ones(3))  # [P4_in, P4_td, P3_out_dn]
        self.w5_out = nn.Parameter(torch.ones(3))  # [P5_in, P5_td, P4_out_dn]

    @staticmethod
    def _fuse(inputs, w):
        """快速归一化融合: w' = ReLU(w), 归一化后加权求和."""
        wr = torch.relu(w)
        wn = wr / (wr.sum() + 1e-4)
        return sum(wn[i] * inputs[i] for i in range(len(inputs)))

    def forward(self, p3, p4, p5):
        # ---- Top-down ----
        p5_td = self.td_p5(p5)
        p4_td = self.td_p4(self._fuse(
            [p4, F.interpolate(p5_td, p4.shape[2:], mode="nearest")],
            self.w4_td,
        ))
        p3_td = self.td_p3(self._fuse(
            [p3, F.interpolate(p4_td, p3.shape[2:], mode="nearest")],
            self.w3_td,
        ))

        # ---- Bottom-up ----
        p3_out = self.out_p3(self._fuse([p3, p3_td], self.w3_out))
        p4_out = self.out_p4(self._fuse(
            [p4, p4_td, self.down_p3(p3_out)], self.w4_out,
        ))
        p5_out = self.out_p5(self._fuse(
            [p5, p5_td, self.down_p4(p4_out)], self.w5_out,
        ))

        return p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    """BiFPN 颈部: 堆叠 num_layers 个 BiFPNBlock.

    与 LiteFPN 接口完全兼容, 接收 backbone 的三尺度特征 (C3, C4, C5),
    输出融合后的三尺度特征 (P3, P4, P5).

    Args:
        in_channels:  backbone 三个尺度的通道数
        fpn_channels: FPN 统一通道数
        num_layers:   BiFPN 块堆叠次数 (≥1, 推荐 1~3)
    """

    def __init__(self, in_channels: tuple = (24, 48, 576),
                 fpn_channels: int = 64, num_layers: int = 1):
        super().__init__()
        # 通道投影 (backbone → FPN 统一通道)
        self.p3_proj = nn.Conv2d(in_channels[0], fpn_channels, 1)
        self.p4_proj = nn.Conv2d(in_channels[1], fpn_channels, 1)
        self.p5_proj = nn.Conv2d(in_channels[2], fpn_channels, 1)

        self.blocks = nn.ModuleList([
            BiFPNBlock(fpn_channels) for _ in range(num_layers)
        ])

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor):
        p3 = self.p3_proj(c3)
        p4 = self.p4_proj(c4)
        p5 = self.p5_proj(c5)
        for block in self.blocks:
            p3, p4, p5 = block(p3, p4, p5)
        return p3, p4, p5
