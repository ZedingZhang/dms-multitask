"""
多任务损失函数:
  1. Focal Loss   —— 目标分类
  2. Smooth L1    —— 边框回归
  3. Wing Loss    —— 关键点回归 (对小偏差更敏感)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    """Focal Loss: 解决正负样本极度不均衡的问题."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred:   (N, num_classes) logits
        target: (N,) class indices, -1 表示忽略
        """
        valid = target >= 0
        pred = pred[valid]
        target = target[valid]

        if pred.numel() == 0:
            return pred.sum() * 0.0

        ce = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


class WingLoss(nn.Module):
    """Wing Loss: 关键点回归专用, 对小误差比 L1 更敏感."""

    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        pred:   (N, num_landmarks*2)
        target: (N, num_landmarks*2)
        mask:   (N,) bool, 仅对人脸正样本计算
        """
        pred = pred[mask]
        target = target[mask]

        if pred.numel() == 0:
            return pred.sum() * 0.0

        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.epsilon),
            diff - self.C,
        )
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """多任务联合损失: cls + bbox + landmark.

    支持两种加权方式:
      - 固定权重 (uncertainty=False): 使用 cls/bbox/lmk_weight 超参
      - 不确定性加权 (uncertainty=True):  学习每个任务的最优权重
        参考 Kendall et al. "Multi-Task Learning Using Uncertainty to
        Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)

        loss = exp(-s_i) * task_loss + s_i
        其中 s_i = log(σ_i²) 是可学习参数, 优化器自动平衡三个任务的收敛速度
    """

    def __init__(self, cls_weight: float = 2.0, bbox_weight: float = 1.0,
                 lmk_weight: float = 1.0, uncertainty: bool = False):
        super().__init__()
        self.cls_loss_fn = FocalLoss()
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction="mean")
        self.lmk_loss_fn = WingLoss()

        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.lmk_weight = lmk_weight
        self.uncertainty = uncertainty

        if uncertainty:
            # s = log(σ²), 初始 s=0 → σ²=1 → 初始权重 exp(-0)=1
            self.log_sigma_cls = nn.Parameter(torch.zeros(1))
            self.log_sigma_bbox = nn.Parameter(torch.zeros(1))
            self.log_sigma_lmk = nn.Parameter(torch.zeros(1))
        else:
            self.log_sigma_cls = None
            self.log_sigma_bbox = None
            self.log_sigma_lmk = None

    def forward(self, predictions: dict, targets: dict):
        """
        predictions:
            cls_preds:  (N, num_classes)   所有 anchor 的分类 logits
            bbox_preds: (N, 4)             所有 anchor 的边框偏移
            lmk_preds:  (N, num_landmarks*2) 所有 anchor 的关键点偏移
        targets:
            cls_targets:  (N,)             类别标签 (-1=忽略, 0~C-1=正类)
            bbox_targets: (N, 4)           回归目标
            lmk_targets:  (N, num_landmarks*2) 关键点目标
            face_mask:    (N,) bool        是否为人脸正样本 (仅对这些计算关键点损失)
        """
        cls_loss = self.cls_loss_fn(predictions["cls_preds"],
                                    targets["cls_targets"])
        # bbox loss 仅对正样本计算
        pos_mask = targets["cls_targets"] > 0
        if pos_mask.any():
            bbox_loss = self.bbox_loss_fn(predictions["bbox_preds"][pos_mask],
                                          targets["bbox_targets"][pos_mask])
        else:
            bbox_loss = predictions["bbox_preds"].sum() * 0.0

        lmk_loss = self.lmk_loss_fn(predictions["lmk_preds"],
                                     targets["lmk_targets"],
                                     targets["face_mask"])

        if self.uncertainty:
            # 学习权重: w_i = exp(-s_i), 正则项: + s_i
            w_cls = torch.exp(-self.log_sigma_cls)
            w_bbox = torch.exp(-self.log_sigma_bbox)
            w_lmk = torch.exp(-self.log_sigma_lmk)

            total = (w_cls * cls_loss + self.log_sigma_cls +
                     w_bbox * bbox_loss + self.log_sigma_bbox +
                     w_lmk * lmk_loss + self.log_sigma_lmk)
        else:
            total = (self.cls_weight * cls_loss +
                     self.bbox_weight * bbox_loss +
                     self.lmk_weight * lmk_loss)

        return {
            "total_loss": total,
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "lmk_loss": lmk_loss.detach(),
        }
