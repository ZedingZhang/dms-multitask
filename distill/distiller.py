"""
知识蒸馏模块 —— Teacher-Student 架构

蒸馏策略:
  1. 特征蒸馏 (Feature Distillation): FPN 各层特征 MSE Loss
  2. Logit 蒸馏 (Logit Distillation): 分类输出 KL 散度 + 软标签
  3. 边框/关键点蒸馏: Teacher 预测作为软目标辅助 Student 学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignModule(nn.Module):
    """通道对齐: 当 Teacher/Student FPN 通道数不同时做 1x1 映射."""

    def __init__(self, student_ch: int, teacher_ch: int):
        super().__init__()
        self.align = nn.Conv2d(student_ch, teacher_ch, 1, bias=False)

    def forward(self, x):
        return self.align(x)


class MultiTaskDistiller(nn.Module):
    """
    多任务知识蒸馏器.

    用法:
        teacher = DMSNet(backbone_type="large")
        student = DMSNet(backbone_type="small")
        distiller = MultiTaskDistiller(student, teacher, cfg)

        for images, targets in loader:
            # 正常前向
            s_cls, s_bbox, s_lmk = student(images)
            task_loss = compute_task_loss(...)

            # 蒸馏损失
            distill_loss = distiller(images, (s_cls, s_bbox, s_lmk))
            total_loss = task_loss + distill_loss
    """

    def __init__(self, student: nn.Module, teacher: nn.Module,
                 temperature: float = 4.0,
                 alpha_feat: float = 50.0,
                 alpha_logit: float = 1.0,
                 alpha_resp: float = 0.5,
                 student_fpn_ch: int = 64,
                 teacher_fpn_ch: int = 64):
        super().__init__()

        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha_feat = alpha_feat
        self.alpha_logit = alpha_logit
        self.alpha_resp = alpha_resp

        # 冻结 Teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # 特征对齐层 (如果通道数不同)
        if student_fpn_ch != teacher_fpn_ch:
            self.align_layers = nn.ModuleList([
                FeatureAlignModule(student_fpn_ch, teacher_fpn_ch)
                for _ in range(3)  # P3, P4, P5
            ])
        else:
            self.align_layers = None

    @torch.no_grad()
    def _teacher_forward(self, x):
        self.teacher.eval()
        t_feats = self.teacher.get_fpn_features(x)
        t_cls_scores, t_bbox_preds = self.teacher.det_head(t_feats)
        t_lmk_preds = self.teacher.lmk_head(t_feats)
        return t_feats, t_cls_scores, t_bbox_preds, t_lmk_preds

    def feature_distill_loss(self, s_feats, t_feats):
        """FPN 各层特征的 MSE 蒸馏损失."""
        loss = 0.0
        for i, (sf, tf) in enumerate(zip(s_feats, t_feats)):
            if self.align_layers is not None:
                sf = self.align_layers[i](sf)
            # 尺寸对齐
            if sf.shape != tf.shape:
                sf = F.interpolate(sf, size=tf.shape[2:], mode="bilinear",
                                   align_corners=False)
            loss += F.mse_loss(sf, tf)
        return loss / len(s_feats)

    def logit_distill_loss(self, s_cls_list, t_cls_list):
        """分类 logit 的 KL 散度蒸馏损失."""
        T = self.temperature
        loss = 0.0
        for s_cls, t_cls in zip(s_cls_list, t_cls_list):
            # 展平空间维度
            B = s_cls.shape[0]
            s_flat = s_cls.reshape(B, -1)
            t_flat = t_cls.reshape(B, -1)

            s_log_soft = F.log_softmax(s_flat / T, dim=-1)
            t_soft = F.softmax(t_flat / T, dim=-1)
            loss += F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (T ** 2)
        return loss / len(s_cls_list)

    def response_distill_loss(self, s_reg_list, t_reg_list):
        """回归 Head (bbox / landmark) 的 Smooth-L1 蒸馏."""
        loss = 0.0
        for s_reg, t_reg in zip(s_reg_list, t_reg_list):
            if s_reg.shape != t_reg.shape:
                s_reg = F.interpolate(s_reg, size=t_reg.shape[2:],
                                      mode="nearest")
            loss += F.smooth_l1_loss(s_reg, t_reg)
        return loss / len(s_reg_list)

    def forward(self, images, student_outputs):
        """
        计算总蒸馏损失.

        Args:
            images: 输入图像
            student_outputs: (s_cls_scores, s_bbox_preds, s_lmk_preds)
        Returns:
            distill_loss: 标量
        """
        s_cls_scores, s_bbox_preds, s_lmk_preds = student_outputs

        # Teacher 前向
        t_feats, t_cls_scores, t_bbox_preds, t_lmk_preds = \
            self._teacher_forward(images)

        # Student FPN 特征
        s_feats = self.student.get_fpn_features(images)

        # 1. 特征蒸馏
        feat_loss = self.feature_distill_loss(s_feats, t_feats)

        # 2. 分类 logit 蒸馏
        logit_loss = self.logit_distill_loss(s_cls_scores, t_cls_scores)

        # 3. 边框 + 关键点响应蒸馏
        resp_loss = self.response_distill_loss(s_bbox_preds, t_bbox_preds) + \
                    self.response_distill_loss(s_lmk_preds, t_lmk_preds)

        total = (self.alpha_feat * feat_loss +
                 self.alpha_logit * logit_loss +
                 self.alpha_resp * resp_loss)
        return total
