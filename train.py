"""
DMS 多任务模型训练脚本

支持:
  - 普通训练 (Backbone + FPN + DetHead + LmkHead)
  - 知识蒸馏训练 (Teacher: MobileNetV3-Large → Student: MobileNetV3-Small)
  - Cosine LR Schedule + Warmup
  - TensorBoard 日志
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.dms_net import DMSNet, build_model
from data.dataset import DMSDataset, collate_fn
from data.augment import get_train_transforms, get_val_transforms
from utils.loss import MultiTaskLoss
from utils.anchors import AnchorGenerator
from distill.distiller import MultiTaskDistiller


def match_anchors_to_targets(anchors, targets, num_classes, num_landmarks,
                              pos_iou=0.35, neg_iou=0.2):
    """
    将 anchor 与 GT 匹配, 生成训练标签.

    Returns:
        cls_targets:  (N,)   类别标签, -1=忽略, 0=背景(负样本), 1~C=正类
        bbox_targets: (N, 4) 边框回归目标
        lmk_targets:  (N, num_landmarks*2) 关键点目标
        face_mask:    (N,) bool  是否是人脸正样本
    """
    N = anchors.shape[0]
    cls_targets = torch.zeros(N, dtype=torch.long)       # 默认背景
    bbox_targets = torch.zeros(N, 4)
    lmk_targets = torch.zeros(N, num_landmarks * 2)
    face_mask = torch.zeros(N, dtype=torch.bool)

    cls_ids = targets["cls_ids"]
    bboxes = targets["bboxes"]
    landmarks = targets["landmarks"]

    if len(cls_ids) == 0:
        return cls_targets, bbox_targets, lmk_targets, face_mask

    # 计算 IoU: anchors (cx,cy,w,h) → (x1,y1,x2,y2)
    ax1 = anchors[:, 0] - anchors[:, 2] / 2
    ay1 = anchors[:, 1] - anchors[:, 3] / 2
    ax2 = anchors[:, 0] + anchors[:, 2] / 2
    ay2 = anchors[:, 1] + anchors[:, 3] / 2

    gx1, gy1, gx2, gy2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # IoU 矩阵 (N, M)
    inter_x1 = torch.max(ax1.unsqueeze(1), gx1.unsqueeze(0))
    inter_y1 = torch.max(ay1.unsqueeze(1), gy1.unsqueeze(0))
    inter_x2 = torch.min(ax2.unsqueeze(1), gx2.unsqueeze(0))
    inter_y2 = torch.min(ay2.unsqueeze(1), gy2.unsqueeze(0))

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    anchor_area = (ax2 - ax1) * (ay2 - ay1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union = anchor_area.unsqueeze(1) + gt_area.unsqueeze(0) - inter_area
    iou = inter_area / (union + 1e-6)

    # 每个 anchor 匹配最佳 GT
    best_iou, best_gt_idx = iou.max(dim=1)

    # 正样本
    pos = best_iou >= pos_iou
    # 忽略区间
    ignore = (~pos) & (best_iou >= neg_iou)

    cls_targets[pos] = cls_ids[best_gt_idx[pos]] + 1   # +1 因为 0 是背景
    cls_targets[ignore] = -1                             # 忽略

    # 边框回归目标 (编码为相对 anchor 的偏移)
    matched_gt = bboxes[best_gt_idx]
    gt_cx = (matched_gt[:, 0] + matched_gt[:, 2]) / 2
    gt_cy = (matched_gt[:, 1] + matched_gt[:, 3]) / 2
    gt_w = matched_gt[:, 2] - matched_gt[:, 0]
    gt_h = matched_gt[:, 3] - matched_gt[:, 1]

    bbox_targets[:, 0] = (gt_cx - anchors[:, 0]) / (anchors[:, 2] + 1e-6)
    bbox_targets[:, 1] = (gt_cy - anchors[:, 1]) / (anchors[:, 3] + 1e-6)
    bbox_targets[:, 2] = torch.log(gt_w / (anchors[:, 2] + 1e-6) + 1e-6)
    bbox_targets[:, 3] = torch.log(gt_h / (anchors[:, 3] + 1e-6) + 1e-6)

    # 关键点目标 (仅人脸)
    matched_lmk = landmarks[best_gt_idx]
    lmk_targets = matched_lmk.clone()
    face_mask = pos & (cls_ids[best_gt_idx] == 0)   # class 0 = face

    return cls_targets, bbox_targets, lmk_targets, face_mask


def match_anchors_to_targets_batch(anchors, targets_list, num_classes, num_landmarks,
                                   pos_iou=0.35, neg_iou=0.2):
    """批量 anchor 匹配: 一次处理整个 batch, 避免逐样本 CPU-GPU 往返.

    原理: 将 batch 内所有 GT 填充到统一长度, 利用广播一次计算
    (N_anchors × B × max_M) 的 IoU 矩阵, 然后 batch 维度并行匹配.

    Returns:
        cls_targets:  (B, N) 类别标签 (已 on device)
        bbox_targets: (B, N, 4) 回归目标
        lmk_targets:  (B, N, L*2) 关键点目标
        face_mask:    (B, N) bool
    """
    device = anchors.device
    B = len(targets_list)
    N = anchors.shape[0]

    cls_targets = torch.zeros(B, N, dtype=torch.long, device=device)
    bbox_targets = torch.zeros(B, N, 4, device=device)
    lmk_targets = torch.zeros(B, N, num_landmarks * 2, device=device)
    face_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    # Padding GT 到 max_M
    max_M = max((len(t["cls_ids"]) for t in targets_list), default=0)
    if max_M == 0:
        return cls_targets, bbox_targets, lmk_targets, face_mask

    cls_ids_pad = torch.zeros(B, max_M, dtype=torch.long, device=device)
    bboxes_pad = torch.zeros(B, max_M, 4, device=device)
    lmks_pad = torch.zeros(B, max_M, num_landmarks * 2, device=device)
    valid_gt = torch.zeros(B, max_M, dtype=torch.bool, device=device)

    for b, t in enumerate(targets_list):
        M = len(t["cls_ids"])
        if M > 0:
            cls_ids_pad[b, :M] = t["cls_ids"].to(device)
            bboxes_pad[b, :M] = t["bboxes"].to(device)
            lmks_pad[b, :M] = t["landmarks"].to(device)
            valid_gt[b, :M] = True

    # IoU 矩阵: anchors (N, 4) × GT (B, M, 4) → (N, B, M)
    ax1 = anchors[:, 0] - anchors[:, 2] / 2  # (N,)
    ay1 = anchors[:, 1] - anchors[:, 3] / 2
    ax2 = anchors[:, 0] + anchors[:, 2] / 2
    ay2 = anchors[:, 1] + anchors[:, 3] / 2

    gx1, gy1 = bboxes_pad[:, :, 0], bboxes_pad[:, :, 1]
    gx2, gy2 = bboxes_pad[:, :, 2], bboxes_pad[:, :, 3]

    inter_x1 = torch.max(ax1[:, None, None], gx1[None, :, :])
    inter_y1 = torch.max(ay1[:, None, None], gy1[None, :, :])
    inter_x2 = torch.min(ax2[:, None, None], gx2[None, :, :])
    inter_y2 = torch.min(ay2[:, None, None], gy2[None, :, :])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    anchor_area = (ax2 - ax1) * (ay2 - ay1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union = anchor_area[:, None, None] + gt_area[None, :, :] - inter
    iou = inter / (union + 1e-6)  # (N, B, M)

    # 屏蔽填充位置
    iou = iou.masked_fill(~valid_gt[None, :, :], 0.0)

    # 按 batch 维度排列: (B, N, M)
    iou = iou.permute(1, 0, 2)
    best_iou, best_gt_idx = iou.max(dim=2)  # (B, N)

    pos = best_iou >= pos_iou
    ignore = (~pos) & (best_iou >= neg_iou)

    # 类别标签
    matched_cls = cls_ids_pad.gather(1, best_gt_idx)  # (B, N)
    cls_targets[pos] = matched_cls[pos] + 1
    cls_targets[ignore] = -1

    # 边框回归目标
    matched_bbox = bboxes_pad.gather(
        1, best_gt_idx[:, :, None].expand(-1, -1, 4))  # (B, N, 4)
    gt_cx = (matched_bbox[:, :, 0] + matched_bbox[:, :, 2]) / 2
    gt_cy = (matched_bbox[:, :, 1] + matched_bbox[:, :, 3]) / 2
    gt_w = matched_bbox[:, :, 2] - matched_bbox[:, :, 0]
    gt_h = matched_bbox[:, :, 3] - matched_bbox[:, :, 1]

    aw, ah = anchors[:, 2:3], anchors[:, 3:4]  # (N, 1) → broadcasts to (1, N) → (B, N)
    bbox_targets[:, :, 0] = (gt_cx - anchors[None, :, 0]) / (anchors[None, :, 2] + 1e-6)
    bbox_targets[:, :, 1] = (gt_cy - anchors[None, :, 1]) / (anchors[None, :, 3] + 1e-6)
    bbox_targets[:, :, 2] = torch.log(gt_w / (anchors[None, :, 2] + 1e-6) + 1e-6)
    bbox_targets[:, :, 3] = torch.log(gt_h / (anchors[None, :, 3] + 1e-6) + 1e-6)

    # 关键点目标
    matched_lmk = lmks_pad.gather(
        1, best_gt_idx[:, :, None].expand(-1, -1, num_landmarks * 2))
    lmk_targets = matched_lmk
    face_mask = pos & (matched_cls == 0)

    return cls_targets, bbox_targets, lmk_targets, face_mask


class ModelEMA:
    """模型权重指数移动平均 (Exponential Moving Average).

    维护一份 shadow 参数, 每个 step 后做加权平均:
        shadow = decay * shadow + (1 - decay) * model_params

    验证和保存时切换到 shadow, 通常带来 0.5~1% 精度提升。

    用法:
        ema = ModelEMA(model, decay=0.9998)
        for ...:
            train_one_epoch(...)
            ema.update()  # 每个 step 后调用
        ema.apply_shadow()
        torch.save(model.state_dict(), "best.pt")
        ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


def train_one_epoch(model, loader, criterion, optimizer, anchor_gen, device, cfg,
                    scaler=None, ema=None):
    model.train()
    total_losses = {"total_loss": 0, "cls_loss": 0, "bbox_loss": 0, "lmk_loss": 0}
    num_batches = 0
    use_amp = scaler is not None

    num_classes = cfg["model"]["num_classes"]
    num_landmarks = cfg["model"]["num_landmarks"]
    num_anchors = cfg["model"]["num_anchors"]

    pbar = tqdm(loader, desc="Training")
    for images, targets_list in pbar:
        images = images.to(device)

        # 前向传播使用 FP16 自动混合精度
        with autocast(enabled=use_amp):
            cls_scores, bbox_preds, lmk_preds = model(images)

            B = images.shape[0]

            # 拉平所有 FPN 层的输出
            all_cls = []
            all_bbox = []
            all_lmk = []
            for cs, bp, lp in zip(cls_scores, bbox_preds, lmk_preds):
                H, W = cs.shape[2:]
                # (B, A*C, H, W) -> (B, H*W*A, C)
                all_cls.append(
                    cs.reshape(B, num_anchors, num_classes, H, W)
                      .permute(0, 3, 4, 1, 2).reshape(B, -1, num_classes))
                all_bbox.append(
                    bp.reshape(B, num_anchors, 4, H, W)
                      .permute(0, 3, 4, 1, 2).reshape(B, -1, 4))
                all_lmk.append(
                    lp.reshape(B, num_anchors, num_landmarks * 2, H, W)
                      .permute(0, 3, 4, 1, 2).reshape(B, -1, num_landmarks * 2))

            all_cls = torch.cat(all_cls, dim=1)    # (B, N_total, C)
            all_bbox = torch.cat(all_bbox, dim=1)  # (B, N_total, 4)
            all_lmk = torch.cat(all_lmk, dim=1)   # (B, N_total, L*2)

            # 生成 anchors
            feat_maps = [(cs.shape[2], cs.shape[3]) for cs in cls_scores]
            anchors = anchor_gen(feat_maps).to(device)

            # 批量匹配: 一次完成整个 batch 的锚框匹配
            cls_t, bbox_t, lmk_t, fmask = match_anchors_to_targets_batch(
                anchors, targets_list, num_classes, num_landmarks)
            # all: (B, N_total, *)

            # 逐样本损失汇总 (损失函数需单样本输入)
            batch_loss = {"total_loss": 0, "cls_loss": 0, "bbox_loss": 0, "lmk_loss": 0}
            for b in range(B):
                loss_dict = criterion(
                    {"cls_preds": all_cls[b], "bbox_preds": all_bbox[b],
                     "lmk_preds": all_lmk[b]},
                    {"cls_targets": cls_t[b], "bbox_targets": bbox_t[b],
                     "lmk_targets": lmk_t[b], "face_mask": fmask[b]},
                )
                for k in batch_loss:
                    batch_loss[k] += loss_dict[k]

            loss = batch_loss["total_loss"] / B

        # 反向传播
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update()

        for k in total_losses:
            total_losses[k] += batch_loss[k].item() / B
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{total_losses['total_loss'] / num_batches:.4f}",
            "cls": f"{total_losses['cls_loss'] / num_batches:.4f}",
            "bbox": f"{total_losses['bbox_loss'] / num_batches:.4f}",
            "lmk": f"{total_losses['lmk_loss'] / num_batches:.4f}",
        })

    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, anchor_gen, device, cfg):
    """在验证集上计算多任务损失."""
    model.eval()
    val_losses = {"total_loss": 0, "cls_loss": 0, "bbox_loss": 0, "lmk_loss": 0}
    num_batches = 0

    num_classes = cfg["model"]["num_classes"]
    num_landmarks = cfg["model"]["num_landmarks"]
    num_anchors = cfg["model"]["num_anchors"]

    pbar = tqdm(loader, desc="Validation", leave=False)
    for images, targets_list in pbar:
        images = images.to(device)
        cls_scores, bbox_preds, lmk_preds = model(images)

        B = images.shape[0]

        # 拉平所有 FPN 层的输出
        all_cls, all_bbox, all_lmk = [], [], []
        for cs, bp, lp in zip(cls_scores, bbox_preds, lmk_preds):
            H, W = cs.shape[2:]
            all_cls.append(
                cs.reshape(B, num_anchors, num_classes, H, W)
                  .permute(0, 3, 4, 1, 2).reshape(B, -1, num_classes))
            all_bbox.append(
                bp.reshape(B, num_anchors, 4, H, W)
                  .permute(0, 3, 4, 1, 2).reshape(B, -1, 4))
            all_lmk.append(
                lp.reshape(B, num_anchors, num_landmarks * 2, H, W)
                  .permute(0, 3, 4, 1, 2).reshape(B, -1, num_landmarks * 2))
        all_cls = torch.cat(all_cls, dim=1)
        all_bbox = torch.cat(all_bbox, dim=1)
        all_lmk = torch.cat(all_lmk, dim=1)

        feat_maps = [(cs.shape[2], cs.shape[3]) for cs in cls_scores]
        anchors = anchor_gen(feat_maps).to(device)

        cls_t, bbox_t, lmk_t, fmask = match_anchors_to_targets_batch(
            anchors, targets_list, num_classes, num_landmarks)

        batch_loss = {"total_loss": 0, "cls_loss": 0, "bbox_loss": 0, "lmk_loss": 0}
        for b in range(B):
            loss_dict = criterion(
                {"cls_preds": all_cls[b], "bbox_preds": all_bbox[b],
                 "lmk_preds": all_lmk[b]},
                {"cls_targets": cls_t[b], "bbox_targets": bbox_t[b],
                 "lmk_targets": lmk_t[b], "face_mask": fmask[b]},
            )
            for k in batch_loss:
                batch_loss[k] += loss_dict[k]

        for k in val_losses:
            val_losses[k] += batch_loss[k].item() / B
        num_batches += 1

        pbar.set_postfix({
            "val_loss": f"{val_losses['total_loss'] / num_batches:.4f}",
        })

    return {k: v / max(num_batches, 1) for k, v in val_losses.items()}


def main():
    parser = argparse.ArgumentParser(description="DMS Multi-Task Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default="",
                       help="从 checkpoint 恢复训练 (e.g. weights/checkpoint.pt)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- 构建模型 ----
    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.2f}M")

    # ---- 数据集 ----
    train_set = DMSDataset(
        root=cfg["data"]["train_root"],
        img_size=cfg["train"]["img_size"],
        num_landmarks=cfg["model"]["num_landmarks"],
        transform=get_train_transforms(cfg["train"]["img_size"]),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---- 验证集 ----
    val_root = cfg["data"].get("val_root", "")
    has_val = os.path.isdir(os.path.join(val_root, "images")) if val_root else False
    val_loader = None
    if has_val:
        val_set = DMSDataset(
            root=val_root,
            img_size=cfg["train"]["img_size"],
            num_landmarks=cfg["model"]["num_landmarks"],
            transform=get_val_transforms(cfg["train"]["img_size"]),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )
        print(f"Validation set: {len(val_set)} samples")
    else:
        print("Validation set: not found, will monitor training loss only")

    # ---- 损失 / 优化器 / 调度器 ----
    uncertainty = cfg["train"]["loss_weights"].get("uncertainty", False)
    criterion = MultiTaskLoss(
        cls_weight=cfg["train"]["loss_weights"]["cls"],
        bbox_weight=cfg["train"]["loss_weights"]["bbox"],
        lmk_weight=cfg["train"]["loss_weights"]["landmark"],
        uncertainty=uncertainty,
    )
    if uncertainty:
        print("Loss weighting: uncertainty (learnable log_sigma)")

    # 优化器参数: 模型 + (可选) loss 中的可学习 log_sigma
    if uncertainty:
        opt_params = [
            {"params": model.parameters()},
            {"params": criterion.parameters(), "weight_decay": 0.0},
        ]
    else:
        opt_params = model.parameters()
    optimizer = torch.optim.SGD(
        opt_params,
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    warmup = LinearLR(optimizer, start_factor=0.01,
                      total_iters=cfg["train"]["warmup_epochs"])
    cosine = CosineAnnealingLR(optimizer,
                                T_max=cfg["train"]["epochs"] - cfg["train"]["warmup_epochs"])
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                              milestones=[cfg["train"]["warmup_epochs"]])

    # ---- Anchor 生成器 ----
    scales_cfg = cfg["anchors"]["scales"]
    steps = cfg["anchors"]["steps"]
    # 每层 2 个 scale → 拆分为 [[s0,s1],[s2,s3],[s4,s5]]
    scales_per_level = [scales_cfg[i:i + 2] for i in range(0, len(scales_cfg), 2)]
    anchor_gen = AnchorGenerator(scales=scales_per_level, steps=steps,
                                  img_size=cfg["train"]["img_size"])

    # ---- 知识蒸馏 (可选) ----
    distiller = None
    if cfg["distill"]["enable"]:
        teacher = DMSNet(
            num_classes=cfg["model"]["num_classes"],
            num_landmarks=cfg["model"]["num_landmarks"],
            num_anchors=cfg["model"]["num_anchors"],
            fpn_channels=cfg["model"]["fpn_channels"],
            backbone_type="large",
            pretrained=True,
        ).to(device)
        distiller = MultiTaskDistiller(
            student=model, teacher=teacher,
            temperature=cfg["distill"]["temperature"],
            alpha_feat=cfg["distill"]["alpha_feat"],
            alpha_logit=cfg["distill"]["alpha_logit"],
            alpha_resp=cfg["distill"].get("alpha_response", 0.5),
        ).to(device)
        print("Knowledge Distillation enabled (Teacher: MobileNetV3-Large)")

    # ---- TensorBoard ----
    writer = SummaryWriter("runs/dms_multitask")

    # ---- 混合精度 (仅 CUDA) ----
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    print(f"AMP (Automatic Mixed Precision): {'enabled' if use_amp else 'disabled'}")

    # ---- EMA ----
    ema_decay = cfg["train"].get("ema_decay", 0.0)
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None
    if ema is not None:
        print(f"EMA: enabled (decay={ema_decay})")

    # ---- 断点续训 ----
    os.makedirs("weights", exist_ok=True)
    best_loss = float("inf")
    start_epoch = 0

    if args.resume:
        if not os.path.isfile(args.resume):
            print(f"Warning: checkpoint not found at {args.resume}, "
                  "starting from scratch")
        else:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            if ema is not None and ckpt.get("ema"):
                ema.load_state_dict(ckpt["ema"])
            if uncertainty and ckpt.get("criterion"):
                criterion.load_state_dict(ckpt["criterion"])
            start_epoch = ckpt["epoch"]
            best_loss = ckpt.get("best_loss", float("inf"))
            print(f"Checkpoint loaded: epoch={start_epoch}, "
                  f"best_loss={best_loss:.4f}")

    # ---- 训练循环 ----
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{cfg['train']['epochs']}  "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")

        # -- 训练 --
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer,
                                       anchor_gen, device, cfg, scaler, ema)
        scheduler.step()

        # 日志: 训练
        for k, v in train_losses.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # 日志: 不确定性权重
        if uncertainty:
            w_cls = criterion.log_sigma_cls.detach().exp().item()
            w_bbox = criterion.log_sigma_bbox.detach().exp().item()
            w_lmk = criterion.log_sigma_lmk.detach().exp().item()
            writer.add_scalar("weights/cls", w_cls, epoch)
            writer.add_scalar("weights/bbox", w_bbox, epoch)
            writer.add_scalar("weights/lmk", w_lmk, epoch)

        # -- 验证 (使用 EMA shadow) --
        if val_loader is not None:
            if ema is not None:
                ema.apply_shadow()
            val_losses = validate(model, val_loader, criterion, anchor_gen,
                                  device, cfg)
            if ema is not None:
                ema.restore()
            for k, v in val_losses.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            print(f"  Train ─ loss={train_losses['total_loss']:.4f}  "
                  f"cls={train_losses['cls_loss']:.4f}  "
                  f"bbox={train_losses['bbox_loss']:.4f}  "
                  f"lmk={train_losses['lmk_loss']:.4f}"
                  + (f"  | w=[{criterion.log_sigma_cls.detach().exp().item():.3f}, "
                     f"{criterion.log_sigma_bbox.detach().exp().item():.3f}, "
                     f"{criterion.log_sigma_lmk.detach().exp().item():.3f}]"
                     if uncertainty else ""))
            print(f"  Val   ─ loss={val_losses['total_loss']:.4f}  "
                  f"cls={val_losses['cls_loss']:.4f}  "
                  f"bbox={val_losses['bbox_loss']:.4f}  "
                  f"lmk={val_losses['lmk_loss']:.4f}")

            monitor_loss = val_losses["total_loss"]
        else:
            monitor_loss = train_losses["total_loss"]

        # 构建完整 checkpoint (用于断点续训)
        checkpoint = {
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "ema": ema.state_dict() if ema else None,
            "criterion": criterion.state_dict() if uncertainty else None,
            "cfg": cfg,
        }

        # 保存最佳 (EMA shadow 权重)
        if monitor_loss < best_loss:
            best_loss = monitor_loss
            checkpoint["best_loss"] = best_loss
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), "weights/best.pt")
                ema.restore()
            else:
                torch.save(model.state_dict(), "weights/best.pt")
            torch.save(checkpoint, "weights/best_ckpt.pt")
            print(f"  ✓ Best model saved (loss={best_loss:.4f})")

        # 定期保存 + 每 epoch 更新可恢复 checkpoint
        torch.save(checkpoint, "weights/checkpoint.pt")
        if (epoch + 1) % 10 == 0:
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), f"weights/epoch_{epoch + 1}.pt")
                ema.restore()
            else:
                torch.save(model.state_dict(), f"weights/epoch_{epoch + 1}.pt")

    # 最终保存 (EMA shadow)
    if ema is not None:
        ema.apply_shadow()
    torch.save(model.state_dict(), "weights/last.pt")
    torch.save({
        "epoch": cfg["train"]["epochs"],
        "best_loss": best_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.state_dict() if ema else None,
        "criterion": criterion.state_dict() if uncertainty else None,
        "cfg": cfg,
    }, "weights/last_ckpt.pt")
    if ema is not None:
        ema.restore()
    writer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
