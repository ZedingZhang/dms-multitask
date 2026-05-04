"""
知识蒸馏训练脚本 —— ResNet50 Teacher → MobileNetV3-Small Student

蒸馏策略:
  1. Feature Distillation  : FPN 各层特征图 MSE (通过 1×1 Conv 对齐通道)
  2. Logit Distillation     : 分类 Head 的 KL 散度 (温度软化)
  3. Response Distillation  : bbox / landmark 回归 Head 的 Smooth-L1

训练流程:
  Step 1 — 先用标注数据训练 Teacher 至收敛 (或使用预训练权重)
  Step 2 — 冻结 Teacher, 联合 task_loss + distill_loss 训练 Student
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.dms_net import build_model
from distill.teacher import ResNet50Teacher
from data.dataset import DMSDataset, collate_fn
from data.augment import get_train_transforms, get_val_transforms
from utils.loss import MultiTaskLoss
from utils.anchors import AnchorGenerator
from train import (match_anchors_to_targets, match_anchors_to_targets_batch,
                    validate, ModelEMA)


# ────────────────────────────────────────────────────────────
#  通道对齐 + 蒸馏损失
# ────────────────────────────────────────────────────────────

class ChannelAligner(nn.Module):
    """当 Student / Teacher FPN 通道不同时, 1×1 Conv 对齐."""

    def __init__(self, s_ch: int, t_ch: int, num_levels: int = 3):
        super().__init__()
        self.aligns = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, 1, bias=False) for _ in range(num_levels)
        ])

    def forward(self, s_feats):
        return [a(f) for a, f in zip(self.aligns, s_feats)]


def feature_distill_loss(s_feats, t_feats, aligner=None):
    """FPN 逐层特征 MSE."""
    if aligner is not None:
        s_feats = aligner(s_feats)
    loss = 0.0
    for sf, tf in zip(s_feats, t_feats):
        if sf.shape[2:] != tf.shape[2:]:
            sf = F.interpolate(sf, tf.shape[2:], mode="bilinear",
                               align_corners=False)
        loss += F.mse_loss(sf, tf)
    return loss / len(s_feats)


def logit_distill_loss(s_cls_list, t_cls_list, temperature=4.0):
    """分类 logit KL 散度蒸馏."""
    T = temperature
    loss = 0.0
    for s_cls, t_cls in zip(s_cls_list, t_cls_list):
        B = s_cls.shape[0]
        s_flat = s_cls.reshape(B, -1)
        t_flat = t_cls.reshape(B, -1)
        loss += F.kl_div(
            F.log_softmax(s_flat / T, dim=-1),
            F.softmax(t_flat / T, dim=-1),
            reduction="batchmean",
        ) * (T ** 2)
    return loss / len(s_cls_list)


def response_distill_loss(s_reg_list, t_reg_list):
    """回归 Head (bbox / landmark) 的 response 蒸馏."""
    loss = 0.0
    for s_reg, t_reg in zip(s_reg_list, t_reg_list):
        loss += F.smooth_l1_loss(s_reg, t_reg)
    return loss / len(s_reg_list)


# ────────────────────────────────────────────────────────────
#  训练主循环
# ────────────────────────────────────────────────────────────

def train_one_epoch_distill(student, teacher, loader, criterion, optimizer,
                             anchor_gen, aligner, device, cfg, scaler=None,
                             ema=None):
    student.train()
    teacher.eval()
    use_amp = scaler is not None

    num_classes = cfg["model"]["num_classes"]
    num_landmarks = cfg["model"]["num_landmarks"]
    num_anchors = cfg["model"]["num_anchors"]
    T = cfg["distill"]["temperature"]
    alpha_feat = cfg["distill"]["alpha_feat"]
    alpha_logit = cfg["distill"]["alpha_logit"]
    alpha_resp = cfg["distill"].get("alpha_response", 0.5)

    stats = {"total": 0, "task": 0, "feat": 0, "logit": 0, "resp": 0}
    n = 0

    for images, targets_list in tqdm(loader, desc="Distill-Train"):
        images = images.to(device)

        with autocast(enabled=use_amp):
            # Student forward
            s_cls, s_bbox, s_lmk = student(images)
            s_feats = student.get_fpn_features(images)

            # Teacher forward (no grad)
            with torch.no_grad():
                t_cls, t_bbox, t_lmk = teacher(images)
                t_feats = teacher.get_fpn_features(images)

            # ---- Task Loss (与标注计算) ----
            B = images.shape[0]
            all_cls, all_bbox, all_lmk = [], [], []
            for cs, bp, lp in zip(s_cls, s_bbox, s_lmk):
                H, W = cs.shape[2:]
                all_cls.append(cs.reshape(B, num_anchors, num_classes, H, W)
                               .permute(0, 3, 4, 1, 2).reshape(B, -1, num_classes))
                all_bbox.append(bp.reshape(B, num_anchors, 4, H, W)
                                .permute(0, 3, 4, 1, 2).reshape(B, -1, 4))
                all_lmk.append(lp.reshape(B, num_anchors, num_landmarks * 2, H, W)
                               .permute(0, 3, 4, 1, 2).reshape(B, -1, num_landmarks * 2))
            all_cls = torch.cat(all_cls, dim=1)
            all_bbox = torch.cat(all_bbox, dim=1)
            all_lmk = torch.cat(all_lmk, dim=1)

            feat_maps = [(cs.shape[2], cs.shape[3]) for cs in s_cls]
            anchors = anchor_gen(feat_maps).to(device)

            cls_t, bbox_t, lmk_t, fmask = match_anchors_to_targets_batch(
                anchors, targets_list, num_classes, num_landmarks)

            task_loss_val = torch.tensor(0.0, device=device)
            for b in range(B):
                loss_dict = criterion(
                    {"cls_preds": all_cls[b], "bbox_preds": all_bbox[b],
                     "lmk_preds": all_lmk[b]},
                    {"cls_targets": cls_t[b], "bbox_targets": bbox_t[b],
                     "lmk_targets": lmk_t[b], "face_mask": fmask[b]},
                )
                task_loss_val += loss_dict["total_loss"]
            task_loss_val /= B

            # ---- Distillation Losses ----
            feat_loss = feature_distill_loss(s_feats, t_feats, aligner)
            logit_loss = logit_distill_loss(s_cls, t_cls, T)
            resp_loss = response_distill_loss(s_bbox, t_bbox) + \
                        response_distill_loss(s_lmk, t_lmk)

            total = task_loss_val + \
                    alpha_feat * feat_loss + \
                    alpha_logit * logit_loss + \
                    alpha_resp * resp_loss

        # 反向传播
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        if ema is not None:
            ema.update()

        stats["total"] += total.item()
        stats["task"] += task_loss_val.item()
        stats["feat"] += feat_loss.item()
        stats["logit"] += logit_loss.item()
        stats["resp"] += resp_loss.item()
        n += 1

    return {k: v / max(n, 1) for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser("DMS Knowledge Distillation Training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--teacher_weights", default="", help="预训练 Teacher 权重路径")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="",
                       help="从 checkpoint 恢复训练 (e.g. weights/distill_checkpoint.pt)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # 强制开启蒸馏
    cfg["distill"]["enable"] = True

    device = torch.device(args.device)

    # ---- Student ----
    student = build_model(cfg).to(device)
    s_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"Student params: {s_params:.2f}M")

    # ---- Teacher (ResNet50, 大容量) ----
    teacher = ResNet50Teacher(
        num_classes=cfg["model"]["num_classes"],
        num_landmarks=cfg["model"]["num_landmarks"],
        num_anchors=cfg["model"]["num_anchors"],
        fpn_channels=128,
        pretrained=True,
    ).to(device)
    if args.teacher_weights and os.path.isfile(args.teacher_weights):
        teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
        print(f"Teacher weights loaded: {args.teacher_weights}")
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    t_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"Teacher params: {t_params:.2f}M  (frozen)")

    # ---- 通道对齐 (Student 64ch → Teacher 128ch) ----
    aligner = ChannelAligner(
        s_ch=cfg["model"]["fpn_channels"],  # 64
        t_ch=teacher.fpn_channels,           # 128
    ).to(device)

    # ---- Data ----
    train_set = DMSDataset(
        root=cfg["data"]["train_root"],
        img_size=cfg["train"]["img_size"],
        num_landmarks=cfg["model"]["num_landmarks"],
        transform=get_train_transforms(cfg["train"]["img_size"]),
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn, pin_memory=True,
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

    # ---- Optimizer ----
    uncertainty = cfg["train"]["loss_weights"].get("uncertainty", False)
    criterion = MultiTaskLoss(
        cls_weight=cfg["train"]["loss_weights"]["cls"],
        bbox_weight=cfg["train"]["loss_weights"]["bbox"],
        lmk_weight=cfg["train"]["loss_weights"]["landmark"],
        uncertainty=uncertainty,
    )
    if uncertainty:
        print("Loss weighting: uncertainty (learnable log_sigma)")

    # 把对齐层 + (可选) uncertainty 参数加入优化
    if uncertainty:
        opt_params = [
            {"params": student.parameters()},
            {"params": aligner.parameters()},
            {"params": criterion.parameters(), "weight_decay": 0.0},
        ]
    else:
        opt_params = list(student.parameters()) + list(aligner.parameters())
    optimizer = torch.optim.SGD(opt_params, lr=cfg["train"]["lr"],
                                momentum=cfg["train"]["momentum"],
                                weight_decay=cfg["train"]["weight_decay"])
    warmup = LinearLR(optimizer, start_factor=0.01,
                      total_iters=cfg["train"]["warmup_epochs"])
    cosine = CosineAnnealingLR(optimizer,
                                T_max=cfg["train"]["epochs"] - cfg["train"]["warmup_epochs"])
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                              milestones=[cfg["train"]["warmup_epochs"]])

    # ---- Anchor ----
    scales_cfg = cfg["anchors"]["scales"]
    steps = cfg["anchors"]["steps"]
    scales_per_level = [scales_cfg[i:i + 2] for i in range(0, len(scales_cfg), 2)]
    anchor_gen = AnchorGenerator(scales=scales_per_level, steps=steps,
                                  img_size=cfg["train"]["img_size"])

    # ---- 混合精度 (仅 CUDA) ----
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    print(f"AMP (Automatic Mixed Precision): {'enabled' if use_amp else 'disabled'}")

    # ---- EMA ----
    ema_decay = cfg["train"].get("ema_decay", 0.0)
    ema = ModelEMA(student, decay=ema_decay) if ema_decay > 0 else None
    if ema is not None:
        print(f"EMA: enabled (decay={ema_decay})")

    # ---- 断点续训 ----
    os.makedirs("weights", exist_ok=True)
    best = float("inf")
    start_epoch = 0

    if args.resume:
        if not os.path.isfile(args.resume):
            print(f"Warning: checkpoint not found at {args.resume}, "
                  "starting from scratch")
        else:
            ckpt = torch.load(args.resume, map_location=device)
            student.load_state_dict(ckpt["student"])
            teacher.load_state_dict(ckpt["teacher"])
            aligner.load_state_dict(ckpt["aligner"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            if ema is not None and ckpt.get("ema"):
                ema.load_state_dict(ckpt["ema"])
            if uncertainty and ckpt.get("criterion"):
                criterion.load_state_dict(ckpt["criterion"])
            start_epoch = ckpt["epoch"]
            best = ckpt.get("best_loss", float("inf"))
            print(f"Checkpoint loaded: epoch={start_epoch}, "
                  f"best_loss={best:.4f}")

    writer = SummaryWriter("runs/dms_distill")
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        print(f"\n{'='*60}")
        print(f"[Distill] Epoch {epoch+1}/{cfg['train']['epochs']}  "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")

        # -- 训练 --
        losses = train_one_epoch_distill(
            student, teacher, train_loader, criterion, optimizer,
            anchor_gen, aligner, device, cfg, scaler, ema,
        )
        scheduler.step()

        for k, v in losses.items():
            writer.add_scalar(f"distill/train_{k}", v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if uncertainty:
            w_cls = criterion.log_sigma_cls.detach().exp().item()
            w_bbox = criterion.log_sigma_bbox.detach().exp().item()
            w_lmk = criterion.log_sigma_lmk.detach().exp().item()
            writer.add_scalar("distill/weights_cls", w_cls, epoch)
            writer.add_scalar("distill/weights_bbox", w_bbox, epoch)
            writer.add_scalar("distill/weights_lmk", w_lmk, epoch)

        print(f"  Train ─ total={losses['total']:.4f}  task={losses['task']:.4f}  "
              f"feat={losses['feat']:.4f}  logit={losses['logit']:.4f}  "
              f"resp={losses['resp']:.4f}"
              + (f"  | w=[{criterion.log_sigma_cls.detach().exp().item():.3f}, "
                 f"{criterion.log_sigma_bbox.detach().exp().item():.3f}, "
                 f"{criterion.log_sigma_lmk.detach().exp().item():.3f}]"
                 if uncertainty else ""))

        # -- 验证 (使用 EMA shadow, 仅任务损失) --
        if val_loader is not None:
            if ema is not None:
                ema.apply_shadow()
            val_losses = validate(student, val_loader, criterion, anchor_gen,
                                  device, cfg)
            if ema is not None:
                ema.restore()
            for k, v in val_losses.items():
                writer.add_scalar(f"distill/val_{k}", v, epoch)

            print(f"  Val   ─ loss={val_losses['total_loss']:.4f}  "
                  f"cls={val_losses['cls_loss']:.4f}  "
                  f"bbox={val_losses['bbox_loss']:.4f}  "
                  f"lmk={val_losses['lmk_loss']:.4f}")

            monitor_loss = val_losses["total_loss"]
        else:
            monitor_loss = losses["total"]

        # 构建完整 checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "best_loss": best,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "aligner": aligner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "ema": ema.state_dict() if ema else None,
            "criterion": criterion.state_dict() if uncertainty else None,
            "cfg": cfg,
        }

        if monitor_loss < best:
            best = monitor_loss
            checkpoint["best_loss"] = best
            if ema is not None:
                ema.apply_shadow()
                torch.save(student.state_dict(), "weights/distill_best.pt")
                ema.restore()
            else:
                torch.save(student.state_dict(), "weights/distill_best.pt")
            torch.save(checkpoint, "weights/distill_best_ckpt.pt")
            print(f"  ✓ Best model saved (loss={best:.4f})")

        torch.save(checkpoint, "weights/distill_checkpoint.pt")

    # 最终保存 (EMA shadow)
    if ema is not None:
        ema.apply_shadow()
    torch.save(student.state_dict(), "weights/distill_last.pt")
    torch.save({
        "epoch": cfg["train"]["epochs"],
        "best_loss": best,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "aligner": aligner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.state_dict() if ema else None,
        "criterion": criterion.state_dict() if uncertainty else None,
        "cfg": cfg,
    }, "weights/distill_last_ckpt.pt")
    if ema is not None:
        ema.restore()
    writer.close()
    print("Distillation training complete.")


if __name__ == "__main__":
    main()
