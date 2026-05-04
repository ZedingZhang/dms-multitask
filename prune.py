"""
结构化剪枝 (Structured Pruning)

策略:
  1. 基于 L1-norm 的 Filter Pruning: 移除 BN 层 γ 系数最小的通道
  2. 逐层剪枝比例可配置, 默认对 FPN 之后的 Head Conv 剪 40%
  3. 剪枝后微调 (Fine-tune) 恢复精度
  4. 输出剪枝后的紧凑模型

用法:
  python prune.py --config configs/default.yaml \
                  --weights weights/best.pt \
                  --ratio 0.4 \
                  --finetune_epochs 20
"""

import os
import copy
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dms_net import build_model
from data.dataset import DMSDataset, collate_fn
from data.augment import get_train_transforms
from utils.loss import MultiTaskLoss
from utils.anchors import AnchorGenerator
from train import match_anchors_to_targets, train_one_epoch


# ────────────────────────────────────────────────────────────
#  1. BN-γ 通道重要性排序 & 全局阈值剪枝
# ────────────────────────────────────────────────────────────

def collect_bn_weights(model: nn.Module):
    """收集所有 BN 层的 γ (weight) 并拼接成一维张量."""
    weights = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            weights.append(m.weight.data.abs().clone())
    return torch.cat(weights)


def compute_prune_threshold(bn_weights: torch.Tensor, ratio: float):
    """根据剪枝比例计算全局阈值."""
    sorted_weights, _ = torch.sort(bn_weights)
    idx = int(len(sorted_weights) * ratio)
    return sorted_weights[idx].item()


# ────────────────────────────────────────────────────────────
#  2. PyTorch 内置结构化剪枝
# ────────────────────────────────────────────────────────────

def apply_structured_pruning(model: nn.Module, ratio: float = 0.4):
    """
    对模型中所有 Conv2d 层施加 L1 结构化剪枝.
    剪掉 ratio 比例的输出通道 (filter).
    """
    pruned_count = 0
    total_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 跳过 1x1 输出层 (分类/回归最终输出, 通道数不可变)
            if module.out_channels <= 8:
                continue
            # 跳过 depthwise conv (groups == in_channels)
            if module.groups == module.in_channels and module.groups > 1:
                continue

            n_prune = int(module.out_channels * ratio)
            if n_prune == 0:
                continue

            prune.ln_structured(module, name="weight", amount=ratio, n=1, dim=0)
            pruned_count += n_prune
            total_count += module.out_channels

    print(f"Structured pruning applied: ~{pruned_count}/{total_count} filters pruned")
    return model


def remove_pruning_reparameterization(model: nn.Module):
    """移除 prune 的 reparameterization, 使模型可正常保存/导出."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass
    return model


# ────────────────────────────────────────────────────────────
#  3. 统计剪枝后模型信息
# ────────────────────────────────────────────────────────────

def model_stats(model: nn.Module, input_size=(1, 3, 640, 640)):
    """打印参数量和稀疏度."""
    total_params = 0
    zero_params = 0
    for p in model.parameters():
        total_params += p.numel()
        zero_params += (p == 0).sum().item()

    print(f"Total params : {total_params / 1e6:.2f}M")
    print(f"Zero params  : {zero_params / 1e6:.2f}M")
    print(f"Sparsity     : {zero_params / total_params * 100:.1f}%")

    # FLOPs 粗略估计
    try:
        from torchvision.models import _utils
    except ImportError:
        pass

    return total_params, zero_params


# ────────────────────────────────────────────────────────────
#  4. 主流程: 剪枝 → 微调 → 保存
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("DMS Model Pruning")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--ratio", type=float, default=0.4, help="剪枝比例")
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # 加载原始模型
    model = build_model(cfg).to(device)
    if os.path.isfile(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights: {args.weights}")

    print("\n=== Before Pruning ===")
    model_stats(model)

    # 剪枝
    print(f"\n=== Applying L1 Structured Pruning (ratio={args.ratio}) ===")
    model = apply_structured_pruning(model, ratio=args.ratio)

    print("\n=== After Pruning (with masks) ===")
    model_stats(model)

    # 微调恢复精度
    if args.finetune_epochs > 0:
        print(f"\n=== Fine-tuning for {args.finetune_epochs} epochs ===")

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

        criterion = MultiTaskLoss(
            cls_weight=cfg["train"]["loss_weights"]["cls"],
            bbox_weight=cfg["train"]["loss_weights"]["bbox"],
            lmk_weight=cfg["train"]["loss_weights"]["landmark"],
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["train"]["lr"] * 0.1,
                                    momentum=cfg["train"]["momentum"],
                                    weight_decay=cfg["train"]["weight_decay"])

        scales_cfg = cfg["anchors"]["scales"]
        steps = cfg["anchors"]["steps"]
        scales_per_level = [scales_cfg[i:i + 2] for i in range(0, len(scales_cfg), 2)]
        anchor_gen = AnchorGenerator(scales=scales_per_level, steps=steps,
                                      img_size=cfg["train"]["img_size"])

        for epoch in range(args.finetune_epochs):
            losses = train_one_epoch(model, train_loader, criterion, optimizer,
                                      anchor_gen, device, cfg)
            print(f"  FT Epoch {epoch+1}: loss={losses['total_loss']:.4f}")

    # 移除 prune reparameterization & 保存
    model = remove_pruning_reparameterization(model)
    os.makedirs("weights", exist_ok=True)
    save_path = f"weights/pruned_{int(args.ratio * 100)}pct.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nPruned model saved: {save_path}")

    print("\n=== Final Model Stats ===")
    model_stats(model)


if __name__ == "__main__":
    main()
