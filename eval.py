"""
性能评测脚本 —— mAP / Recall / Latency / Throughput

评测维度:
  1. 精度指标: mAP@0.5, mAP@0.5:0.95, Recall@0.5
  2. 速度指标: 端到端延迟 (ms/frame), 吞吐量 (FPS)
  3. 模型指标: 参数量 (M), FLOPs (G), 模型大小 (MB)

用法:
  python eval.py --config configs/default.yaml \
                 --weights weights/best.pt \
                 --data_root data/val \
                 --device cuda

  # 对比剪枝 / 量化前后
  python eval.py --weights weights/pruned_40pct.pt --tag pruned
"""

import os
import time
import yaml
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms as torch_nms
from tqdm import tqdm

from models.dms_net import build_model
from data.dataset import DMSDataset, collate_fn
from utils.anchors import AnchorGenerator


# ════════════════════════════════════════════════════════════
#  1. IoU 计算工具
# ════════════════════════════════════════════════════════════

def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算 IoU 矩阵.
    boxes1: (N, 4), boxes2: (M, 4)  格式 [x1, y1, x2, y2]
    返回: (N, M) IoU 矩阵
    """
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-6)


# ════════════════════════════════════════════════════════════
#  2. mAP / Recall 计算
# ════════════════════════════════════════════════════════════

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """计算单类 AP (11-point interpolation)."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max()
    return ap / 11.0


def evaluate_detections(all_preds: list, all_gts: list,
                         num_classes: int = 3,
                         iou_thresholds: list = None):
    """
    计算每个类别在各 IoU 阈值下的 AP 和 Recall.

    Args:
        all_preds: 每张图的预测 list[list[dict]]
                   dict: {class_id, score, bbox: [x1,y1,x2,y2]}
        all_gts:   每张图的 GT   list[list[dict]]
                   dict: {class_id, bbox: [x1,y1,x2,y2]}
        iou_thresholds: 默认 [0.5, 0.55, ..., 0.95]

    Returns:
        results: dict
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    CLASS_NAMES = ["face", "phone", "cigarette"]
    results = {}

    for iou_t in iou_thresholds:
        aps = []
        recalls = []

        for cls_id in range(num_classes):
            # 收集所有该类的预测和 GT
            all_scores = []
            all_tp_fp = []
            total_gt = 0

            for preds, gts in zip(all_preds, all_gts):
                # 该图中该类的 GT
                gt_boxes = np.array([g["bbox"] for g in gts
                                      if g["class_id"] == cls_id])
                total_gt += len(gt_boxes)
                gt_matched = np.zeros(len(gt_boxes), dtype=bool)

                # 该图中该类的预测, 按分数降序
                cls_preds = sorted(
                    [p for p in preds if p["class_id"] == cls_id],
                    key=lambda x: -x["score"],
                )

                for p in cls_preds:
                    all_scores.append(p["score"])
                    if len(gt_boxes) == 0:
                        all_tp_fp.append(0)  # FP
                        continue

                    pred_box = np.array(p["bbox"]).reshape(1, 4)
                    ious = box_iou(pred_box, gt_boxes)[0]
                    best_idx = ious.argmax()

                    if ious[best_idx] >= iou_t and not gt_matched[best_idx]:
                        all_tp_fp.append(1)  # TP
                        gt_matched[best_idx] = True
                    else:
                        all_tp_fp.append(0)  # FP

            if total_gt == 0:
                aps.append(0.0)
                recalls.append(0.0)
                continue

            # 按分数排序
            sorted_idx = np.argsort(-np.array(all_scores))
            tp_fp = np.array(all_tp_fp)[sorted_idx]

            tp_cum = np.cumsum(tp_fp)
            fp_cum = np.cumsum(1 - tp_fp)
            rec = tp_cum / total_gt
            prec = tp_cum / (tp_cum + fp_cum)

            ap = compute_ap(rec, prec)
            aps.append(ap)
            recalls.append(rec[-1] if len(rec) > 0 else 0.0)

        iou_key = f"IoU={iou_t:.2f}"
        results[iou_key] = {
            "mAP": float(np.mean(aps)),
            "per_class_AP": {CLASS_NAMES[i]: float(aps[i])
                              for i in range(num_classes)},
            "per_class_Recall": {CLASS_NAMES[i]: float(recalls[i])
                                  for i in range(num_classes)},
        }

    # 汇总指标
    map50 = results.get("IoU=0.50", {}).get("mAP", 0)
    map_all = np.mean([results[k]["mAP"] for k in results])
    recall50 = results.get("IoU=0.50", {}).get("per_class_Recall", {})

    results["summary"] = {
        "mAP@0.5": float(map50),
        "mAP@0.5:0.95": float(map_all),
        "Recall@0.5": {k: float(v) for k, v in recall50.items()},
    }

    return results


# ════════════════════════════════════════════════════════════
#  3. 关键点精度评估
# ════════════════════════════════════════════════════════════

def evaluate_landmarks(all_preds: list, all_gts: list,
                       iou_threshold: float = 0.5):
    """评估人脸关键点精度.

    通过检测框 IoU 匹配预测人脸与 GT 人脸, 然后计算:
      - NME (Normalized Mean Error): 用双眼间距 (IOD) 归一化
      - FR@α (Failure Rate): NME > α 的人脸比例
      - 分区误差: 左眼 / 右眼 / 嘴巴

    关键点布局 (14 点):
      0-5:   左眼 6 点
      6-11:  右眼 6 点
      12-13: 嘴角 2 点

    Returns:
        dict
    """
    all_nme = []          # 所有人脸的 NME
    left_eye_errs = []
    right_eye_errs = []
    mouth_errs = []
    total_gt = 0           # GT 人脸总数
    fail_08 = 0             # NME > 0.08
    fail_10 = 0             # NME > 0.10

    for preds, gts in zip(all_preds, all_gts):
        face_preds = [p for p in preds if p["class_id"] == 0]
        face_gts = [g for g in gts if g["class_id"] == 0]
        total_gt += len(face_gts)

        if len(face_preds) == 0:
            fail_08 += len(face_gts)
            fail_10 += len(face_gts)
            continue
        if len(face_gts) == 0:
            continue

        pred_boxes = np.array([p["bbox"] for p in face_preds])
        gt_boxes = np.array([g["bbox"] for g in face_gts])
        ious = box_iou(pred_boxes, gt_boxes)

        # 按置信度降序贪心匹配
        matched_gt = set()
        pred_order = np.argsort(-np.array([p["score"] for p in face_preds]))

        for pred_i in pred_order:
            gt_i = ious[pred_i].argmax()
            if ious[pred_i, gt_i] >= iou_threshold and gt_i not in matched_gt:
                matched_gt.add(gt_i)

                pred_lmks = np.array(face_preds[pred_i].get("landmarks"))
                gt_lmks = np.array(face_gts[gt_i].get("landmarks"))
                if pred_lmks is None or gt_lmks is None:
                    continue
                if len(pred_lmks) == 0 or len(gt_lmks) == 0:
                    continue
                # 跳过无效 GT (非人脸目标标记为 -1)
                if (gt_lmks < 0).any():
                    continue

                # ---- 归一化因子: 双眼间距 (IOD) ----
                left_center = gt_lmks[0:6].reshape(-1, 2).mean(axis=0)
                right_center = gt_lmks[6:12].reshape(-1, 2).mean(axis=0)
                iod = np.linalg.norm(left_center - right_center)
                if iod < 1e-6:
                    continue

                pred_pts = pred_lmks.reshape(-1, 2)
                gt_pts = gt_lmks.reshape(-1, 2)
                per_pt_nme = np.linalg.norm(pred_pts - gt_pts, axis=1) / iod
                nme = float(per_pt_nme.mean())

                all_nme.append(nme)
                if nme > 0.08:
                    fail_08 += 1
                if nme > 0.10:
                    fail_10 += 1

                left_eye_errs.append(per_pt_nme[0:6].mean())
                right_eye_errs.append(per_pt_nme[6:12].mean())
                mouth_errs.append(per_pt_nme[12:14].mean())

        # 未匹配的 GT 人脸计入失败
        unmatched = len(face_gts) - len(matched_gt)
        fail_08 += unmatched
        fail_10 += unmatched

    matched = len(all_nme)
    if matched == 0:
        return {
            "nme_mean": 0.0, "nme_median": 0.0, "nme_std": 0.0,
            "fr_08": 1.0, "fr_10": 1.0,
            "left_eye_nme": 0.0, "right_eye_nme": 0.0, "mouth_nme": 0.0,
            "total_faces": total_gt, "matched_faces": 0,
        }

    arr = np.array(all_nme)
    return {
        "nme_mean":   float(arr.mean()),
        "nme_median": float(np.median(arr)),
        "nme_std":    float(arr.std()),
        "fr_08":      float(fail_08 / max(total_gt, 1)),
        "fr_10":      float(fail_10 / max(total_gt, 1)),
        "left_eye_nme":  float(np.mean(left_eye_errs)) if left_eye_errs else 0.0,
        "right_eye_nme": float(np.mean(right_eye_errs)) if right_eye_errs else 0.0,
        "mouth_nme":     float(np.mean(mouth_errs)) if mouth_errs else 0.0,
        "total_faces":   total_gt,
        "matched_faces": matched,
    }


# ════════════════════════════════════════════════════════════
#  5. Latency / Throughput 测速
# ════════════════════════════════════════════════════════════

def benchmark_latency(model: torch.nn.Module, device: torch.device,
                       input_size=(1, 3, 640, 640),
                       warmup: int = 50, repeats: int = 200):
    """
    测量端到端推理延迟 (含预处理张量构造).

    Returns:
        avg_ms:    平均延迟 (ms)
        fps:       吞吐量 (frames/sec)
        p50_ms:    P50 延迟
        p99_ms:    P99 延迟
    """
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 计时
    latencies = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)
    avg_ms = float(latencies.mean())
    fps = 1000.0 / avg_ms
    p50 = float(np.percentile(latencies, 50))
    p99 = float(np.percentile(latencies, 99))

    return {
        "avg_ms": avg_ms,
        "fps": fps,
        "p50_ms": p50,
        "p99_ms": p99,
        "std_ms": float(latencies.std()),
    }


# ════════════════════════════════════════════════════════════
#  6. 模型指标
# ════════════════════════════════════════════════════════════

def model_info(model: torch.nn.Module, input_size=(1, 3, 640, 640)):
    """统计模型参数量和文件大小."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 模型大小
    tmp_path = "/tmp/dms_eval_tmp.pt"
    torch.save(model.state_dict(), tmp_path)
    model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)

    return {
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable / 1e6,
        "model_size_MB": model_size_mb,
    }


# ════════════════════════════════════════════════════════════
#  7. 推理 + 收集预测结果
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, loader, anchor_gen, device, cfg,
                   conf_threshold=0.3):
    """对整个验证集推理, 收集预测和 GT."""
    model.eval()
    num_classes = cfg["model"]["num_classes"]
    num_anchors = cfg["model"]["num_anchors"]
    num_landmarks = cfg["model"]["num_landmarks"]

    all_preds = []
    all_gts = []

    for images, targets_list in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        cls_scores, bbox_preds, lmk_preds = model(images)

        B = images.shape[0]
        feat_maps = [(c.shape[2], c.shape[3]) for c in cls_scores]
        anchors = anchor_gen(feat_maps).to(device)

        # 拉平
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

        flat_cls = torch.cat(all_cls, dim=1).sigmoid()
        flat_bbox = torch.cat(all_bbox, dim=1)
        flat_lmk = torch.cat(all_lmk, dim=1)

        for b in range(B):
            # 解码预测
            from utils.anchors import AnchorGenerator as AG
            boxes = AG.decode_boxes(anchors, flat_bbox[b])
            landmarks = AG.decode_landmarks(anchors, flat_lmk[b],
                                            num_landmarks)
            scores = flat_cls[b]

            preds = []
            for c in range(1, num_classes):  # skip class 0 (background)
                mask = scores[:, c] > conf_threshold
                if not mask.any():
                    continue
                c_boxes = boxes[mask]
                c_scores = scores[mask, c]
                c_lmks = landmarks[mask]

                keep = torch_nms(c_boxes, c_scores, iou_threshold=0.45)
                for k in keep:
                    pred = {
                        "class_id": c - 1,  # map back to 0-indexed object class
                        "score": float(c_scores[k]),
                        "bbox": c_boxes[k].tolist(),
                    }
                    if c == 0:
                        pred["landmarks"] = c_lmks[k].reshape(-1, 2).tolist()
                    preds.append(pred)

            # GT
            gt = targets_list[b]
            gts = []
            for j in range(len(gt["cls_ids"])):
                cls_id = int(gt["cls_ids"][j])
                gt_dict = {
                    "class_id": cls_id,
                    "bbox": gt["bboxes"][j].tolist(),
                }
                if cls_id == 0:
                    gt_dict["landmarks"] = gt["landmarks"][j].reshape(-1, 2).tolist()
                gts.append(gt_dict)

            all_preds.append(preds)
            all_gts.append(gts)

    return all_preds, all_gts


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser("DMS Performance Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--tag", default="baseline", help="评测标签 (用于对比)")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    data_root = args.data_root or cfg["data"]["val_root"]

    # ---- 构建模型 ----
    model = build_model(cfg).to(device)
    if os.path.isfile(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    print(f"\n{'='*65}")
    print(f"  DMS Performance Evaluation — [{args.tag}]")
    print(f"{'='*65}")

    # ---- 模型指标 ----
    print("\n📊 Model Info:")
    info = model_info(model)
    print(f"  Parameters:  {info['total_params_M']:.2f}M")
    print(f"  Model size:  {info['model_size_MB']:.1f} MB")

    # ---- 延迟 / 吞吐 ----
    print(f"\n⏱  Latency Benchmark (device={device}):")
    speed = benchmark_latency(model, device,
                               input_size=(1, 3, cfg["train"]["img_size"],
                                           cfg["train"]["img_size"]))
    print(f"  Avg latency: {speed['avg_ms']:.2f} ms")
    print(f"  P50 latency: {speed['p50_ms']:.2f} ms")
    print(f"  P99 latency: {speed['p99_ms']:.2f} ms")
    print(f"  Throughput:  {speed['fps']:.1f} FPS")

    # ---- mAP / Recall ----
    print(f"\n🎯 Accuracy Evaluation:")
    val_set = DMSDataset(
        root=data_root,
        img_size=cfg["train"]["img_size"],
        num_landmarks=cfg["model"]["num_landmarks"],
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=0,
        collate_fn=collate_fn,
    )

    scales_cfg = cfg["anchors"]["scales"]
    steps = cfg["anchors"]["steps"]
    scales_per_level = [scales_cfg[i:i + 2] for i in range(0, len(scales_cfg), 2)]
    anchor_gen = AnchorGenerator(scales=scales_per_level, steps=steps,
                                  img_size=cfg["train"]["img_size"])

    all_preds, all_gts = run_inference(model, val_loader, anchor_gen,
                                        device, cfg, args.conf)
    eval_results = evaluate_detections(all_preds, all_gts,
                                        num_classes=cfg["model"]["num_classes"] - 1)

    summary = eval_results["summary"]
    print(f"  mAP@0.5:      {summary['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {summary['mAP@0.5:0.95']:.4f}")
    print(f"  Recall@0.5:")
    for cls_name, val in summary["Recall@0.5"].items():
        print(f"    {cls_name:12s}: {val:.4f}")

    # ---- 关键点评估 ----
    print(f"\n👁  Landmark Evaluation (Normalized by IOD):")
    lmk_results = evaluate_landmarks(all_preds, all_gts)
    print(f"  NME mean/median/std: {lmk_results['nme_mean']:.4f} / "
          f"{lmk_results['nme_median']:.4f} / {lmk_results['nme_std']:.4f}")
    print(f"  FR@0.08:            {lmk_results['fr_08']:.4f}  "
          f"(NME > 8% IOD)")
    print(f"  FR@0.10:            {lmk_results['fr_10']:.4f}  "
          f"(NME > 10% IOD)")
    print(f"  Per-region NME ── left_eye: {lmk_results['left_eye_nme']:.4f}  "
          f"right_eye: {lmk_results['right_eye_nme']:.4f}  "
          f"mouth: {lmk_results['mouth_nme']:.4f}")
    print(f"  Matched faces:      {lmk_results['matched_faces']} / "
          f"{lmk_results['total_faces']}")

    # ---- 汇总输出 ----
    report = {
        "tag": args.tag,
        "weights": args.weights,
        "device": str(device),
        "model_info": info,
        "speed": speed,
        "accuracy": eval_results,
        "landmark": lmk_results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Full report saved: {args.output}")

    # 表格式总览
    print(f"\n{'='*65}")
    print(f"  {'Metric':<25s} {'Value':>15s}")
    print(f"  {'-'*40}")
    print(f"  {'Params (M)':<25s} {info['total_params_M']:>15.2f}")
    print(f"  {'Model Size (MB)':<25s} {info['model_size_MB']:>15.1f}")
    print(f"  {'Latency (ms)':<25s} {speed['avg_ms']:>15.2f}")
    print(f"  {'Throughput (FPS)':<25s} {speed['fps']:>15.1f}")
    print(f"  {'mAP@0.5':<25s} {summary['mAP@0.5']:>15.4f}")
    print(f"  {'mAP@0.5:0.95':<25s} {summary['mAP@0.5:0.95']:>15.4f}")
    print(f"  {'Landmark NME':<25s} {lmk_results['nme_mean']:>15.4f}")
    print(f"  {'Landmark FR@0.08':<25s} {lmk_results['fr_08']:>15.4f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
