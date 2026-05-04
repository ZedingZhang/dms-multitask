"""
推理 Demo —— 读取摄像头/视频, 实时检测 + 关键点 + PERCLOS 疲劳预警
"""

import cv2
import yaml
import torch
import argparse
import numpy as np
from torchvision.ops import nms as torch_nms

from models.dms_net import build_model
from utils.anchors import AnchorGenerator
from utils.metrics import compute_ear, compute_mar, PERCLOSTracker, YawnDetector


@torch.no_grad()
def infer_frame(model, frame, anchor_gen, device, cfg,
                conf_threshold=0.5, nms_threshold=0.4):
    """单帧推理."""
    img_size = cfg["train"]["img_size"]
    num_classes = cfg["model"]["num_classes"]
    num_anchors = cfg["model"]["num_anchors"]
    num_landmarks = cfg["model"]["num_landmarks"]

    h0, w0 = frame.shape[:2]
    img = cv2.resize(frame, (img_size, img_size))
    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_t = img_t.to(device)

    cls_scores, bbox_preds, lmk_preds = model(img_t)

    # 拉平
    all_cls, all_bbox, all_lmk = [], [], []
    for cs, bp, lp in zip(cls_scores, bbox_preds, lmk_preds):
        H, W = cs.shape[2:]
        all_cls.append(
            cs.reshape(1, num_anchors, num_classes, H, W)
              .permute(0, 3, 4, 1, 2).reshape(-1, num_classes))
        all_bbox.append(
            bp.reshape(1, num_anchors, 4, H, W)
              .permute(0, 3, 4, 1, 2).reshape(-1, 4))
        all_lmk.append(
            lp.reshape(1, num_anchors, num_landmarks * 2, H, W)
              .permute(0, 3, 4, 1, 2).reshape(-1, num_landmarks * 2))

    all_cls = torch.cat(all_cls, dim=0).sigmoid()
    all_bbox = torch.cat(all_bbox, dim=0)
    all_lmk = torch.cat(all_lmk, dim=0)

    feat_maps = [(cs.shape[2], cs.shape[3]) for cs in cls_scores]
    anchors = anchor_gen(feat_maps).to(device)

    # 解码边框
    boxes = AnchorGenerator.decode_boxes(anchors, all_bbox)
    landmarks = AnchorGenerator.decode_landmarks(anchors, all_lmk, num_landmarks)

    # 转 numpy
    boxes = boxes.cpu().numpy()
    landmarks = landmarks.cpu().numpy()
    scores = all_cls.cpu().numpy()

    results = []
    CLASS_NAMES = ["face", "phone", "cigarette"]

    for c in range(1, num_classes):  # skip class 0 (background)
        cls_scores_c = scores[:, c]
        mask = cls_scores_c > conf_threshold
        if not mask.any():
            continue
        c_boxes = boxes[mask] * np.array([w0, h0, w0, h0])
        c_scores = cls_scores_c[mask]
        c_lmks = landmarks[mask] * np.array([w0, h0])

        keep = torch_nms(
            torch.from_numpy(c_boxes).float(),
            torch.from_numpy(c_scores).float(),
            nms_threshold,
        ).numpy()
        for k in keep:
            results.append({
                "class": CLASS_NAMES[c - 1],
                "class_id": c - 1,
                "score": float(c_scores[k]),
                "bbox": c_boxes[k].tolist(),
                "landmarks": c_lmks[k].tolist() if c == 0 else None,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--source", default="0", help="摄像头编号或视频路径")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    scales_cfg = cfg["anchors"]["scales"]
    steps = cfg["anchors"]["steps"]
    scales_per_level = [scales_cfg[i:i + 2] for i in range(0, len(scales_cfg), 2)]
    anchor_gen = AnchorGenerator(scales=scales_per_level, steps=steps,
                                  img_size=cfg["train"]["img_size"])

    # 疲劳检测追踪器
    perclos_tracker = PERCLOSTracker(window_sec=60, fps=30)
    yawn_detector = YawnDetector()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = infer_frame(model, frame, anchor_gen, device, cfg, args.conf)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for det in results:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label = f"{det['class']} {det['score']:.2f}"
            color = (0, 255, 0) if det["class"] == "face" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 人脸关键点 → EAR / PERCLOS
            if det["class"] == "face" and det["landmarks"] is not None:
                lmks = np.array(det["landmarks"])  # (14, 2)
                for pt in lmks:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)

                left_ear = compute_ear(lmks[:6])
                right_ear = compute_ear(lmks[6:12])
                avg_ear = (left_ear + right_ear) / 2.0
                perclos_tracker.update(avg_ear)

                if len(lmks) >= 14:
                    mar = compute_mar(lmks[12], lmks[13])
                    yawn_detector.update(mar)

        # 显示 PERCLOS
        perclos = perclos_tracker.perclos
        status = "FATIGUE!" if perclos_tracker.is_fatigued() else "Normal"
        cv2.putText(frame, f"PERCLOS: {perclos:.2f} [{status}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Yawns: {yawn_detector.total_yawns}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("DMS Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
