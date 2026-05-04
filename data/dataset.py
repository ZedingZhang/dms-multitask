"""
DMS 数据集 —— 支持同时加载检测标注和关键点标注

标注格式 (每行一个目标):
  class_id  x1 y1 x2 y2  lm0_x lm0_y lm1_x lm1_y ... lm13_x lm13_y
  - class_id: 0=face, 1=phone, 2=cigarette
  - x1 y1 x2 y2: 归一化边框坐标
  - lm*: 归一化关键点坐标 (仅 face 类有效, 其他类填 -1)
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class DMSDataset(Dataset):
    """DMS 多任务数据集."""

    CLASS_NAMES = ("face", "phone", "cigarette")

    def __init__(self, root: str, img_size: int = 640,
                 num_landmarks: int = 14, transform=None):
        """
        Args:
            root:    数据集根目录, 包含 images/ 和 labels/ 子目录
            img_size: 统一输入尺寸
            num_landmarks: 关键点数量
            transform: albumentations 变换 (可选)
        """
        self.root = root
        self.img_size = img_size
        self.num_landmarks = num_landmarks
        self.transform = transform

        img_dir = os.path.join(root, "images")
        lbl_dir = os.path.join(root, "labels")

        if not os.path.isdir(img_dir):
            # 如果数据目录不存在, 生成 demo 样本用于调试
            self.samples = self._generate_demo_samples(50)
            return

        self.img_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        self.lbl_paths = [
            os.path.join(lbl_dir, os.path.splitext(os.path.basename(p))[0] + ".txt")
            for p in self.img_paths
        ]
        self.samples = None  # 使用真实数据

    def _generate_demo_samples(self, n: int):
        """生成随机 demo 数据用于调试流程."""
        samples = []
        for _ in range(n):
            img = np.random.randint(0, 255, (self.img_size, self.img_size, 3),
                                    dtype=np.uint8)
            num_objs = np.random.randint(1, 4)
            targets = []
            for _ in range(num_objs):
                cls_id = np.random.randint(0, 3)
                cx, cy = np.random.uniform(0.2, 0.8, 2)
                w, h = np.random.uniform(0.05, 0.3, 2)
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                if cls_id == 0:  # face → 生成关键点
                    lms = np.random.uniform(x1, x2, self.num_landmarks * 2)
                    # y 坐标范围
                    lms[1::2] = np.random.uniform(y1, y2, self.num_landmarks)
                else:
                    lms = np.full(self.num_landmarks * 2, -1.0)
                targets.append({
                    "class_id": cls_id,
                    "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
                    "landmarks": lms.astype(np.float32),
                })
            samples.append((img, targets))
        return samples

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return len(self.img_paths)

    def _parse_label(self, path: str):
        """解析标注文件."""
        targets = []
        if not os.path.isfile(path):
            return targets
        with open(path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                bbox = np.array(parts[1:5], dtype=np.float32)
                lms = np.array(parts[5:5 + self.num_landmarks * 2],
                               dtype=np.float32)
                if len(lms) < self.num_landmarks * 2:
                    lms = np.full(self.num_landmarks * 2, -1.0, dtype=np.float32)
                targets.append({
                    "class_id": cls_id,
                    "bbox": bbox,
                    "landmarks": lms,
                })
        return targets

    def __getitem__(self, idx: int):
        if self.samples is not None:
            img, targets = self.samples[idx]
        else:
            img = cv2.imread(self.img_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            targets = self._parse_label(self.lbl_paths[idx])

        # resize
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        # HWC -> CHW, [0, 255] -> [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 整理标注
        cls_ids = []
        bboxes = []
        landmarks = []
        for t in targets:
            cls_ids.append(t["class_id"])
            bboxes.append(t["bbox"])
            landmarks.append(t["landmarks"])

        if len(cls_ids) == 0:
            cls_ids = torch.zeros(0, dtype=torch.long)
            bboxes = torch.zeros(0, 4)
            landmarks = torch.zeros(0, self.num_landmarks * 2)
        else:
            cls_ids = torch.tensor(cls_ids, dtype=torch.long)
            bboxes = torch.tensor(np.stack(bboxes), dtype=torch.float32)
            landmarks = torch.tensor(np.stack(landmarks), dtype=torch.float32)

        return img, {
            "cls_ids": cls_ids,
            "bboxes": bboxes,
            "landmarks": landmarks,
        }


def collate_fn(batch):
    """自定义 collate: 图像堆叠, 标注保持 list."""
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(targets)
