# DMS-MultiTask

轻量级驾驶员监控系统（Driver Monitoring System）多任务视觉感知模型，同时完成目标检测、面部关键点回归和疲劳行为分析。

## 功能

- **目标检测** — 检测 3 类目标：人脸 (face)、手机 (phone)、香烟 (cigarette)
- **关键点检测** — 14 点面部关键点（左眼 6 + 右眼 6 + 嘴角 2）
- **疲劳预警** — 通过 EAR/PERCLOS 检测疲劳，通过 MAR 检测打哈欠

## 架构

```
输入 (640×640)
    │
    ▼
MobileNetV3-Small Backbone (共享特征)
    │ C3 / C4 / C5 (stride 8/16/32)
    ▼
LiteFPN Neck (多尺度融合, 64 通道)
    │ P3 / P4 / P5
    ├──────────────┐
    ▼              ▼
DetectionHead   LandmarkHead
(分类+边框)     (关键点回归)
```

模型仅 **1.14M 参数**，适合端侧部署。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 准备数据

```
data/
├── train/
│   ├── images/    # 训练图像 (.jpg/.png)
│   └── labels/    # 标注文件 (.txt)
└── val/
    ├── images/    # 验证图像
    └── labels/    # 标注文件
```

标注格式（每行一个目标）：

```
class_id  x1 y1 x2 y2  lm0_x lm0_y ... lm13_x lm13_y
```

- `class_id`: 0=face, 1=phone, 2=cigarette
- `x1 y1 x2 y2`: 归一化边界框 [0, 1]
- `lm*`: 归一化关键点坐标（非人脸目标填 -1）

> 无数据时自动生成随机 demo 样本用于调试。

### 训练

```bash
python train.py --config configs/default.yaml
```

训练配置在 [configs/default.yaml](configs/default.yaml)，支持：
- Cosine LR + Warmup
- EMA 指数移动平均
- 自动混合精度 (AMP)
- 多任务不确定性加权 (Kendall et al. 2018)
- 断点续训

### 评估

```bash
python eval.py --config configs/default.yaml --weights weights/best.pt
```

评估指标：mAP@0.5、mAP@0.5:0.95、Recall、NME、FR@0.08、延迟、FPS。

### 实时推理

```bash
python demo_infer.py --weights weights/best.pt --source 0    # 摄像头
python demo_infer.py --weights weights/best.pt --source video.mp4
```

## 模型压缩

| 方法 | 命令 |
|---|---|
| 知识蒸馏 | `python train_distill.py --config configs/default.yaml --teacher_weights weights/teacher.pt` |
| 结构化剪枝 | `python prune.py --weights weights/best.pt --ratio 0.4 --finetune_epochs 20` |
| 训练后量化 | `python quantize.py --mode pytorch` 或 `--mode tensorrt` |

## 部署

导出 ONNX：

```bash
python export_onnx.py --config configs/default.yaml --weights weights/best.pt
```

TensorRT C++ 推理：

```bash
cd deploy && mkdir build && cd build
cmake .. && make -j
./dms_engine --onnx ../../weights/dms_model.onnx --image test.jpg
```

## 项目结构

```
├── configs/          # 配置文件
├── data/             # 数据集加载与增强
├── models/           # 模型架构 (Backbone/Neck/Heads)
├── utils/            # Anchor / 损失函数 / 行为指标
├── distill/          # 知识蒸馏
├── deploy/           # C++ TensorRT 部署
├── train.py          # 主训练脚本
├── train_distill.py  # 蒸馏训练
├── eval.py           # 性能评估
├── demo_infer.py     # 实时推理 Demo
├── export_onnx.py    # ONNX 导出
├── prune.py          # 结构化剪枝
└── quantize.py       # 训练后量化
```

## License

MIT
