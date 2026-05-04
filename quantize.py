"""
训练后量化 (Post-Training Quantization, PTQ) —— 量化至 INT8

支持两种方式:
  1. PyTorch 静态量化 (CPU 部署)
  2. TensorRT INT8 校准 (GPU 端侧部署, 生成 calibration cache)

用法:
  # PyTorch 静态量化
  python quantize.py --config configs/default.yaml --weights weights/best.pt --mode pytorch

  # TensorRT INT8 校准表生成
  python quantize.py --config configs/default.yaml --weights weights/best.pt --mode tensorrt
"""

import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dms_net import build_model
from data.dataset import DMSDataset, collate_fn


# ════════════════════════════════════════════════════════════
#  方式 1: PyTorch 静态量化 (CPU)
# ════════════════════════════════════════════════════════════

class QuantizableDMSNet(nn.Module):
    """为 PyTorch 静态量化添加 QuantStub / DeQuantStub."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.model = model

    def forward(self, x):
        x = self.quant(x)
        cls_scores, bbox_preds, lmk_preds = self.model(x)
        cls_scores = [self.dequant(c) for c in cls_scores]
        bbox_preds = [self.dequant(b) for b in bbox_preds]
        lmk_preds = [self.dequant(l) for l in lmk_preds]
        return cls_scores, bbox_preds, lmk_preds


def pytorch_ptq(model: nn.Module, calib_loader: DataLoader, device: torch.device):
    """
    PyTorch 静态量化流程:
      1. 准备量化配置 (per-channel weight, histogram observer for activation)
      2. 插入 observer, 跑校准集收集统计量
      3. 转为量化模型
    """
    model_cpu = model.cpu()
    model_cpu.eval()

    q_model = QuantizableDMSNet(model_cpu)

    # 量化配置: 对称量化 weight, 直方图 observer activation
    q_model.qconfig = torch.ao.quantization.get_default_qconfig("x86")

    # fuse 常见模式: Conv-BN-ReLU
    fused_modules_list = []
    for name, m in q_model.model.named_modules():
        if isinstance(m, nn.Sequential):
            children = list(m.named_children())
            i = 0
            while i < len(children) - 1:
                names = [f"model.{name}.{children[j][0]}" if name else
                         f"model.{children[j][0]}" for j in range(i, min(i + 3, len(children)))]
                types = [type(children[j][1]) for j in range(i, min(i + 3, len(children)))]

                if len(types) >= 3 and types[0] == nn.Conv2d and \
                   types[1] == nn.BatchNorm2d and \
                   isinstance(children[i + 2][1], (nn.ReLU, nn.ReLU6)):
                    fused_modules_list.append(names[:3])
                    i += 3
                elif len(types) >= 2 and types[0] == nn.Conv2d and \
                     types[1] == nn.BatchNorm2d:
                    fused_modules_list.append(names[:2])
                    i += 2
                else:
                    i += 1

    if fused_modules_list:
        try:
            torch.ao.quantization.fuse_modules(q_model, fused_modules_list,
                                                inplace=True)
        except Exception:
            print("Warning: auto-fusion skipped, proceeding without fusion")

    torch.ao.quantization.prepare(q_model, inplace=True)

    # 校准: 跑几个 batch 收集激活值分布
    print("Running calibration...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(calib_loader, desc="Calibrating")):
            q_model(images.cpu())
            if i >= 50:  # 50 batch 足以覆盖激活分布
                break

    # 转为量化模型
    torch.ao.quantization.convert(q_model, inplace=True)
    print("PyTorch static quantization complete.")
    return q_model


# ════════════════════════════════════════════════════════════
#  方式 2: TensorRT INT8 校准数据生成
# ════════════════════════════════════════════════════════════

class TensorRTCalibrator:
    """
    生成 TensorRT INT8 校准所需的二进制数据.
    TensorRT 在构建 engine 时读取这些数据来确定每层的量化范围.

    输出:
      - calib_data/  目录: 包含 N 个 .bin 文件 (FP32 NCHW 格式)
      - calib_list.txt: 文件列表
    """

    def __init__(self, loader: DataLoader, save_dir: str = "calib_data",
                 max_batches: int = 100):
        self.loader = loader
        self.save_dir = save_dir
        self.max_batches = max_batches
        os.makedirs(save_dir, exist_ok=True)

    def generate(self):
        """遍历数据集, 保存为 .bin 文件."""
        file_list = []
        for i, (images, _) in enumerate(tqdm(self.loader, desc="Generating calib data")):
            if i >= self.max_batches:
                break
            # 每个 batch 保存一个文件
            fp = os.path.join(self.save_dir, f"batch_{i:04d}.bin")
            images.numpy().astype(np.float32).tofile(fp)
            file_list.append(fp)

        # 写入文件列表
        list_path = os.path.join(self.save_dir, "calib_list.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(file_list))

        print(f"Calibration data saved: {len(file_list)} batches → {self.save_dir}/")
        print(f"File list: {list_path}")

        # 生成 TensorRT python 校准脚本
        self._write_trt_calibrator_script()
        return list_path

    def _write_trt_calibrator_script(self):
        """输出可直接用于 TensorRT Python API 的校准器代码."""
        script = '''"""
TensorRT INT8 Calibrator — 在构建 engine 时使用
"""
import os
import numpy as np

try:
    import tensorrt as trt

    class DMSCalibrator(trt.IInt8EntropyCalibrator2):
        """INT8 熵校准器."""

        def __init__(self, calib_dir="calib_data", batch_size=1,
                     input_shape=(3, 640, 640), cache_file="dms_int8.cache"):
            super().__init__()
            self.cache_file = cache_file
            self.batch_size = batch_size
            self.input_shape = input_shape

            list_path = os.path.join(calib_dir, "calib_list.txt")
            with open(list_path) as f:
                self.files = [l.strip() for l in f if l.strip()]
            self.idx = 0

            import pycuda.driver as cuda
            import pycuda.autoinit
            nbytes = batch_size * int(np.prod(input_shape)) * 4
            self.device_input = cuda.mem_alloc(nbytes)

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.idx >= len(self.files):
                return None
            data = np.fromfile(self.files[self.idx], dtype=np.float32)
            data = data.reshape(self.batch_size, *self.input_shape)
            import pycuda.driver as cuda
            cuda.memcpy_htod(self.device_input, data)
            self.idx += 1
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if os.path.isfile(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                f.write(cache)

except ImportError:
    print("TensorRT not installed, calibrator not available")
'''
        path = os.path.join(self.save_dir, "trt_calibrator.py")
        with open(path, "w") as f:
            f.write(script)
        print(f"TensorRT calibrator script: {path}")


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser("DMS Post-Training Quantization")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--mode", choices=["pytorch", "tensorrt"], default="tensorrt")
    parser.add_argument("--calib_batches", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # 加载模型
    model = build_model(cfg)
    if os.path.isfile(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        print(f"Loaded: {args.weights}")

    # 校准数据集
    calib_set = DMSDataset(
        root=cfg["data"]["val_root"],
        img_size=cfg["train"]["img_size"],
        num_landmarks=cfg["model"]["num_landmarks"],
    )
    calib_loader = DataLoader(
        calib_set, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    if args.mode == "pytorch":
        # ---- PyTorch 静态量化 ----
        q_model = pytorch_ptq(model, calib_loader, device)
        os.makedirs("weights", exist_ok=True)
        save_path = "weights/quantized_int8.pt"
        torch.save(q_model.state_dict(), save_path)
        print(f"Quantized model saved: {save_path}")

    else:
        # ---- TensorRT INT8 校准数据 ----
        # 先确保 ONNX 模型存在
        onnx_path = cfg["export"]["onnx_path"]
        if not os.path.isfile(onnx_path):
            print(f"ONNX model not found at {onnx_path}")
            print("Run `python export_onnx.py` first, then use trtexec:")
            print(f"  trtexec --onnx={onnx_path} --int8 "
                  f"--calib=calib_data/dms_int8.cache "
                  f"--saveEngine=weights/dms_int8.engine")

        calibrator = TensorRTCalibrator(
            calib_loader, save_dir="calib_data",
            max_batches=args.calib_batches,
        )
        calibrator.generate()

        print("\n=== 使用 trtexec 构建 INT8 Engine ===")
        print(f"  trtexec --onnx={onnx_path} \\")
        print(f"          --int8 --calib=calib_data/dms_int8.cache \\")
        print(f"          --saveEngine=weights/dms_int8.engine \\")
        print(f"          --workspace=1024")


if __name__ == "__main__":
    main()
