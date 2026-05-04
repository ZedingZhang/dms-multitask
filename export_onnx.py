"""
模型导出: PyTorch → ONNX
后续可通过 TensorRT 将 ONNX 转为 engine 用于端侧部署
"""

import os
import yaml
import argparse
import torch
from models.dms_net import build_model


def export_onnx(cfg: dict, weights_path: str, output_path: str):
    """将训练好的模型导出为 ONNX 格式."""

    # 构建模型并加载权重
    model = build_model(cfg)
    if os.path.isfile(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded weights from: {weights_path}")
    else:
        print(f"Warning: weights not found at {weights_path}, exporting random weights")

    model.eval()

    # 构造 dummy input
    input_size = cfg["export"]["input_size"]  # [B, C, H, W]
    dummy = torch.randn(*input_size)

    # 导出
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=cfg["export"]["opset_version"],
        input_names=["input"],
        output_names=["cls_p3", "cls_p4", "cls_p5",
                       "bbox_p3", "bbox_p4", "bbox_p5",
                       "lmk_p3", "lmk_p4", "lmk_p5"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"ONNX model exported to: {output_path}")

    # 验证
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")

    # 可选: onnx-simplifier
    if cfg["export"].get("simplify", False):
        try:
            import onnxsim
            model_opt, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_opt, output_path)
                print("ONNX model simplified.")
        except ImportError:
            print("onnxsim not installed, skipping simplification")


def main():
    parser = argparse.ArgumentParser(description="Export DMS model to ONNX")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--weights", type=str, default="weights/best.pt")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    output = args.output or cfg["export"]["onnx_path"]
    export_onnx(cfg, args.weights, output)


if __name__ == "__main__":
    main()
