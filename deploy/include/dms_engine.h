/**
 * DMS TensorRT 推理引擎
 *
 * 功能:
 *   1. 从 ONNX 构建 TensorRT Engine (支持 FP16 / INT8)
 *   2. 序列化 / 反序列化 Engine
 *   3. 高效推理: CUDA 预处理 → TensorRT 推理 → NMS 后处理
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace dms {

// ──────────────────────────────────────────────
//  数据结构
// ──────────────────────────────────────────────

struct Detection {
    float x1, y1, x2, y2;          // bounding box
    float score;                    // confidence
    int   class_id;                 // 0=face, 1=phone, 2=cigarette
    float landmarks[28];            // 14 keypoints × 2 (仅 face 有效)
};

enum class Precision { FP32, FP16, INT8 };

struct EngineConfig {
    std::string onnx_path;
    std::string engine_path;
    Precision   precision     = Precision::FP16;
    int         max_batch     = 1;
    int         input_h       = 640;
    int         input_w       = 640;
    float       conf_thresh   = 0.45f;
    float       nms_thresh    = 0.45f;
    std::string calib_dir     = "";   // INT8 校准数据目录
};

// ──────────────────────────────────────────────
//  TensorRT Logger
// ──────────────────────────────────────────────

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// ──────────────────────────────────────────────
//  DMS TensorRT Engine
// ──────────────────────────────────────────────

class DMSEngine {
public:
    explicit DMSEngine(const EngineConfig& cfg);
    ~DMSEngine();

    // 禁止拷贝
    DMSEngine(const DMSEngine&) = delete;
    DMSEngine& operator=(const DMSEngine&) = delete;

    /**
     * 构建 Engine: ONNX → TensorRT, 支持 FP16/INT8
     * 若 engine_path 已存在则直接反序列化
     */
    bool build();

    /**
     * 推理单帧
     * @param bgr_img  OpenCV BGR 图像 (HWC, uint8)
     * @param img_h    原图高
     * @param img_w    原图宽
     * @return 检测结果列表
     */
    std::vector<Detection> infer(const unsigned char* bgr_img,
                                  int img_h, int img_w);

private:
    bool build_from_onnx();
    bool load_engine();
    bool save_engine();
    void allocate_buffers();
    void release_buffers();

    EngineConfig cfg_;
    TrtLogger    logger_;

    // TensorRT 核心对象
    std::shared_ptr<nvinfer1::ICudaEngine>       engine_;
    std::shared_ptr<nvinfer1::IExecutionContext>  context_;

    // GPU 缓冲区
    std::vector<void*> gpu_buffers_;
    std::vector<size_t> buffer_sizes_;
    int num_bindings_ = 0;

    // CUDA 预处理缓冲区
    void* d_input_img_  = nullptr;   // 原始图像 (GPU)
    void* d_input_blob_ = nullptr;   // 预处理后的 NCHW float (GPU)

    cudaStream_t stream_ = nullptr;
};

// ──────────────────────────────────────────────
//  CUDA 预处理 (外部实现在 preprocess.cu)
// ──────────────────────────────────────────────

/**
 * GPU 加速图像预处理:
 *   1. BGR → RGB
 *   2. Bilinear Resize 到 (input_h, input_w)
 *   3. Normalize: /255.0
 *   4. HWC → NCHW
 *
 * 全部在 GPU 上完成, 零拷贝.
 */
void cuda_preprocess(const unsigned char* d_src, int src_h, int src_w,
                     float* d_dst, int dst_h, int dst_w,
                     cudaStream_t stream);

// ──────────────────────────────────────────────
//  NMS 后处理 (外部实现在 postprocess.cpp)
// ──────────────────────────────────────────────

/**
 * 对网络原始输出做解码 + NMS, 返回最终检测结果.
 */
std::vector<Detection> postprocess(
    const float* cls_data,    // 分类得分 (所有 FPN 层拼接)
    const float* bbox_data,   // 边框偏移
    const float* lmk_data,    // 关键点偏移
    int num_anchors,
    int num_classes,
    int num_landmarks,
    int img_h, int img_w,
    float conf_thresh,
    float nms_thresh
);

}  // namespace dms
