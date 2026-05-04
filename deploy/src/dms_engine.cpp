/**
 * DMSEngine 实现 —— TensorRT 推理引擎
 *
 * 流程: ONNX → Build Engine (FP16/INT8) → Serialize
 *       Load Engine → Allocate Buffers → Infer
 */

#include "dms_engine.h"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <cassert>
#include <numeric>
#include <cstring>

namespace dms {

// ─── Logger ───
void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << std::endl;
}

// ─── 构造 / 析构 ───
DMSEngine::DMSEngine(const EngineConfig& cfg) : cfg_(cfg) {
    cudaStreamCreate(&stream_);
}

DMSEngine::~DMSEngine() {
    release_buffers();
    if (d_input_img_)  cudaFree(d_input_img_);
    if (d_input_blob_) cudaFree(d_input_blob_);
    if (stream_)       cudaStreamDestroy(stream_);
}

// ─── Build ───
bool DMSEngine::build() {
    // 优先加载已序列化的 engine
    if (!cfg_.engine_path.empty() && load_engine()) {
        std::cout << "Engine loaded from: " << cfg_.engine_path << std::endl;
        allocate_buffers();
        return true;
    }
    // 从 ONNX 构建
    if (!build_from_onnx()) return false;
    if (!cfg_.engine_path.empty()) save_engine();
    allocate_buffers();
    return true;
}

bool DMSEngine::build_from_onnx() {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) return false;

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(
            1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));

    if (!parser->parseFromFile(cfg_.onnx_path.c_str(),
                                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX: " << cfg_.onnx_path << std::endl;
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 1 GB

    // 精度设置
    if (cfg_.precision == Precision::FP16) {
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cout << "Building with FP16" << std::endl;
        }
    } else if (cfg_.precision == Precision::INT8) {
        if (builder->platformHasFastInt8()) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            // INT8 需要校准器 (此处需用户实现 IInt8Calibrator)
            std::cout << "Building with INT8" << std::endl;
            std::cout << "Note: provide calibrator via setInt8Calibrator()" << std::endl;
        }
    }

    // 构建 Engine
    std::cout << "Building TensorRT engine (this may take minutes)..." << std::endl;
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        std::cerr << "Engine build failed" << std::endl;
        return false;
    }

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime->deserializeCudaEngine(
        serialized->data(), serialized->size()));

    context_.reset(engine_->createExecutionContext());
    std::cout << "Engine built successfully." << std::endl;
    return true;
}

// ─── Serialize / Deserialize ───
bool DMSEngine::save_engine() {
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        engine_->serialize());
    std::ofstream ofs(cfg_.engine_path, std::ios::binary);
    ofs.write(static_cast<const char*>(serialized->data()), serialized->size());
    std::cout << "Engine saved: " << cfg_.engine_path << std::endl;
    return true;
}

bool DMSEngine::load_engine() {
    std::ifstream ifs(cfg_.engine_path, std::ios::binary | std::ios::ate);
    if (!ifs.good()) return false;

    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    ifs.read(data.data(), size);

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime->deserializeCudaEngine(data.data(), size));
    if (!engine_) return false;

    context_.reset(engine_->createExecutionContext());
    return context_ != nullptr;
}

// ─── Buffer 管理 ───
void DMSEngine::allocate_buffers() {
    num_bindings_ = engine_->getNbIOTensors();
    gpu_buffers_.resize(num_bindings_);
    buffer_sizes_.resize(num_bindings_);

    for (int i = 0; i < num_bindings_; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; ++d)
            vol *= (dims.d[d] > 0 ? dims.d[d] : 1);
        size_t bytes = vol * sizeof(float);
        buffer_sizes_[i] = bytes;

        cudaMalloc(&gpu_buffers_[i], bytes);
        std::cout << "Binding [" << name << "] "
                  << dims.d[0];
        for (int d = 1; d < dims.nbDims; ++d)
            std::cout << "x" << dims.d[d];
        std::cout << " (" << bytes / 1024 << " KB)" << std::endl;
    }

    // 预处理用的 GPU 缓冲
    size_t max_img_bytes = 3840 * 2160 * 3;  // 最大 4K
    cudaMalloc(&d_input_img_, max_img_bytes);
    cudaMalloc(&d_input_blob_,
               cfg_.max_batch * 3 * cfg_.input_h * cfg_.input_w * sizeof(float));
}

void DMSEngine::release_buffers() {
    for (auto& buf : gpu_buffers_)
        if (buf) cudaFree(buf);
    gpu_buffers_.clear();
}

// ─── 推理 ───
std::vector<Detection> DMSEngine::infer(const unsigned char* bgr_img,
                                         int img_h, int img_w) {
    // 1. 上传原图到 GPU
    size_t img_bytes = img_h * img_w * 3;
    cudaMemcpyAsync(d_input_img_, bgr_img, img_bytes,
                    cudaMemcpyHostToDevice, stream_);

    // 2. CUDA 预处理: BGR→RGB, Resize, Normalize, HWC→NCHW
    cuda_preprocess(
        static_cast<const unsigned char*>(d_input_img_),
        img_h, img_w,
        static_cast<float*>(gpu_buffers_[0]),  // input binding
        cfg_.input_h, cfg_.input_w,
        stream_
    );

    // 3. TensorRT 推理
    for (int i = 0; i < num_bindings_; ++i) {
        const char* name = engine_->getIOTensorName(i);
        context_->setTensorAddress(name, gpu_buffers_[i]);
    }
    context_->enqueueV3(stream_);

    // 4. 拷贝输出到 CPU
    // 输出 binding 布局: cls_p3, cls_p4, cls_p5, bbox_p3..., lmk_p3...
    // 简化: 拼接所有 cls / bbox / lmk
    int num_outputs = num_bindings_ - 1;  // 减去 input
    std::vector<std::vector<float>> outputs(num_outputs);
    for (int i = 1; i < num_bindings_; ++i) {
        size_t count = buffer_sizes_[i] / sizeof(float);
        outputs[i - 1].resize(count);
        cudaMemcpyAsync(outputs[i - 1].data(), gpu_buffers_[i],
                        buffer_sizes_[i], cudaMemcpyDeviceToHost, stream_);
    }
    cudaStreamSynchronize(stream_);

    // 5. 拼接各 FPN 层输出
    // 假设输出顺序: cls_p3, cls_p4, cls_p5, bbox_p3, bbox_p4, bbox_p5,
    //              lmk_p3, lmk_p4, lmk_p5
    int levels = 3;
    std::vector<float> all_cls, all_bbox, all_lmk;
    for (int i = 0; i < levels; ++i) {
        all_cls.insert(all_cls.end(), outputs[i].begin(), outputs[i].end());
        all_bbox.insert(all_bbox.end(), outputs[levels + i].begin(),
                        outputs[levels + i].end());
        all_lmk.insert(all_lmk.end(), outputs[2 * levels + i].begin(),
                        outputs[2 * levels + i].end());
    }

    // 6. 后处理: 解码 + NMS
    int total_anchors = static_cast<int>(all_cls.size()) / 3;  // num_classes=3
    return postprocess(
        all_cls.data(), all_bbox.data(), all_lmk.data(),
        total_anchors, 3, 14,
        img_h, img_w,
        cfg_.conf_thresh, cfg_.nms_thresh
    );
}

}  // namespace dms
