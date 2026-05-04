/**
 * DMS TensorRT 推理 — 主程序
 *
 * 用法:
 *   # 构建 engine 并推理单张图
 *   ./dms_engine --onnx ../weights/dms_model.onnx \
 *                --engine dms_fp16.engine \
 *                --precision fp16 \
 *                --image test.jpg
 *
 *   # 推理视频
 *   ./dms_engine --engine dms_fp16.engine --video camera.mp4
 *
 *   # 摄像头实时推理
 *   ./dms_engine --engine dms_fp16.engine --video 0
 */

#include "dms_engine.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

using namespace dms;

// ─── 命令行参数 ───
struct Args {
    std::string onnx_path;
    std::string engine_path = "dms_fp16.engine";
    std::string precision   = "fp16";
    std::string image_path;
    std::string video_path;
    float conf  = 0.45f;
    float nms   = 0.45f;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--onnx"      && i + 1 < argc) a.onnx_path   = argv[++i];
        if (arg == "--engine"    && i + 1 < argc) a.engine_path  = argv[++i];
        if (arg == "--precision" && i + 1 < argc) a.precision    = argv[++i];
        if (arg == "--image"     && i + 1 < argc) a.image_path   = argv[++i];
        if (arg == "--video"     && i + 1 < argc) a.video_path   = argv[++i];
        if (arg == "--conf"      && i + 1 < argc) a.conf = std::stof(argv[++i]);
        if (arg == "--nms"       && i + 1 < argc) a.nms  = std::stof(argv[++i]);
    }
    return a;
}

// ─── 可视化 ───
static const char* CLASS_NAMES[] = {"face", "phone", "cigarette"};
static const cv::Scalar COLORS[] = {
    {0, 255, 0},    // face: green
    {0, 0, 255},    // phone: red
    {0, 165, 255},  // cigarette: orange
};

static void draw_detections(cv::Mat& img, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::Rect box(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
        cv::Scalar color = COLORS[d.class_id % 3];
        cv::rectangle(img, box, color, 2);

        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", CLASS_NAMES[d.class_id], d.score);
        cv::putText(img, label, {box.x, box.y - 5},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

        // 绘制人脸关键点
        if (d.class_id == 0) {
            for (int p = 0; p < 14; ++p) {
                int px = static_cast<int>(d.landmarks[p * 2]);
                int py = static_cast<int>(d.landmarks[p * 2 + 1]);
                cv::circle(img, {px, py}, 2, {255, 0, 0}, -1);
            }
        }
    }
}

// ─── 单图推理 ───
static void infer_image(DMSEngine& engine, const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << path << std::endl;
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto dets = engine.infer(img.data, img.rows, img.cols);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Detections: " << dets.size()
              << "  Latency: " << ms << " ms" << std::endl;

    for (const auto& d : dets) {
        std::cout << "  " << CLASS_NAMES[d.class_id]
                  << " " << d.score
                  << " [" << d.x1 << "," << d.y1 << ","
                  << d.x2 << "," << d.y2 << "]" << std::endl;
    }

    draw_detections(img, dets);
    cv::imwrite("result.jpg", img);
    std::cout << "Result saved: result.jpg" << std::endl;
}

// ─── 视频 / 摄像头推理 ───
static void infer_video(DMSEngine& engine, const std::string& source) {
    cv::VideoCapture cap;
    if (source.size() == 1 && std::isdigit(source[0]))
        cap.open(source[0] - '0');
    else
        cap.open(source);

    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << source << std::endl;
        return;
    }

    int frame_count = 0;
    double total_ms = 0;
    cv::Mat frame;

    while (cap.read(frame)) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto dets = engine.infer(frame.data, frame.rows, frame.cols);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        frame_count++;

        draw_detections(frame, dets);

        char info[128];
        snprintf(info, sizeof(info), "FPS: %.1f  Latency: %.1fms",
                 1000.0 / ms, ms);
        cv::putText(frame, info, {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 255}, 2);

        cv::imshow("DMS Engine", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    double avg_ms = total_ms / std::max(frame_count, 1);
    std::cout << "\nProcessed " << frame_count << " frames" << std::endl;
    std::cout << "Avg latency: " << avg_ms << " ms" << std::endl;
    std::cout << "Avg FPS: " << 1000.0 / avg_ms << std::endl;
}

// ─── Main ───
int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    EngineConfig cfg;
    cfg.onnx_path   = args.onnx_path;
    cfg.engine_path = args.engine_path;
    cfg.conf_thresh = args.conf;
    cfg.nms_thresh  = args.nms;

    if (args.precision == "fp16") cfg.precision = Precision::FP16;
    else if (args.precision == "int8") cfg.precision = Precision::INT8;
    else cfg.precision = Precision::FP32;

    DMSEngine engine(cfg);
    if (!engine.build()) {
        std::cerr << "Failed to build engine" << std::endl;
        return 1;
    }

    if (!args.image_path.empty()) {
        infer_image(engine, args.image_path);
    } else if (!args.video_path.empty()) {
        infer_video(engine, args.video_path);
    } else {
        std::cout << "Usage: dms_engine --onnx model.onnx --engine out.engine "
                  << "[--image img.jpg | --video video.mp4 | --video 0]"
                  << std::endl;
    }

    return 0;
}
