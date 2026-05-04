/**
 * 后处理模块: Anchor 解码 + NMS
 *
 * 将网络输出的 (cls_scores, bbox_offsets, landmark_offsets)
 * 解码为实际检测框, 并通过 NMS 去重.
 */

#include "dms_engine.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace dms {

// ─── Anchor 预生成 ───
struct Anchor { float cx, cy, w, h; };

static std::vector<Anchor> generate_anchors(int img_h, int img_w) {
    // 与 Python 端 AnchorGenerator 保持一致
    const int steps[]     = {8, 16, 32};
    const int scales[][2] = {{16, 32}, {64, 128}, {256, 512}};
    const int num_levels  = 3;

    std::vector<Anchor> anchors;
    for (int l = 0; l < num_levels; ++l) {
        int fh = img_h / steps[l];
        int fw = img_w / steps[l];
        for (int i = 0; i < fh; ++i) {
            for (int j = 0; j < fw; ++j) {
                float cx = (j + 0.5f) * steps[l] / img_w;
                float cy = (i + 0.5f) * steps[l] / img_h;
                for (int s = 0; s < 2; ++s) {
                    float w = static_cast<float>(scales[l][s]) / img_w;
                    float h = static_cast<float>(scales[l][s]) / img_h;
                    anchors.push_back({cx, cy, w, h});
                }
            }
        }
    }
    return anchors;
}

// ─── Sigmoid ───
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ─── IoU ───
static float iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

// ─── NMS ───
static std::vector<Detection> nms(std::vector<Detection>& dets, float threshold) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;
    result.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] &&
                dets[i].class_id == dets[j].class_id &&
                iou(dets[i], dets[j]) > threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

// ─── 解码 + NMS 主函数 ───
std::vector<Detection> postprocess(
    const float* cls_data,
    const float* bbox_data,
    const float* lmk_data,
    int num_anchors,
    int num_classes,
    int num_landmarks,
    int img_h, int img_w,
    float conf_thresh,
    float nms_thresh)
{
    // 模型训练时用 640×640, anchor 也基于此
    auto anchors = generate_anchors(640, 640);

    // 确保 anchor 数量匹配
    int N = std::min(num_anchors, static_cast<int>(anchors.size()));

    std::vector<Detection> candidates;
    candidates.reserve(N);

    for (int i = 0; i < N; ++i) {
        // 找到该 anchor 最大类别得分
        float max_score = 0;
        int   max_cls   = 0;
        for (int c = 0; c < num_classes; ++c) {
            float s = sigmoid(cls_data[i * num_classes + c]);
            if (s > max_score) {
                max_score = s;
                max_cls   = c;
            }
        }

        if (max_score < conf_thresh) continue;

        // 解码边框
        const Anchor& a = anchors[i];
        float dx = bbox_data[i * 4 + 0];
        float dy = bbox_data[i * 4 + 1];
        float dw = bbox_data[i * 4 + 2];
        float dh = bbox_data[i * 4 + 3];

        float cx = a.cx + dx * a.w;
        float cy = a.cy + dy * a.h;
        float w  = a.w * std::exp(dw);
        float h  = a.h * std::exp(dh);

        Detection det;
        det.x1 = (cx - w / 2) * img_w;
        det.y1 = (cy - h / 2) * img_h;
        det.x2 = (cx + w / 2) * img_w;
        det.y2 = (cy + h / 2) * img_h;
        det.score    = max_score;
        det.class_id = max_cls;

        // 解码关键点 (仅 face)
        std::memset(det.landmarks, 0, sizeof(det.landmarks));
        if (max_cls == 0) {
            for (int p = 0; p < num_landmarks; ++p) {
                float lx = lmk_data[i * num_landmarks * 2 + p * 2 + 0];
                float ly = lmk_data[i * num_landmarks * 2 + p * 2 + 1];
                det.landmarks[p * 2 + 0] = (a.cx + lx * a.w) * img_w;
                det.landmarks[p * 2 + 1] = (a.cy + ly * a.h) * img_h;
            }
        }

        candidates.push_back(det);
    }

    return nms(candidates, nms_thresh);
}

}  // namespace dms
