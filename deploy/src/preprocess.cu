/**
 * CUDA 加速图像预处理
 *
 * 在 GPU 上完成全部预处理, 避免 CPU-GPU 数据传输瓶颈:
 *   1. BGR → RGB 通道交换
 *   2. Bilinear Resize (src_h × src_w → dst_h × dst_w)
 *   3. Normalize: pixel / 255.0f
 *   4. HWC → NCHW (interleaved → planar)
 */

#include <cuda_runtime.h>
#include <cstdio>

namespace dms {

/**
 * 核心 CUDA Kernel: 对每个输出像素, 从源图做双线性插值并完成格式转换.
 *
 * 每个线程处理输出图的一个像素 (x, y), 产出 3 个通道值写入 planar 布局.
 */
__global__ void preprocess_kernel(
    const unsigned char* __restrict__ src,  // 源图 BGR HWC
    float* __restrict__ dst,                // 目标 NCHW float
    int src_h, int src_w,
    int dst_h, int dst_w)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;  // 输出 x
    int dy = blockIdx.y * blockDim.y + threadIdx.y;  // 输出 y

    if (dx >= dst_w || dy >= dst_h) return;

    // 计算源图上对应的浮点坐标 (bilinear)
    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;

    float sx = (dx + 0.5f) * scale_x - 0.5f;
    float sy = (dy + 0.5f) * scale_y - 0.5f;

    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // clamp
    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));

    float fx = sx - floorf(sx);
    float fy = sy - floorf(sy);

    // 对 B, G, R 三通道做双线性插值
    float val[3];
    for (int c = 0; c < 3; ++c) {
        float v00 = src[(y0 * src_w + x0) * 3 + c];
        float v01 = src[(y0 * src_w + x1) * 3 + c];
        float v10 = src[(y1 * src_w + x0) * 3 + c];
        float v11 = src[(y1 * src_w + x1) * 3 + c];

        val[c] = (1 - fy) * ((1 - fx) * v00 + fx * v01) +
                      fy  * ((1 - fx) * v10 + fx * v11);
    }

    // BGR → RGB + Normalize (/255) + HWC → NCHW
    int area = dst_h * dst_w;
    int idx  = dy * dst_w + dx;

    dst[0 * area + idx] = val[2] / 255.0f;  // R
    dst[1 * area + idx] = val[1] / 255.0f;  // G
    dst[2 * area + idx] = val[0] / 255.0f;  // B
}

/**
 * 主机端接口: 启动预处理 Kernel
 */
void cuda_preprocess(const unsigned char* d_src, int src_h, int src_w,
                     float* d_dst, int dst_h, int dst_w,
                     cudaStream_t stream)
{
    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, src_h, src_w, dst_h, dst_w);
}

}  // namespace dms
