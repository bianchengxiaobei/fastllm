//
// Created by huangyuyang on 10/30/25.
//

#include <cstdint>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifdef __AVX2__
#include "immintrin.h"
#endif

#include "utils.h"
#include "fastllm.h"

namespace fastllm {
#ifdef __AVX2__
    // BF16到FP32的转换辅助函数
    inline __m256 bf16_to_fp32_avx2(__m128i bf16_data) {
        // BF16只是FP32的高16位，所以左移16位即可
        __m256i fp32_data = _mm256_cvtepu16_epi32(bf16_data);
        fp32_data = _mm256_slli_epi32(fp32_data, 16);
        return _mm256_castsi256_ps(fp32_data);
    }
#endif

    void AddBiasAVX2(float *outputData, float *biasData, int n, int k, int st, int end) {
#ifdef __AVX2__
        if (biasData) {
            for (int i = 0; i < n; i++) {
                int j = st;
                for (; j + 7 < end; j += 8) {                    
                    _mm256_storeu_ps(outputData + i * k + j, 
                        _mm256_add_ps(_mm256_loadu_ps(outputData + i * k + j), _mm256_loadu_ps(biasData + j)));
                }
                for (; j < end; j++) {
                    outputData[i * k + j] += biasData[j];
                }
            }
        }
#endif
    }

    void Float32ToInfInt8PerChannelAVX2(const float* srcData, uint8_t* dstData, size_t columns) {
#ifdef __AVX2__
        // 目标内存布局：
        // [int8 * columns] [float scale] [int sum]
        int8_t* quantizedData = (int8_t*)dstData;
        float* scalePtr = (float*)(dstData + columns);
        int* sumPtr = (int*)(dstData + columns + sizeof(float));
        
        // 1. 找到这一行的最大绝对值
        float maxAbs = 0.0f;
        size_t i = 0;
        
        // AVX2 向量化处理，每次处理8个float
        __m256 vMaxAbs = _mm256_setzero_ps();
        for (; i + 7 < columns; i += 8) {
            __m256 vData = _mm256_loadu_ps(&srcData[i]);
            // 计算绝对值：通过清除符号位
            __m256 vAbs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vData);
            // 更新最大值
            vMaxAbs = _mm256_max_ps(vMaxAbs, vAbs);
        }
        
        // 水平归约找到最大值
        __m128 vMax128 = _mm_max_ps(_mm256_castps256_ps128(vMaxAbs), 
                                    _mm256_extractf128_ps(vMaxAbs, 1));
        vMax128 = _mm_max_ps(vMax128, _mm_shuffle_ps(vMax128, vMax128, _MM_SHUFFLE(1, 0, 3, 2)));
        vMax128 = _mm_max_ps(vMax128, _mm_shuffle_ps(vMax128, vMax128, _MM_SHUFFLE(2, 3, 0, 1)));
        maxAbs = _mm_cvtss_f32(vMax128);
        
        // 处理剩余元素
        for (; i < columns; i++) {
            float absVal = std::abs(srcData[i]);
            if (absVal > maxAbs) {
                maxAbs = absVal;
            }
        }
        
        // 2. 计算scale
        float scale;
        if (maxAbs > 0) {
            scale = maxAbs / 127.0f;
        } else {
            scale = 1.0f;
        }
        float invScale = 1.0f / scale;
        
        // 3. 量化并计算sum
        __m256i vSum = _mm256_setzero_si256();
        __m256 vInvScale = _mm256_set1_ps(invScale);
        __m256i vMax127 = _mm256_set1_epi32(127);
        __m256i vMin127 = _mm256_set1_epi32(-127);
        
        i = 0;
        // 主循环：每次处理8个float
        for (; i + 7 < columns; i += 8) {
            // 加载float数据
            __m256 vData = _mm256_loadu_ps(&srcData[i]);
            
            // 量化: q = round(x / scale)
            __m256 vScaled = _mm256_mul_ps(vData, vInvScale);
            __m256i vQuantized = _mm256_cvtps_epi32(vScaled);  // 四舍五入转换
            
            // 裁剪到 [-127, 127]
            vQuantized = _mm256_min_epi32(vQuantized, vMax127);
            vQuantized = _mm256_max_epi32(vQuantized, vMin127);
            
            // 累加到sum
            vSum = _mm256_add_epi32(vSum, vQuantized);
            
            // 打包成int8并存储
            // AVX2方法：将8个int32打包成8个int8
            // 首先提取低128位和高128位
            __m128i vLow = _mm256_castsi256_si128(vQuantized);
            __m128i vHigh = _mm256_extracti128_si256(vQuantized, 1);
            
            // 将两个__m128i（各含4个int32）打包成一个__m128i（含8个int16）
            __m128i vPacked16 = _mm_packs_epi32(vLow, vHigh);
            
            // 将int16打包成int8（由于我们已经限制在[-127,127]范围内，所以不会溢出）
            __m128i vPacked8 = _mm_packs_epi16(vPacked16, _mm_setzero_si128());
            
            // 存储8个int8
            _mm_storel_epi64((__m128i*)&quantizedData[i], vPacked8);
        }
        
        // 水平求和
        __m128i vSum128 = _mm_add_epi32(_mm256_castsi256_si128(vSum), 
                                        _mm256_extracti128_si256(vSum, 1));
        vSum128 = _mm_hadd_epi32(vSum128, vSum128);
        vSum128 = _mm_hadd_epi32(vSum128, vSum128);
        int sum = _mm_cvtsi128_si32(vSum128);
        
        // 处理剩余元素
        for (; i < columns; i++) {
            int quantized = std::round(srcData[i] * invScale);
            
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;
            
            quantizedData[i] = (int8_t)quantized;
            sum += quantized;
        }
        
        // 4. 存储scale和sum
        *scalePtr = scale;
        *sumPtr = sum;
#endif
    }

    template <int BROW, int AROW>
    void mul_mat_bf16_bf16_direct_avx2(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX2__
        constexpr int SIMD_WIDTH_FP32 = 8;  // AVX2 一次处理 8 个 float
        constexpr int SIMD_WIDTH_BF16 = 8;  // 一次加载 8 个 bf16
        
        int nb = n / SIMD_WIDTH_BF16;
        int remainder = n % SIMD_WIDTH_BF16;
        
        // 累加器
        __m256 acc[AROW * BROW];
        
        // 初始化累加器
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm256_setzero_ps();
        }
        
        // 主循环 - 每次处理8个元素
        for (int i = 0; i < nb; ++i) {
            // 对每个A的行
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                // 加载8个BF16值并转换为FP32
                __m128i a_bf16 = _mm_loadu_si128((__m128i const*)(a_row + i * SIMD_WIDTH_BF16));
                __m256 a_fp32 = bf16_to_fp32_avx2(a_bf16);
                
                // 对每个B的行
                for (int iy = 0; iy < BROW; ++iy) {
                    const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                    // 加载8个BF16值并转换为FP32
                    __m128i b_bf16 = _mm_loadu_si128((__m128i const*)(b_row + i * SIMD_WIDTH_BF16));
                    __m256 b_fp32 = bf16_to_fp32_avx2(b_bf16);
                    
                    // 执行FMA操作
                    int acc_idx = ix * BROW + iy;
                    acc[acc_idx] = _mm256_fmadd_ps(a_fp32, b_fp32, acc[acc_idx]);
                }
            }
        }

        // 水平求和并存储结果
        for (int ix = 0; ix < AROW; ++ix) {
            for (int iy = 0; iy < BROW; ++iy) {
                int acc_idx = ix * BROW + iy;
                
                // 水平求和 AVX2版本
                __m256 sum = acc[acc_idx];
                __m128 sum_low = _mm256_castps256_ps128(sum);
                __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                
                // 继续水平求和
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                
                float result = _mm_cvtss_f32(sum128);
                
                // 存储结果
                float* c_row = (float*)((char*)C + iy * stride_c);
                c_row[ix] = result;
            }
        }
#endif
    }

    // 更高效的版本 - 一次处理多个8元素块
    template <int BROW, int AROW>
    void mul_mat_bf16_bf16_direct_avx2_optimized(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX2__
        constexpr int UNROLL = 4;  // 展开因子
        constexpr int SIMD_WIDTH = 8;
        
        int nb = n / (SIMD_WIDTH * UNROLL);
        int remainder = n % (SIMD_WIDTH * UNROLL);
        
        // 累加器
        __m256 acc[AROW * BROW];
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm256_setzero_ps();
        }
        
        // 主循环 - 展开4次
        for (int i = 0; i < nb; ++i) {
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                
                // 预加载4组A数据
                __m256 a_fp32[UNROLL];
                for (int u = 0; u < UNROLL; ++u) {
                    __m128i a_bf16 = _mm_loadu_si128((__m128i const*)(a_row + (i * UNROLL + u) * SIMD_WIDTH));
                    a_fp32[u] = bf16_to_fp32_avx2(a_bf16);
                }
                
                for (int iy = 0; iy < BROW; ++iy) {
                    const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                    int acc_idx = ix * BROW + iy;
                    
                    // 展开的内部循环
                    for (int u = 0; u < UNROLL; ++u) {
                        __m128i b_bf16 = _mm_loadu_si128((__m128i const*)(b_row + (i * UNROLL + u) * SIMD_WIDTH));
                        __m256 b_fp32 = bf16_to_fp32_avx2(b_bf16);
                        acc[acc_idx] = _mm256_fmadd_ps(a_fp32[u], b_fp32, acc[acc_idx]);
                    }
                }
            }
        }
        
        // 处理剩余部分
        int start = nb * SIMD_WIDTH * UNROLL;
        for (int i = start; i < n; ++i) {
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                uint32_t a_val_int = ((uint32_t)a_row[i]) << 16;
                float a_val = *((float*)&a_val_int);
                
                for (int iy = 0; iy < BROW; ++iy) {
                    const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                    uint32_t b_val_int = ((uint32_t)b_row[i]) << 16;
                    float b_val = *((float*)&b_val_int);
                    
                    int acc_idx = ix * BROW + iy;
                    float scalar_result = a_val * b_val;
                    
                    // 使用标量加法
                    __m256 temp = acc[acc_idx];
                    float* temp_arr = (float*)&temp;
                    temp_arr[0] += scalar_result;
                    acc[acc_idx] = temp;
                }
            }
        }
        
        // 水平求和并存储
        for (int ix = 0; ix < AROW; ++ix) {
            for (int iy = 0; iy < BROW; ++iy) {
                int acc_idx = ix * BROW + iy;
                
                __m256 sum = acc[acc_idx];
                __m128 sum_low = _mm256_castps256_ps128(sum);
                __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                
                float* c_row = (float*)((char*)C + iy * stride_c);
                c_row[ix] = _mm_cvtss_f32(sum128);
            }
        }
#endif
    }

    template <int BROW>
    void mul_mat_bf16_bf16_direct_avx2_pair(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX2__
        constexpr int SIMD_WIDTH = 8;
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;

        __m256 acc[2][BROW];
        for (int c = 0; c < 2; c++) {
            for (int r = 0; r < BROW; r++) {
                acc[c][r] = _mm256_setzero_ps();
            }
        }

        const uint16_t* a0 = A;
        const uint16_t* a1 = (const uint16_t*)((const char*)A + stride_a);

        for (int i = 0; i < nb; ++i) {
            __m256 w0 = bf16_to_fp32_avx2(
                _mm_loadu_si128((const __m128i*)(a0 + i * SIMD_WIDTH)));
            __m256 w1 = bf16_to_fp32_avx2(
                _mm_loadu_si128((const __m128i*)(a1 + i * SIMD_WIDTH)));
            for (int r = 0; r < BROW; ++r) {
                const uint16_t* b_row =
                    (const uint16_t*)((const char*)B + r * stride_b);
                __m256 iv = bf16_to_fp32_avx2(
                    _mm_loadu_si128((const __m128i*)(b_row + i * SIMD_WIDTH)));
                acc[0][r] = _mm256_fmadd_ps(iv, w0, acc[0][r]);
                acc[1][r] = _mm256_fmadd_ps(iv, w1, acc[1][r]);
            }
        }

        float tailSum[2][BROW];
        memset(tailSum, 0, sizeof(tailSum));
        for (int j = 0; j < remainder; ++j) {
            int idx = nb * SIMD_WIDTH + j;
            uint32_t ta0 = (uint32_t)a0[idx] << 16;
            uint32_t ta1 = (uint32_t)a1[idx] << 16;
            float a0v, a1v;
            memcpy(&a0v, &ta0, sizeof(a0v));
            memcpy(&a1v, &ta1, sizeof(a1v));
            for (int r = 0; r < BROW; ++r) {
                const uint16_t* b_row =
                    (const uint16_t*)((const char*)B + r * stride_b);
                uint32_t tb = (uint32_t)b_row[idx] << 16;
                float bv;
                memcpy(&bv, &tb, sizeof(bv));
                tailSum[0][r] += a0v * bv;
                tailSum[1][r] += a1v * bv;
            }
        }

        for (int c = 0; c < 2; c++) {
            for (int r = 0; r < BROW; r++) {
                float* c_row = (float*)((char*)C + r * stride_c);
                c_row[c] = Floatsum(acc[c][r]) + tailSum[c][r];
            }
        }
#endif
    }

    // bf16 weight (A, 1 column) x fp32 input (B, BROW rows) -> fp32 output (1 column)
    template <int BROW>
    void mul_mat_bf16_f32_direct_avx2(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const float* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX2__
        constexpr int SIMD_WIDTH = 8;
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;

        __m256 acc[BROW];
        for (int r = 0; r < BROW; r++) {
            acc[r] = _mm256_setzero_ps();
        }

        const uint16_t* a0 = A;
        for (int i = 0; i < nb; ++i) {
            __m256 w0 = bf16_to_fp32_avx2(
                _mm_loadu_si128((const __m128i*)(a0 + i * SIMD_WIDTH)));
            for (int r = 0; r < BROW; ++r) {
                const float* b_row =
                    (const float*)((const char*)B + r * stride_b);
                __m256 iv = _mm256_loadu_ps(b_row + i * SIMD_WIDTH);
                acc[r] = _mm256_fmadd_ps(iv, w0, acc[r]);
            }
        }

        float tailSum[BROW];
        memset(tailSum, 0, sizeof(tailSum));
        for (int j = 0; j < remainder; ++j) {
            int idx = nb * SIMD_WIDTH + j;
            uint32_t wa0 = (uint32_t)a0[idx] << 16;
            float w0v;
            memcpy(&w0v, &wa0, sizeof(w0v));
            for (int r = 0; r < BROW; ++r) {
                const float* b_row =
                    (const float*)((const char*)B + r * stride_b);
                tailSum[r] += w0v * b_row[idx];
            }
        }

        for (int r = 0; r < BROW; r++) {
            float* c_row = (float*)((char*)C + r * stride_c);
            c_row[0] = Floatsum(acc[r]) + tailSum[r];
        }
#endif
    }

    // bf16 weight (A, 2 columns) x fp32 input (B, BROW rows) -> fp32 output (2 columns)
    template <int BROW>
    void mul_mat_bf16_f32_direct_avx2_pair(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const float* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX2__
        constexpr int SIMD_WIDTH = 8;
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;

        __m256 acc[2][BROW];
        for (int c = 0; c < 2; c++) {
            for (int r = 0; r < BROW; r++) {
                acc[c][r] = _mm256_setzero_ps();
            }
        }

        const uint16_t* a0 = A;
        const uint16_t* a1 = (const uint16_t*)((const char*)A + stride_a);

        for (int i = 0; i < nb; ++i) {
            __m256 w0 = bf16_to_fp32_avx2(
                _mm_loadu_si128((const __m128i*)(a0 + i * SIMD_WIDTH)));
            __m256 w1 = bf16_to_fp32_avx2(
                _mm_loadu_si128((const __m128i*)(a1 + i * SIMD_WIDTH)));
            for (int r = 0; r < BROW; ++r) {
                const float* b_row =
                    (const float*)((const char*)B + r * stride_b);
                __m256 iv = _mm256_loadu_ps(b_row + i * SIMD_WIDTH);
                acc[0][r] = _mm256_fmadd_ps(iv, w0, acc[0][r]);
                acc[1][r] = _mm256_fmadd_ps(iv, w1, acc[1][r]);
            }
        }

        float tailSum[2][BROW];
        memset(tailSum, 0, sizeof(tailSum));
        for (int j = 0; j < remainder; ++j) {
            int idx = nb * SIMD_WIDTH + j;
            uint32_t wa0 = (uint32_t)a0[idx] << 16;
            uint32_t wa1 = (uint32_t)a1[idx] << 16;
            float w0v, w1v;
            memcpy(&w0v, &wa0, sizeof(w0v));
            memcpy(&w1v, &wa1, sizeof(w1v));
            for (int r = 0; r < BROW; r++) {
                const float* b_row =
                    (const float*)((const char*)B + r * stride_b);
                float bv = b_row[idx];
                tailSum[0][r] += w0v * bv;
                tailSum[1][r] += w1v * bv;
            }
        }

        for (int c = 0; c < 2; c++) {
            for (int r = 0; r < BROW; r++) {
                float* c_row = (float*)((char*)C + r * stride_c);
                c_row[c] = Floatsum(acc[c][r]) + tailSum[c][r];
            }
        }
#endif
    }

    bool LinearBFloat16BFloat16_AVX2_Kernel(
        uint16_t *inputData,
        uint16_t *weightData,
        float *biasData,
        float *outputData,
        int n, int m, int k, int st, int end)
    {
        // 输入按超块先转成 fp32，权重行再作外层循环：每对权重行只流式读取
        // 一次，遍历块内所有 token 时都驻留在 L1/L2。原来 token 块在外层的
        // 顺序每 5 个 token 就要把 [st, end) 的整块权重重新读一遍。
        // 超块限制 scratch 大小，n 更大时权重最多重读 ceil(n / 128) 次。
        constexpr int superBlock = 128;
        thread_local std::vector<float> fp32Input;
        if ((size_t)fp32Input.size() < (size_t)superBlock * m) {
            fp32Input.resize((size_t)superBlock * m);
        }

        static const bool profileKernel =
            std::getenv("FASTLLM_PROFILE_BF16_KERNEL") != nullptr;
        double convertSeconds = 0, fmaSeconds = 0;
        auto phaseStart = std::chrono::steady_clock::now();

        for (int iSuper = 0; iSuper < n; iSuper += superBlock) {
            const int superRows = std::min(superBlock, n - iSuper);
            for (int r = 0; r < superRows; r++) {
                const uint16_t *src = inputData + (size_t)(iSuper + r) * m;
                float *dst = fp32Input.data() + (size_t)r * m;
                int l = 0;
                for (; l + 7 < m; l += 8) {
                    _mm256_storeu_ps(dst + l, bf16_to_fp32_avx2(
                        _mm_loadu_si128((const __m128i*)(src + l))));
                }
                for (; l < m; l++) {
                    uint32_t x = (uint32_t)src[l] << 16;
                    memcpy(dst + l, &x, sizeof(float));
                }
            }
            auto fmaStart = std::chrono::steady_clock::now();
            convertSeconds +=
                std::chrono::duration<double>(fmaStart - phaseStart).count();

            const int tailBegin = superRows - superRows % 5;
            int j = st;
            for (; j + 1 < end; j += 2) {
                uint16_t *weightPair = weightData + (size_t)j * m;
                for (int i = 0; i < tailBegin; i += 5) {
                    mul_mat_bf16_f32_direct_avx2_pair<5>(m, weightPair, m * sizeof(uint16_t),
                        fp32Input.data() + (size_t)i * m, m * sizeof(float),
                        outputData + (size_t)(iSuper + i) * k + j, k * sizeof(float));
                }
                switch (superRows - tailBegin) {
                    case 0: break;
                    case 1: mul_mat_bf16_f32_direct_avx2_pair<1>(m, weightPair, m * sizeof(uint16_t),
                        fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                        outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                    case 2: mul_mat_bf16_f32_direct_avx2_pair<2>(m, weightPair, m * sizeof(uint16_t),
                        fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                        outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                    case 3: mul_mat_bf16_f32_direct_avx2_pair<3>(m, weightPair, m * sizeof(uint16_t),
                        fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                        outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                    case 4: mul_mat_bf16_f32_direct_avx2_pair<4>(m, weightPair, m * sizeof(uint16_t),
                        fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                        outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                }
            }
            if (j < end) {
                uint16_t *weightRow = weightData + (size_t)j * m;
                for (int i = 0; i < tailBegin; i += 5) {
                    mul_mat_bf16_f32_direct_avx2<5>(m, weightRow, m * sizeof(uint16_t),
                        fp32Input.data() + (size_t)i * m, m * sizeof(float),
                        outputData + (size_t)(iSuper + i) * k + j, k * sizeof(float));
                }
                switch (superRows - tailBegin) {
                    case 0: break;
                    case 1: mul_mat_bf16_f32_direct_avx2<1>(m, weightRow, m * sizeof(uint16_t),
                        fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                        outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                    case 2: mul_mat_bf16_f32_direct_avx2<2>(m, weightRow, m * sizeof(uint16_t),
                        fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                        outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                    case 3: mul_mat_bf16_f32_direct_avx2<3>(m, weightRow, m * sizeof(uint16_t),
                        fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                        outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                    case 4: mul_mat_bf16_f32_direct_avx2<4>(m, weightRow, m * sizeof(uint16_t),
                        fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                        outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                }
            }
            auto blockEnd = std::chrono::steady_clock::now();
            fmaSeconds +=
                std::chrono::duration<double>(blockEnd - fmaStart).count();
            phaseStart = blockEnd;
        }

        const auto biasStart = std::chrono::steady_clock::now();
        AddBiasAVX2(outputData, biasData, n, k, st, end);
        const auto biasEnd = std::chrono::steady_clock::now();
        const double biasSeconds =
            std::chrono::duration<double>(biasEnd - biasStart).count();

        if (profileKernel) {
            printf(
                "[fastllm-profile-avx2-bf16] n=%d m=%d k=%d threads=1 "
                "convert=%.6f fma=%.6f bias=%.6f\n",
                n, m, k, convertSeconds, fmaSeconds, biasSeconds);
        }
        return true;
    }

    // FP8 E4M3 x BF16 GEMV kernels.  Decode streams the weight bytes from
    // DRAM, so the inner loop is built to keep every load retiring:
    //  - one FP32 copy of A is converted once per input row and reused by
    //    every weight row (the old kernel re-converted A per output column);
    //  - two weight rows are interleaved into four independent FMA chains so
    //    the 5-cycle vfmaddps latency cannot stall the weight stream;
    //  - the FP8 -> BF16 bit trick stays unchanged: decoded values are
    //    pre-scaled by 2^-120 and magicScale compensates once per row/block.
#ifdef __AVX2__
    static inline void DecodeFP8E4M3x16_AVX2(const uint8_t *src, __m256 &lo, __m256 &hi) {
        const __m128i signMask = _mm_set1_epi8((char)0x80);
        const __m128i mantMask = _mm_set1_epi8(0x7F);
        const __m128i zero = _mm_setzero_si128();
        const __m128i bytes = _mm_loadu_si128((const __m128i*)src);
        const __m128i signBytes = _mm_and_si128(bytes, signMask);
        const __m128i mantBytes = _mm_and_si128(bytes, mantMask);
        const __m128i signLo = _mm_unpacklo_epi8(signBytes, zero);
        const __m128i mantLo = _mm_unpacklo_epi8(mantBytes, zero);
        const __m128i bf16Lo = _mm_or_si128(_mm_slli_epi16(signLo, 8), _mm_slli_epi16(mantLo, 4));
        const __m128i signHi = _mm_unpackhi_epi8(signBytes, zero);
        const __m128i mantHi = _mm_unpackhi_epi8(mantBytes, zero);
        const __m128i bf16Hi = _mm_or_si128(_mm_slli_epi16(signHi, 8), _mm_slli_epi16(mantHi, 4));
        lo = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16Lo), 16));
        hi = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16Hi), 16));
    }

    static inline float HorizontalSum8_AVX2(__m256 v) {
        const __m128 lo = _mm256_castps256_ps128(v);
        const __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    }

    static inline float FP8E4M3ScalarToFloat_AVX2(uint8_t fp8Val) {
        const uint16_t bf16Val = (uint16_t)(((uint16_t)(fp8Val & 0x80) << 8) |
                                            ((uint16_t)(fp8Val & 0x7F) << 4));
        const uint32_t bits = (uint32_t)bf16Val << 16;
        float value;
        memcpy(&value, &bits, sizeof(float));
        return value;
    }

    static float *GetFP8GemmAFloatScratch_AVX2(int m) {
        static thread_local std::vector<float> scratch;
        if ((int)scratch.size() < m) {
            scratch.resize(m);
        }
        return scratch.data();
    }

    static inline void ConvertBFloat16ToFloat32_AVX2(const uint16_t *src, float *dst, int len) {
        int i = 0;
        for (; i + 15 < len; i += 16) {
            const __m256i packed = _mm256_loadu_si256((const __m256i*)(src + i));
            const __m128i lo = _mm256_castsi256_si128(packed);
            const __m128i hi = _mm256_extracti128_si256(packed, 1);
            _mm256_storeu_ps(dst + i,
                _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16)));
            _mm256_storeu_ps(dst + i + 8,
                _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16)));
        }
        for (; i < len; i++) {
            const uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(dst + i, &bits, sizeof(float));
        }
    }
#endif

    bool LinearBFloat16_FP8E4M3BLOCK128_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX2__
        const size_t perRow = GetDataBytes(DataType::FP8_E4M3_BLOCK_128, 1, m);
        const float magicScale = (float)pow(2, 120);
        const int blockM = 128;
        const int numBlocks = (m + blockM - 1) / blockM;
        const size_t blockStride = blockM + sizeof(float);

        for (int i = 0; i < n; i++) {
            float *a = GetFP8GemmAFloatScratch_AVX2(m);
            ConvertBFloat16ToFloat32_AVX2(inputData + (size_t)i * m, a, m);
            float *floatC = outputData + (size_t)i * k;

            for (int j = st; j < end; j += 2) {
                const bool hasSecond = (j + 1 < end);
                const uint8_t *row0 = weightData + (size_t)j * perRow;
                const uint8_t *row1 = hasSecond ? (row0 + perRow) : row0;
                __m256 total0 = _mm256_setzero_ps();
                __m256 total1 = _mm256_setzero_ps();
                float tail0 = 0.0f;
                float tail1 = 0.0f;

                for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
                    const int base = blockIdx * blockM;
                    const int blockEnd = std::min(blockM, m - base);
                    const uint8_t *fp80 = row0 + blockIdx * blockStride;
                    const uint8_t *fp81 = row1 + blockIdx * blockStride;
                    __m256 s0 = _mm256_setzero_ps();
                    __m256 s1 = _mm256_setzero_ps();
                    __m256 r0 = _mm256_setzero_ps();
                    __m256 r1 = _mm256_setzero_ps();

                    int p = 0;
                    for (; p + 15 < blockEnd; p += 16) {
                        const __m256 a0 = _mm256_loadu_ps(a + base + p);
                        const __m256 a1 = _mm256_loadu_ps(a + base + p + 8);
                        __m256 w0lo, w0hi, w1lo, w1hi;
                        DecodeFP8E4M3x16_AVX2(fp80 + p, w0lo, w0hi);
                        s0 = _mm256_fmadd_ps(a0, w0lo, s0);
                        s1 = _mm256_fmadd_ps(a1, w0hi, s1);
                        DecodeFP8E4M3x16_AVX2(fp81 + p, w1lo, w1hi);
                        r0 = _mm256_fmadd_ps(a0, w1lo, r0);
                        r1 = _mm256_fmadd_ps(a1, w1hi, r1);
                    }

                    // Scalar tail also carries the block scale: the old
                    // kernel dropped these products below 2^-120.
                    float blockTail0 = 0.0f;
                    float blockTail1 = 0.0f;
                    for (; p < blockEnd; p++) {
                        const float av = a[base + p];
                        blockTail0 += av * FP8E4M3ScalarToFloat_AVX2(fp80[p]);
                        blockTail1 += av * FP8E4M3ScalarToFloat_AVX2(fp81[p]);
                    }

                    float scale0, scale1;
                    memcpy(&scale0, fp80 + blockM, sizeof(float));
                    memcpy(&scale1, fp81 + blockM, sizeof(float));
                    total0 = _mm256_fmadd_ps(_mm256_add_ps(s0, s1), _mm256_set1_ps(scale0), total0);
                    total1 = _mm256_fmadd_ps(_mm256_add_ps(r0, r1), _mm256_set1_ps(scale1), total1);
                    tail0 += blockTail0 * scale0;
                    tail1 += blockTail1 * scale1;
                }

                floatC[j] = (HorizontalSum8_AVX2(total0) + tail0) * magicScale;
                if (hasSecond) {
                    floatC[j + 1] = (HorizontalSum8_AVX2(total1) + tail1) * magicScale;
                }
            }
        }
        return true;
#else
        return false;
#endif
    }

    bool LinearBFloat16_FP8E4M3PERCHANNEL_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX2__
        const size_t perRow = GetDataBytes(DataType::FP8_E4M3_PERCHANNEL, 1, m);
        const float magicScale = (float)pow(2, 120);

        for (int i = 0; i < n; i++) {
            float *a = GetFP8GemmAFloatScratch_AVX2(m);
            ConvertBFloat16ToFloat32_AVX2(inputData + (size_t)i * m, a, m);
            float *floatC = outputData + (size_t)i * k;

            for (int j = st; j < end; j += 2) {
                const bool hasSecond = (j + 1 < end);
                const uint8_t *row0 = weightData + (size_t)j * perRow;
                const uint8_t *row1 = hasSecond ? (row0 + perRow) : row0;
                __m256 s0 = _mm256_setzero_ps();
                __m256 s1 = _mm256_setzero_ps();
                __m256 r0 = _mm256_setzero_ps();
                __m256 r1 = _mm256_setzero_ps();
                float tail0 = 0.0f;
                float tail1 = 0.0f;

                int p = 0;
                for (; p + 15 < m; p += 16) {
                    const __m256 a0 = _mm256_loadu_ps(a + p);
                    const __m256 a1 = _mm256_loadu_ps(a + p + 8);
                    __m256 w0lo, w0hi, w1lo, w1hi;
                    DecodeFP8E4M3x16_AVX2(row0 + p, w0lo, w0hi);
                    s0 = _mm256_fmadd_ps(a0, w0lo, s0);
                    s1 = _mm256_fmadd_ps(a1, w0hi, s1);
                    DecodeFP8E4M3x16_AVX2(row1 + p, w1lo, w1hi);
                    r0 = _mm256_fmadd_ps(a0, w1lo, r0);
                    r1 = _mm256_fmadd_ps(a1, w1hi, r1);
                }
                for (; p < m; p++) {
                    const float av = a[p];
                    tail0 += av * FP8E4M3ScalarToFloat_AVX2(row0[p]);
                    tail1 += av * FP8E4M3ScalarToFloat_AVX2(row1[p]);
                }

                float scale0, scale1;
                memcpy(&scale0, row0 + m, sizeof(float));
                memcpy(&scale1, row1 + m, sizeof(float));
                floatC[j] = (HorizontalSum8_AVX2(_mm256_add_ps(s0, s1)) + tail0) * scale0 * magicScale;
                if (hasSecond) {
                    floatC[j + 1] = (HorizontalSum8_AVX2(_mm256_add_ps(r0, r1)) + tail1) * scale1 * magicScale;
                }
            }
        }
        return true;
#else
        return false;
#endif
    }

#ifdef __AVX2__
    void print_m256i_epi16_v2(const char* name, __m256i vec) {
        alignas(32) int16_t values[16];
        _mm256_store_si256((__m256i*)values, vec);
        
        printf("%s: ", name);
        for (int i = 0; i < 16; i++) {
            printf("%d ", values[i]);
        }
        printf("\n");
    }
    void print_m256(const char* name, __m256 vec) {
        alignas(32) float result[8];
        _mm256_store_ps(result, vec);
        
        printf("%s: [", name);
        for (int i = 0; i < 8; i++) {
            printf("%.6f", result[i]);
            if (i < 7) printf(", ");
        }
        printf("]\n");
    }
    void print_m128i(const char* name, __m128i vec) {
        alignas(32) uint16_t result[8];
        _mm_storeu_si128((__m128i*)result, vec);
        
        printf("%s: [", name);
        for (int i = 0; i < 8; i++) {
            printf("%d", (int)result[i]);
            if (i < 7) printf(", ");
        }
        printf("]\n");
    }

#ifdef _MSC_VER
    __forceinline __m128i fp32_to_bf16_vec(__m256 float_vals) {
#else
    __attribute__((always_inline)) inline __m128i fp32_to_bf16_vec(__m256 float_vals) {
#endif
        __m256i shifted = _mm256_srli_epi32(_mm256_castps_si256(float_vals), 16);
        __m128i lo = _mm256_castsi256_si128(shifted);
        __m128i hi = _mm256_extracti128_si256(shifted, 1);
        return _mm_packus_epi32(lo, hi);
    }
#endif

    bool AWQ4BIT128_TO_BFloat16_AVX2_Kernel(uint8_t *awqData, uint16_t *bf16Data, int m, int st, int end, int ldb) {
#ifdef __AVX2__
        // 参数检查
        if (awqData == nullptr || bf16Data == nullptr) {
            return false;
        }
        if (st < 0 || end <= st || m <= 0) {
            return false;
        }
        if (m % 128 != 0) {
            // AWQ要求m必须是128的倍数
            return false;
        }
        
        const int block_size = 128;
        const int blocks_per_row = m / block_size;
        const int block_bytes = 64 + 1 + 4;
        
        // 用于提取低4位和高4位的掩码
        const __m256i low_mask = _mm256_set1_epi8(0x0F);
        
        // 转换awq4bit到bf16，处理行[st:end]
        for (int j = st; j < end; j++) {
            // 获取当前行的AWQ数据指针
            uint8_t *awqB = awqData + j * ldb;
            // 获取当前行的BF16输出指针
            uint16_t *bf16B_row = bf16Data + (j - st) * m;
            
            // 遍历每个block
            for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
                // 计算当前block的起始位置
                uint8_t *block_start = awqB + block_idx * block_bytes;
                
                // 解析当前block的组成部分
                uint8_t *packedWeights = block_start;                    // 64字节，存储128个uint4
                uint8_t zero = *(block_start + 64);                      // 1字节zero
                float scale = *(float*)(block_start + 64 + 1);          // 4字节scale
                
                // 准备zero和scale的向量
                __m256 scale_vec = _mm256_set1_ps(scale);
                __m256 zero_vec = _mm256_set1_ps((float)zero);
                
                // 处理当前block中的128个元素，每次处理32个（16字节包含32个int4）
                for (int i = 0; i < 64; i += 16) {  // 64字节，每次处理16字节
                    // 加载16字节（包含32个int4值）
                    __m128i packed_128 = _mm_loadu_si128((__m128i*)(packedWeights + i));
                    
                    // 扩展到256位以便处理
                    __m256i packed = _mm256_cvtepu8_epi16(packed_128);
                    
                    // 提取低4位（偶数索引的int4值）
                    __m256i low_nibbles = _mm256_and_si256(packed, _mm256_set1_epi16(0x0F));
                    // 提取高4位（奇数索引的int4值）
                    __m256i high_nibbles = _mm256_srli_epi16(packed, 4);
                    
                    // 将16位整数转换为32位整数，准备转换为float
                    // 处理低4位的前8个
                    __m256i low_lo_32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(low_nibbles));
                    // 处理低4位的后8个
                    __m256i low_hi_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(low_nibbles, 1));
                    // 处理高4位的前8个
                    __m256i high_lo_32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(high_nibbles));
                    // 处理高4位的后8个
                    __m256i high_hi_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(high_nibbles, 1));
                    
                    // 转换为float
                    __m256 low_lo_f = _mm256_cvtepi32_ps(low_lo_32);
                    __m256 low_hi_f = _mm256_cvtepi32_ps(low_hi_32);
                    __m256 high_lo_f = _mm256_cvtepi32_ps(high_lo_32);
                    __m256 high_hi_f = _mm256_cvtepi32_ps(high_hi_32);
                    
                    // 反量化：(w - zero) * scale
                    low_lo_f = _mm256_mul_ps(_mm256_sub_ps(low_lo_f, zero_vec), scale_vec);
                    low_hi_f = _mm256_mul_ps(_mm256_sub_ps(low_hi_f, zero_vec), scale_vec);
                    high_lo_f = _mm256_mul_ps(_mm256_sub_ps(high_lo_f, zero_vec), scale_vec);
                    high_hi_f = _mm256_mul_ps(_mm256_sub_ps(high_hi_f, zero_vec), scale_vec);
                    
                    // 转换为bf16
                    __m128i bf16_low_lo = fp32_to_bf16_vec(low_lo_f);
                    __m128i bf16_low_hi = fp32_to_bf16_vec(low_hi_f);
                    __m128i bf16_high_lo = fp32_to_bf16_vec(high_lo_f);
                    __m128i bf16_high_hi = fp32_to_bf16_vec(high_hi_f);
                    
                    // 计算输出位置
                    int out_idx = block_idx * block_size + (i * 2);
                    
                    // 交错存储，恢复原始顺序
                    // 需要将低4位和高4位的结果交错存储
                    __m128i tmp0, tmp1, tmp2, tmp3;
                    
                    // 交错低4位和高4位的前16个元素
                    tmp0 = _mm_unpacklo_epi16(bf16_low_lo, bf16_high_lo);
                    tmp1 = _mm_unpackhi_epi16(bf16_low_lo, bf16_high_lo);
                    
                    // 交错低4位和高4位的后16个元素
                    tmp2 = _mm_unpacklo_epi16(bf16_low_hi, bf16_high_hi);
                    tmp3 = _mm_unpackhi_epi16(bf16_low_hi, bf16_high_hi);
                    
                    // 存储32个bf16值
                    _mm_storeu_si128((__m128i*)(bf16B_row + out_idx), tmp0);
                    _mm_storeu_si128((__m128i*)(bf16B_row + out_idx + 8), tmp1);
                    _mm_storeu_si128((__m128i*)(bf16B_row + out_idx + 16), tmp2);
                    _mm_storeu_si128((__m128i*)(bf16B_row + out_idx + 24), tmp3);
                }
            }
        }
        
        return true;
#else
        return false;
#endif
    }
    
    bool LinearBFloat16_AWQ4BIT128_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX2__
        size_t perRow = GetDataBytes(DataType::AWQ_4BIT_128, 1, m);
        
        for (int i = 0; i < n; i++) {
            uint16_t *bf16A = inputData + i * m;
            float *floatC = outputData + i * k;
            int j = st;
            // 用于提取低4位和高4位的掩码
            const __m256i low_mask = _mm256_set1_epi8(0x0F);
            
            for (; j < end; j++) {
                float now = 0.0f;
                __m256 last_sum = _mm256_setzero_ps(); // Accumulator for 8 parallel sums
                // 获取当前行的起始位置
                uint8_t *rowData = (uint8_t*)weightData + j * perRow;
                
                // 计算需要多少个block（每个block有128个FP8 + 1个float scale）
                const int block_size = 128;
                const int blocks_per_row = m / block_size;
                const int block_bytes = 64 + 1 + 4;
                
                for (int blockIdx = 0; blockIdx < blocks_per_row; blockIdx++) {
                    // 计算当前block的起始位置
                    uint8_t *block_start = rowData + blockIdx * block_bytes;
                    
                    // 解析当前block的组成部分
                    uint8_t *packedWeights = block_start;                    // 64字节，存储128个uint4
                    uint8_t zero = *(block_start + 64);                      // 1字节zero
                    float scale = *(float*)(block_start + 64 + 1);          // 4字节scale
                    
                    // 准备zero和scale的向量
                    __m256 scale_vec = _mm256_set1_ps(scale);
                    __m256 zero_vec = _mm256_set1_ps((float)zero);
                    
                    // 计算当前block处理的元素范围
                    int blockStart = blockIdx * block_size;
                    int blockEnd = std::min(blockStart + block_size, m);
                    
                    __m256 v_sum = _mm256_setzero_ps(); // Accumulator for 8 parallel sums
                    
                    // 处理当前block内的数据
                    int l = blockStart;
                    for (; l + 15 < blockEnd; l += 16) {
                        // 1. Load 16 BF16 inputs and convert to float
                        __m256i v_input_bf16 = _mm256_loadu_si256((__m256i const*)(bf16A + l));
                        
                        // Convert BF16 to float32 (shift left by 16 bits)
                        __m128i v_input_low = _mm256_extracti128_si256(v_input_bf16, 0);
                        __m128i v_input_high = _mm256_extracti128_si256(v_input_bf16, 1);
                        
                        // Process low 8 BF16 values
                        __m256i v_input_low_32 = _mm256_cvtepu16_epi32(v_input_low);
                        __m256i v_input_low_shifted = _mm256_slli_epi32(v_input_low_32, 16);
                        __m256 v_input_float_low = _mm256_castsi256_ps(v_input_low_shifted);
                        
                        // Process high 8 BF16 values
                        __m256i v_input_high_32 = _mm256_cvtepu16_epi32(v_input_high);
                        __m256i v_input_high_shifted = _mm256_slli_epi32(v_input_high_32, 16);
                        __m256 v_input_float_high = _mm256_castsi256_ps(v_input_high_shifted);
                        
                        // 2. Load and dequantize AWQ4BIT weights
                        // 加载8字节（包含16个int4值）
                        __m128i packed_8bytes = _mm_loadl_epi64((__m128i*)(packedWeights + (l - blockStart) / 2));
                        // 扩展到128位
                        __m128i packed_128 = _mm_cvtepu8_epi16(packed_8bytes);
                        // 提取低4位（偶数索引的int4值）
                        __m128i low_nibbles = _mm_and_si128(packed_128, _mm_set1_epi16(0x0F));
                        // 提取高4位（奇数索引的int4值）  
                        __m128i high_nibbles = _mm_srli_epi16(packed_128, 4);
                        // 转换为32位整数
                        __m256i weight_32_low = _mm256_cvtepu16_epi32(low_nibbles);
                        __m256i weight_32_high = _mm256_cvtepu16_epi32(high_nibbles);
                        // 转换为float
                        __m256 weight_float_tmp_low = _mm256_cvtepi32_ps(weight_32_low);
                        __m256 weight_float_tmp_high = _mm256_cvtepi32_ps(weight_32_high);
                        // 反量化：(w - zero) * scale, scale放到后面乘
                        weight_float_tmp_low = _mm256_sub_ps(weight_float_tmp_low, zero_vec);
                        weight_float_tmp_high = _mm256_sub_ps(weight_float_tmp_high, zero_vec);
                        // 交错低4位和高4位，恢复原始顺序
                        // 需要将weight_float_tmp_low和weight_float_tmp_high交错组合
                        __m256 v_weight_float_low, v_weight_float_high;
                        // 使用unpack操作交错组合
                        __m256 tmp0 = _mm256_unpacklo_ps(weight_float_tmp_low, weight_float_tmp_high);
                        __m256 tmp1 = _mm256_unpackhi_ps(weight_float_tmp_low, weight_float_tmp_high);
                        // 进一步排列以得到正确的顺序
                        v_weight_float_low = _mm256_permute2f128_ps(tmp0, tmp1, 0x20);
                        v_weight_float_high = _mm256_permute2f128_ps(tmp0, tmp1, 0x31);
                        
                        // 3. Compute dot product: multiply and accumulate
                        __m256 v_mul_low = _mm256_mul_ps(v_input_float_low, v_weight_float_low);
                        __m256 v_mul_high = _mm256_mul_ps(v_input_float_high, v_weight_float_high);
                        
                        v_sum = _mm256_add_ps(v_sum, v_mul_low);
                        v_sum = _mm256_add_ps(v_sum, v_mul_high);
                    }
                    
                    last_sum = _mm256_fmadd_ps(v_sum, scale_vec, last_sum);
                }
                
                // Horizontal sum of last_sum
                __m128 sum_low = _mm256_extractf128_ps(last_sum, 0);
                __m128 sum_high = _mm256_extractf128_ps(last_sum, 1);
                __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                
                now += _mm_cvtss_f32(sum_128);
                floatC[j] = now;
            }
        }
        return true;
#else
        return false;
#endif
    }

    bool LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX2_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX2__
        size_t lda = GetDataBytes(DataType::INF_INT8_PERCHANNEL, 1, m);
        size_t ldb = GetDataBytes(DataType::INT8_PERCHANNEL, 1, m);
        size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
        
        for (int i = 0; i < n; i++) {
            // A矩阵的第i行，InfInt8PerChannel格式
            uint8_t *infInt8A = (uint8_t*)inputData + i * lda;
            int8_t *quantizedA = (int8_t*)infInt8A;
            float scaleA = *(float*)(infInt8A + m);
            int sumA = *(int*)(infInt8A + m + sizeof(float));
            
            float *floatC = (float*)((uint8_t*)outputData + i * ldc);
            
            for (int j = st; j < end; j++) {
                // B矩阵的第j行，INT8_PERCHANNEL格式
                uint8_t *int8B = (uint8_t*)weightData + j * ldb;
                float minB = *(float*)(int8B + m);
                float scaleB = *(float*)(int8B + m + sizeof(float));
                
                int sum = 0;
                int i = 0;

                __m256i acc = _mm256_setzero_si256();
                const __m256i ones = _mm256_set1_epi16(1);

                for (; i + 31 < m; i += 32) {
                    __m256i bx = _mm256_loadu_si256((const __m256i *) (int8B + i));
                    __m256i by = _mm256_loadu_si256((const __m256i *) (quantizedA + i));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(bx, by), ones));
                }
                sum = I32sum(acc);
                for (; i < m; i++) {
                    sum += quantizedA[i] * int8B[i];
                }

                floatC[j] = sum * scaleA * scaleB + minB * scaleA * sumA;
            }
        }

        AddBiasAVX2(outputData, biasData, n, k, st, end);
        return true;
#else
        return false;
#endif
    }

    bool LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX2_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX2__
        size_t lda = GetDataBytes(DataType::INF_INT8_PERCHANNEL, 1, m);
        size_t ldb = GetDataBytes(DataType::INT4_PERCHANNEL, 1, m);
        size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
        
        for (int i = 0; i < n; i++) {
            // A矩阵的第i行，InfInt8PerChannel格式
            uint8_t *infInt8A = (uint8_t*)inputData + i * lda;
            int8_t *quantizedA = (int8_t*)infInt8A;
            float scaleA = *(float*)(infInt8A + m);
            int sumA = *(int*)(infInt8A + m + sizeof(float));
            
            float *floatC = (float*)((uint8_t*)outputData + i * ldc);
            
            for (int j = st; j < end; j++) {
                // B矩阵的第j行，INT4_PERCHANNEL格式
                uint8_t *int4B = (uint8_t*)weightData + j * ldb;
                float minB = *(float*)(int4B + (m + 1) / 2);
                float scaleB = *(float*)(int4B + (m + 1) / 2 + sizeof(float));
                
                int sum = 0;
                int i = 0;

                __m256i acc = _mm256_setzero_si256();
                const __m256i lowMask = _mm256_set1_epi8(0xf);
                const __m256i ones = _mm256_set1_epi16(1);
                for (; i + 31 < m; i += 32) {
                    __m128i orix = _mm_loadu_si128((const __m128i *) (int4B + i / 2));
                    __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                    __m256i bx = _mm256_and_si256(lowMask, bytex);
                    __m256i by = _mm256_loadu_si256((const __m256i *) (quantizedA + i));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(bx, by), ones));
                }
                sum = I32sum(acc);

                // 一次处理两个int4值（假设m是偶数）
                for (; i < m; i += 2) {
                    uint8_t packedValue = int4B[i / 2];
                    // 提取高4位和低4位
                    uint8_t int4Value0 = (packedValue >> 4) & 0x0F;  // 第一个int4
                    uint8_t int4Value1 = packedValue & 0x0F;         // 第二个int4
                    
                    // 同时计算两个乘积并累加
                    sum += quantizedA[i] * int4Value0 + quantizedA[i + 1] * int4Value1;
                }

                floatC[j] = sum * scaleA * scaleB + minB * scaleA * sumA;
            }
        }

        AddBiasAVX2(outputData, biasData, n, k, st, end);
        return true;
#else
        return false;
#endif
    }

    bool LinearINT8GROUP128_INT4GROUP128_AVX2_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX2__
        size_t lda = GetDataBytes(DataType::INF_INT8_GROUP128, 1, m);
        size_t ldb = GetDataBytes(DataType::INT4_GROUP128, 1, m);
        size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
        
        int groupCnt = 128;
        int groups = m / 128;
        size_t astride = groupCnt + sizeof(float) + sizeof(int);
        size_t bstride = (groupCnt / 2) + sizeof(float) * 2;

        for (int i = 0; i < n; i++) {
            // A矩阵的第i行，InfInt8PerChannel格式
            float *floatC = (float*)((uint8_t*)outputData + i * ldc);
            
            for (int j = st; j < end; j++) {
                float fsum = 0.0f;
                for (int g = 0; g < groups; g++) {
                    uint8_t *infInt8A = (uint8_t*)inputData + i * lda + g * astride;
                    int8_t *quantizedA = (int8_t*)infInt8A;
                    float scaleA = *(float*)(infInt8A + groupCnt);
                    int sumA = *(int*)(infInt8A + groupCnt + sizeof(float));

                    uint8_t *int4B = (uint8_t*)weightData + j * ldb + g * bstride;
                    float minB = *(float*)(int4B + (groupCnt + 1) / 2);
                    float scaleB = *(float*)(int4B + (groupCnt + 1) / 2 + sizeof(float));

                    int sum = 0;
                    int i = 0;
                    __m256i acc = _mm256_setzero_si256();
                    const __m256i lowMask = _mm256_set1_epi8(0xf);
                    const __m256i ones = _mm256_set1_epi16(1);
                    for (; i + 31 < groupCnt; i += 32) {
                        __m128i orix = _mm_loadu_si128((const __m128i *) (int4B + i / 2));
                        __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                        __m256i bx = _mm256_and_si256(lowMask, bytex);
                        __m256i by = _mm256_loadu_si256((const __m256i *) (quantizedA + i));
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(bx, by), ones));
                    }
                    sum = I32sum(acc);
                    fsum += sum * scaleA * scaleB + minB * scaleA * sumA;
                }

                floatC[j] = fsum;
            }
        }

        AddBiasAVX2(outputData, biasData, n, k, st, end);
        return true;
#else
        return false;
#endif
    }

    template <int BROW, int AROW>
    void mul_mat_f16_f32_direct_avx2(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const float* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#if defined(__AVX2__)
        constexpr int SIMD_WIDTH = 8;  // AVX2 一次处理 8 个 float
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;

        // 累加器
        __m256 acc[AROW * BROW];
        
        // 初始化
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm256_setzero_ps();
        }
        
        // 主循环
        for (int i = 0; i < nb; ++i) {
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                
                // 从 f16 转换到 f32 (F16C)
                __m128i a_f16 = _mm_loadu_si128((const __m128i*)(a_row + i * SIMD_WIDTH));
                __m256 a_vec = _mm256_cvtph_ps(a_f16);
                
                for (int iy = 0; iy < BROW; ++iy) {
                    const float* b_row = (const float*)((const char*)B + iy * stride_b);
                    __m256 b_vec = _mm256_loadu_ps(b_row + i * SIMD_WIDTH);
                    
                    int acc_idx = ix * BROW + iy;
                    acc[acc_idx] = _mm256_fmadd_ps(a_vec, b_vec, acc[acc_idx]);
                }
            }
        }
        
        // 水平求和并存储（剩余元素在归约后统一累加，避免重复计入）
        for (int ix = 0; ix < AROW; ++ix) {
            for (int iy = 0; iy < BROW; ++iy) {
                int acc_idx = ix * BROW + iy;
                
                // AVX2 水平求和
                __m256 temp = acc[acc_idx];
                __m128 low = _mm256_extractf128_ps(temp, 0);
                __m128 high = _mm256_extractf128_ps(temp, 1);
                __m128 sum128 = _mm_add_ps(low, high);
                
                // 继续求和 128 位
                __m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
                __m128 sums = _mm_add_ps(sum128, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                
                float result = _mm_cvtss_f32(sums);
                
                // 处理剩余元素的结果
                if (remainder > 0) {
                    for (int j = 0; j < remainder; ++j) {
                        const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                        const float* b_row = (const float*)((const char*)B + iy * stride_b);
                        
                        __m128i a_f16 = _mm_cvtsi32_si128(a_row[nb * SIMD_WIDTH + j]);
                        __m128 a_scalar = _mm_cvtph_ps(a_f16);
                        float a_val = _mm_cvtss_f32(a_scalar);
                        float b_val = b_row[nb * SIMD_WIDTH + j];
                        
                        result += a_val * b_val;
                    }
                }
                
                float* c_row = (float*)((char*)C + iy * stride_c);
                c_row[ix] = result;
            }
        }
#endif
    }

    template <int BROW>
    void mul_mat_f16_f32_direct_avx2_pair(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const float* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#if defined(__AVX2__)
        constexpr int SIMD_WIDTH = 8;
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;

        __m256 acc[2][BROW];
        for (int c = 0; c < 2; c++) {
            for (int r = 0; r < BROW; r++) {
                acc[c][r] = _mm256_setzero_ps();
            }
        }

        const uint16_t* a0 = A;
        const uint16_t* a1 = (const uint16_t*)((const char*)A + stride_a);

        for (int i = 0; i < nb; ++i) {
            __m256 w0 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i*)(a0 + i * SIMD_WIDTH)));
            __m256 w1 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i*)(a1 + i * SIMD_WIDTH)));
            for (int r = 0; r < BROW; ++r) {
                const float* b_row =
                    (const float*)((const char*)B + r * stride_b);
                __m256 iv = _mm256_loadu_ps(b_row + i * SIMD_WIDTH);
                acc[0][r] = _mm256_fmadd_ps(iv, w0, acc[0][r]);
                acc[1][r] = _mm256_fmadd_ps(iv, w1, acc[1][r]);
            }
        }

        float tailSum[2][BROW];
        memset(tailSum, 0, sizeof(tailSum));
        for (int j = 0; j < remainder; ++j) {
            int idx = nb * SIMD_WIDTH + j;
            float w0v = _mm_cvtss_f32(
                _mm_cvtph_ps(_mm_cvtsi32_si128(a0[idx])));
            float w1v = _mm_cvtss_f32(
                _mm_cvtph_ps(_mm_cvtsi32_si128(a1[idx])));
            for (int r = 0; r < BROW; ++r) {
                const float* b_row =
                    (const float*)((const char*)B + r * stride_b);
                float bv = b_row[idx];
                tailSum[0][r] += w0v * bv;
                tailSum[1][r] += w1v * bv;
            }
        }

        for (int c = 0; c < 2; c++) {
            for (int r = 0; r < BROW; r++) {
                float* c_row = (float*)((char*)C + r * stride_c);
                c_row[c] = Floatsum(acc[c][r]) + tailSum[c][r];
            }
        }
#endif
    }

    // 5x2 微内核沿整个 m 做点积，tile 装不进缓存时必有一个操作数被反复
    // 重读。把权重列面板控制在 ~128KB：它能和一个 5 行 fp32 输入块一起
    // 放进 Broadwell 的 256KB L2，面板每个 token 超块只从 L3 载入一次，
    // 之后每个 token 块都从 L2 重读。返回偶数（权重按对处理）。
    static int LinearWeightPanelColumns(int m) {
        constexpr size_t panelBytesTarget = 128 * 1024;
        size_t perRow = sizeof(uint16_t) * (size_t)std::max(1, m);
        size_t columns = panelBytesTarget / perRow;
        columns = std::min<size_t>(columns, 32);
        columns &= ~size_t(1);
        return (int)std::max<size_t>(columns, 2);
    }

    bool LinearFloat32Float16_AVX2_Kernel(float *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        // 2D 分块：token 按超块、输出列按 L2 面板切。面板内 token 块在
        // 外层循环：权重面板驻留 L2 被所有 token 块复用，每个 5 行输入块
        // 每个面板只读一次。DRAM 流量约为权重一遍 + 输入一遍，不再随 n
        // 放大；输入重读只发生在面板数（列分块）上且落在 L3。
        constexpr int superBlock = 128;
        const int panelCols = LinearWeightPanelColumns(m);
        for (int iSuper = 0; iSuper < n; iSuper += superBlock) {
            const int superRows = std::min(superBlock, n - iSuper);
            const float *superInput = inputData + (size_t)iSuper * m;
            const int tailBegin = superRows - superRows % 5;
            for (int jPanel = st; jPanel < end; jPanel += panelCols) {
                const int panelEnd = std::min(jPanel + panelCols, end);
                int j = jPanel;
                for (; j + 1 < panelEnd; j += 2) {
                    uint16_t *weightPair = weightData + (size_t)j * m;
                    for (int i = 0; i < tailBegin; i += 5) {
                        mul_mat_f16_f32_direct_avx2_pair<5>(m, weightPair, m * sizeof(uint16_t),
                            superInput + (size_t)i * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + i) * k + j, k * sizeof(float));
                    }
                    switch (superRows - tailBegin) {
                        case 0: break;
                        case 1: mul_mat_f16_f32_direct_avx2_pair<1>(m, weightPair, m * sizeof(uint16_t),
                            superInput + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 2: mul_mat_f16_f32_direct_avx2_pair<2>(m, weightPair, m * sizeof(uint16_t),
                            superInput + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 3: mul_mat_f16_f32_direct_avx2_pair<3>(m, weightPair, m * sizeof(uint16_t),
                            superInput + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 4: mul_mat_f16_f32_direct_avx2_pair<4>(m, weightPair, m * sizeof(uint16_t),
                            superInput + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                    }
                }
                if (j < panelEnd) {
                    uint16_t *weightRow = weightData + (size_t)j * m;
                    for (int i = 0; i < tailBegin; i += 5) {
                        mul_mat_f16_f32_direct_avx2<5, 1>(m, weightRow, m * sizeof(uint16_t),
                            superInput + (size_t)i * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + i) * k + j, k * sizeof(float));
                    }
                    switch (superRows - tailBegin) {
                        case 0: break;
                        case 1: mul_mat_f16_f32_direct_avx2<1, 1>(m, weightRow, m * sizeof(uint16_t),
                            superInput + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 2: mul_mat_f16_f32_direct_avx2<2, 1>(m, weightRow, m * sizeof(uint16_t),
                            superInput + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 3: mul_mat_f16_f32_direct_avx2<3, 1>(m, weightRow, m * sizeof(uint16_t),
                            superInput + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 4: mul_mat_f16_f32_direct_avx2<4, 1>(m, weightRow, m * sizeof(uint16_t),
                            superInput + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                    }
                }
            }
        }
        AddBiasAVX2(outputData, biasData, n, k, st, end);
        return true;
    }

    template <int BROW, int AROW>
    void mul_mat_bf16_f16_direct_avx2(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#if defined(__AVX2__)
        constexpr int SIMD_WIDTH = 8;
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;

        __m256 acc[AROW * BROW];
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm256_setzero_ps();
        }

        for (int i = 0; i < nb; ++i) {
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                __m128i a_f16 = _mm_loadu_si128((const __m128i*)(a_row + i * SIMD_WIDTH));
                __m256 a_vec = _mm256_cvtph_ps(a_f16);

                for (int iy = 0; iy < BROW; ++iy) {
                    const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                    __m128i b_bf16 = _mm_loadu_si128((const __m128i*)(b_row + i * SIMD_WIDTH));
                    __m256 b_vec = bf16_to_fp32_avx2(b_bf16);

                    int acc_idx = ix * BROW + iy;
                    acc[acc_idx] = _mm256_fmadd_ps(a_vec, b_vec, acc[acc_idx]);
                }
            }
        }

        for (int ix = 0; ix < AROW; ++ix) {
            for (int iy = 0; iy < BROW; ++iy) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                int acc_idx = ix * BROW + iy;

                float tail = 0.0f;
                for (int j = 0; j < remainder; ++j) {
                    float a_val = _mm_cvtss_f32(
                        _mm_cvtph_ps(_mm_cvtsi32_si128(a_row[nb * SIMD_WIDTH + j])));
                    uint32_t b_bits = (uint32_t)b_row[nb * SIMD_WIDTH + j] << 16;
                    float b_val;
                    memcpy(&b_val, &b_bits, sizeof(b_val));
                    tail += a_val * b_val;
                }

                __m128 low = _mm256_extractf128_ps(acc[acc_idx], 0);
                __m128 high = _mm256_extractf128_ps(acc[acc_idx], 1);
                __m128 sum128 = _mm_add_ps(low, high);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);

                float* c_row = (float*)((char*)C + iy * stride_c);
                c_row[ix] = _mm_cvtss_f32(sum128) + tail;
            }
        }
#endif
    }

    template <int BROW>
    void mul_mat_bf16_f16_direct_avx2_pair(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#if defined(__AVX2__)
        constexpr int SIMD_WIDTH = 8;
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;

        __m256 acc[2][BROW];
        for (int c = 0; c < 2; c++) {
            for (int r = 0; r < BROW; r++) {
                acc[c][r] = _mm256_setzero_ps();
            }
        }

        const uint16_t* a0 = A;
        const uint16_t* a1 = (const uint16_t*)((const char*)A + stride_a);

        for (int i = 0; i < nb; ++i) {
            __m256 w0 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i*)(a0 + i * SIMD_WIDTH)));
            __m256 w1 = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i*)(a1 + i * SIMD_WIDTH)));
            for (int r = 0; r < BROW; ++r) {
                const uint16_t* b_row =
                    (const uint16_t*)((const char*)B + r * stride_b);
                __m256 iv = bf16_to_fp32_avx2(
                    _mm_loadu_si128((const __m128i*)(b_row + i * SIMD_WIDTH)));
                acc[0][r] = _mm256_fmadd_ps(iv, w0, acc[0][r]);
                acc[1][r] = _mm256_fmadd_ps(iv, w1, acc[1][r]);
            }
        }

        float tailSum[2][BROW];
        memset(tailSum, 0, sizeof(tailSum));
        for (int j = 0; j < remainder; ++j) {
            int idx = nb * SIMD_WIDTH + j;
            float w0v = _mm_cvtss_f32(
                _mm_cvtph_ps(_mm_cvtsi32_si128(a0[idx])));
            float w1v = _mm_cvtss_f32(
                _mm_cvtph_ps(_mm_cvtsi32_si128(a1[idx])));
            for (int r = 0; r < BROW; ++r) {
                const uint16_t* b_row =
                    (const uint16_t*)((const char*)B + r * stride_b);
                uint32_t tb = (uint32_t)b_row[idx] << 16;
                float bv;
                memcpy(&bv, &tb, sizeof(bv));
                tailSum[0][r] += w0v * bv;
                tailSum[1][r] += w1v * bv;
            }
        }

        for (int c = 0; c < 2; c++) {
            for (int r = 0; r < BROW; r++) {
                float* c_row = (float*)((char*)C + r * stride_c);
                c_row[c] = Floatsum(acc[c][r]) + tailSum[c][r];
            }
        }
#endif
    }

    bool LinearBFloat16Float16_AVX2_Kernel(
        uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
        int n, int m, int k, int st, int end) {
        // 输入按超块先转成 fp32（转换只做一次，内循环只剩 fp16 权重转换
        // 和 FMA，Broadwell 上 p0/p1 端口压力显著低于在 GEMM 内循环里逐块
        // 转 bf16）。GEMM 用 2D 分块：输出列按 L2 面板切，面板内 token 块
        // 在外层循环，权重面板驻留 L2 被所有 token 块复用，每个 5 行输入块
        // 每个面板只读一次。DRAM 流量约为权重一遍 + 输入一遍，不再随 n
        // 放大；超块把 fp32 scratch 限制在固定大小。
        constexpr int superBlock = 128;
        thread_local std::vector<float> fp32Input;
        if (fp32Input.size() < (size_t)superBlock * m) {
            fp32Input.resize((size_t)superBlock * m);
        }

        const int panelCols = LinearWeightPanelColumns(m);
        for (int iSuper = 0; iSuper < n; iSuper += superBlock) {
            const int superRows = std::min(superBlock, n - iSuper);
            for (int r = 0; r < superRows; r++) {
                const uint16_t *src = inputData + (size_t)(iSuper + r) * m;
                float *dst = fp32Input.data() + (size_t)r * m;
                int l = 0;
                for (; l + 7 < m; l += 8) {
                    _mm256_storeu_ps(dst + l, bf16_to_fp32_avx2(
                        _mm_loadu_si128((const __m128i*)(src + l))));
                }
                for (; l < m; l++) {
                    uint32_t x = (uint32_t)src[l] << 16;
                    memcpy(dst + l, &x, sizeof(float));
                }
            }

            const int tailBegin = superRows - superRows % 5;
            for (int jPanel = st; jPanel < end; jPanel += panelCols) {
                const int panelEnd = std::min(jPanel + panelCols, end);
                int j = jPanel;
                for (; j + 1 < panelEnd; j += 2) {
                    uint16_t *weightPair = weightData + (size_t)j * m;
                    for (int i = 0; i < tailBegin; i += 5) {
                        mul_mat_f16_f32_direct_avx2_pair<5>(m, weightPair, m * sizeof(uint16_t),
                            fp32Input.data() + (size_t)i * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + i) * k + j, k * sizeof(float));
                    }
                    switch (superRows - tailBegin) {
                        case 0: break;
                        case 1: mul_mat_f16_f32_direct_avx2_pair<1>(m, weightPair, m * sizeof(uint16_t),
                            fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 2: mul_mat_f16_f32_direct_avx2_pair<2>(m, weightPair, m * sizeof(uint16_t),
                            fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 3: mul_mat_f16_f32_direct_avx2_pair<3>(m, weightPair, m * sizeof(uint16_t),
                            fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 4: mul_mat_f16_f32_direct_avx2_pair<4>(m, weightPair, m * sizeof(uint16_t),
                            fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                    }
                }
                if (j < panelEnd) {
                    uint16_t *weightRow = weightData + (size_t)j * m;
                    for (int i = 0; i < tailBegin; i += 5) {
                        mul_mat_f16_f32_direct_avx2<5, 1>(m, weightRow, m * sizeof(uint16_t),
                            fp32Input.data() + (size_t)i * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + i) * k + j, k * sizeof(float));
                    }
                    switch (superRows - tailBegin) {
                        case 0: break;
                        case 1: mul_mat_f16_f32_direct_avx2<1, 1>(m, weightRow, m * sizeof(uint16_t),
                            fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 2: mul_mat_f16_f32_direct_avx2<2, 1>(m, weightRow, m * sizeof(uint16_t),
                            fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 3: mul_mat_f16_f32_direct_avx2<3, 1>(m, weightRow, m * sizeof(uint16_t),
                            fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                        case 4: mul_mat_f16_f32_direct_avx2<4, 1>(m, weightRow, m * sizeof(uint16_t),
                            fp32Input.data() + (size_t)tailBegin * m, m * sizeof(float),
                            outputData + (size_t)(iSuper + tailBegin) * k + j, k * sizeof(float)); break;
                    }
                }
            }
        }
        AddBiasAVX2(outputData, biasData, n, k, st, end);
        return true;
    }

#ifdef __AVX2__
    static inline __m256 NVFP4ToFloat32_AVX2(const uint8_t *packed) {
        uint32_t raw;
        memcpy(&raw, packed, sizeof(raw));
        __m128i bytes = _mm_cvtsi32_si128(static_cast<int>(raw));
        const __m128i lowMask = _mm_set1_epi8(0x0F);
        __m128i low = _mm_and_si128(bytes, lowMask);
        __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), lowMask);
        __m128i interleaved = _mm_unpacklo_epi8(low, high);

        __m256i fp4 = _mm256_cvtepu8_epi32(interleaved);
        __m256i sign = _mm256_slli_epi32(_mm256_and_si256(fp4, _mm256_set1_epi32(0x8)), 28);
        __m256i body = _mm256_and_si256(fp4, _mm256_set1_epi32(0x7));

        __m256i exp = _mm256_slli_epi32(_mm256_add_epi32(_mm256_srli_epi32(body, 1), _mm256_set1_epi32(126)), 23);
        __m256i mant = _mm256_slli_epi32(_mm256_and_si256(body, _mm256_set1_epi32(1)), 22);
        mant = _mm256_andnot_si256(_mm256_cmpeq_epi32(body, _mm256_set1_epi32(1)), mant);

        __m256i bits = _mm256_or_si256(sign, _mm256_or_si256(exp, mant));
        bits = _mm256_andnot_si256(_mm256_cmpeq_epi32(body, _mm256_setzero_si256()), bits);
        return _mm256_castsi256_ps(bits);
    }

    static inline float NVFP4E8M0ScaleToFloatFast(uint8_t v) {
        uint32_t bits = v == 0 ? 0x00400000u : ((uint32_t)v << 23);
        float ret;
        memcpy(&ret, &bits, sizeof(ret));
        return ret;
    }

    static constexpr float NVFP4_E2M1_TABLE_AVX2[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
    };

    static inline float GetNVFP4ScaleValue_AVX2(const float *scales, const uint8_t *scaleBytes, size_t idx) {
        return scales != nullptr ? scales[idx] : NVFP4E8M0ScaleToFloatFast(scaleBytes[idx]);
    }

    template <int COLS, bool INPUT_BF16>
    static inline void LinearNVFP4Cols_AVX2(
        const void *inputBase, const uint8_t *weightRows, const float *biasData, float *output,
        int outputCol, int m, int blockK, int blockM, const float *scales, const uint8_t *scaleBytes,
        int ms, int packedM
    ) {
        const float *inputF32 = reinterpret_cast<const float*>(inputBase);
        const uint16_t *inputBF16 = reinterpret_cast<const uint16_t*>(inputBase);

        __m256 acc[COLS];
        float now[COLS];
        for (int c = 0; c < COLS; c++) {
            acc[c] = _mm256_setzero_ps();
            now[c] = biasData ? biasData[outputCol + c] : 0.0f;
        }

        for (int midx = 0; midx < ms; midx++) {
            __m256 scaleVec[COLS];
            float scaleVal[COLS];
            if (outputCol / blockK == (outputCol + COLS - 1) / blockK) {
                float v = GetNVFP4ScaleValue_AVX2(scales, scaleBytes, (size_t)(outputCol / blockK) * ms + midx);
                __m256 vv = _mm256_set1_ps(v);
                for (int c = 0; c < COLS; c++) {
                    scaleVal[c] = v;
                    scaleVec[c] = vv;
                }
            } else {
                for (int c = 0; c < COLS; c++) {
                    size_t scaleIdx = (size_t)((outputCol + c) / blockK) * ms + midx;
                    scaleVal[c] = GetNVFP4ScaleValue_AVX2(scales, scaleBytes, scaleIdx);
                    scaleVec[c] = _mm256_set1_ps(scaleVal[c]);
                }
            }

            int l = midx * blockM;
            int blockEnd = std::min(m, (midx + 1) * blockM);
            __m256 blockAcc[COLS];
            for (int c = 0; c < COLS; c++) {
                blockAcc[c] = _mm256_setzero_ps();
            }
            for (; l + 7 < blockEnd; l += 8) {
                __m256 vi;
                if constexpr (INPUT_BF16) {
                    __m128i bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(inputBF16 + l));
                    vi = bf16_to_fp32_avx2(bf16);
                } else {
                    vi = _mm256_loadu_ps(inputF32 + l);
                }
                for (int c = 0; c < COLS; c++) {
                    __m256 vw = NVFP4ToFloat32_AVX2(weightRows + (size_t)c * packedM + (l >> 1));
                    blockAcc[c] = _mm256_fmadd_ps(vi, vw, blockAcc[c]);
                }
            }
            for (int c = 0; c < COLS; c++) {
                acc[c] = _mm256_fmadd_ps(blockAcc[c], scaleVec[c], acc[c]);
            }

            for (; l < blockEnd; l++) {
                float x;
                if constexpr (INPUT_BF16) {
                    uint32_t inputBits = static_cast<uint32_t>(inputBF16[l]) << 16;
                    memcpy(&x, &inputBits, sizeof(x));
                } else {
                    x = inputF32[l];
                }
                uint8_t shift = (l & 1) ? 4 : 0;
                for (int c = 0; c < COLS; c++) {
                    uint8_t packed = weightRows[(size_t)c * packedM + (l >> 1)];
                    now[c] += scaleVal[c] * x * NVFP4_E2M1_TABLE_AVX2[(packed >> shift) & 0xF];
                }
            }
        }

        for (int c = 0; c < COLS; c++) {
            output[c] = now[c] + Floatsum(acc[c]);
        }
    }

    template <bool INPUT_BF16>
    static inline bool LinearNVFP4_AVX2_Run(
        const void *inputData, uint8_t *weightData, float *biasData, float *outputData,
        int n, int m, int k, int st, int end, int blockK, int blockM,
        const float *scales, const uint8_t *scaleBytes, int ms
    ) {
        int packedM = m >> 1;
        for (int i = 0; i < n; i++) {
            const void *input;
            if constexpr (INPUT_BF16) {
                input = reinterpret_cast<const uint16_t*>(inputData) + (size_t)i * m;
            } else {
                input = reinterpret_cast<const float*>(inputData) + (size_t)i * m;
            }

            int j = st;
            for (; j + 3 < end; j += 4) {
                LinearNVFP4Cols_AVX2<4, INPUT_BF16>(
                    input, weightData + (size_t)j * packedM, biasData, outputData + (size_t)i * k + j,
                    j, m, blockK, blockM, scales, scaleBytes, ms, packedM);
            }
            switch (end - j) {
                case 0: break;
                case 1:
                    LinearNVFP4Cols_AVX2<1, INPUT_BF16>(
                        input, weightData + (size_t)j * packedM, biasData, outputData + (size_t)i * k + j,
                        j, m, blockK, blockM, scales, scaleBytes, ms, packedM);
                    break;
                case 2:
                    LinearNVFP4Cols_AVX2<2, INPUT_BF16>(
                        input, weightData + (size_t)j * packedM, biasData, outputData + (size_t)i * k + j,
                        j, m, blockK, blockM, scales, scaleBytes, ms, packedM);
                    break;
                case 3:
                    LinearNVFP4Cols_AVX2<3, INPUT_BF16>(
                        input, weightData + (size_t)j * packedM, biasData, outputData + (size_t)i * k + j,
                        j, m, blockK, blockM, scales, scaleBytes, ms, packedM);
                    break;
            }
        }
        return true;
    }
#endif

    bool LinearFloat32NVFP4_AVX2_Kernel(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, int blockK, int blockM, float *scales,
                        int ks, int ms) {
#ifdef __AVX2__
        (void)ks;
        if (blockM % 8 != 0 || (m & 1)) {
            return false;
        }
        return LinearNVFP4_AVX2_Run<false>(inputData, weightData, biasData, outputData,
                                           n, m, k, st, end, blockK, blockM, scales, nullptr, ms);
#else
        return false;
#endif
    }

    bool LinearBFloat16NVFP4_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, int blockK, int blockM, float *scales,
                        int ks, int ms) {
#ifdef __AVX2__
        (void)ks;
        if (blockM % 8 != 0 || (m & 1)) {
            return false;
        }
        return LinearNVFP4_AVX2_Run<true>(inputData, weightData, biasData, outputData,
                                          n, m, k, st, end, blockK, blockM, scales, nullptr, ms);
#else
        return false;
#endif
    }

    bool LinearFloat32NVFP4E8M0_AVX2_Kernel(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, int blockK, int blockM, uint8_t *scaleBytes,
                        int ks, int ms) {
#ifdef __AVX2__
        (void)ks;
        if (blockM % 8 != 0 || (m & 1)) {
            return false;
        }
        return LinearNVFP4_AVX2_Run<false>(inputData, weightData, biasData, outputData,
                                           n, m, k, st, end, blockK, blockM, nullptr, scaleBytes, ms);
#else
        return false;
#endif
    }

    bool LinearBFloat16NVFP4E8M0_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, int blockK, int blockM, uint8_t *scaleBytes,
                        int ks, int ms) {
#ifdef __AVX2__
        (void)ks;
        if (blockM % 8 != 0 || (m & 1)) {
            return false;
        }
        return LinearNVFP4_AVX2_Run<true>(inputData, weightData, biasData, outputData,
                                          n, m, k, st, end, blockK, blockM, nullptr, scaleBytes, ms);
#else
        return false;
#endif
    }
}
