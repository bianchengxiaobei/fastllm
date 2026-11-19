#pragma once

// SM75 FP16 tensor-core indexer score.
//
//   score[t][j] = sum_h relu(q[t][h] . k[j]) * w[t][h]
//
// Turing has FP16 MMA but no BF16 MMA, so the shared SM80+ kernel cannot run
// and long-context prefill used to fall back to the FP32 scalar kernel. Q and
// K are scaled per row by an exact power of two (the same trick as the SM75
// sparse-attention backend), which keeps every finite value inside FP16's
// exponent range without changing the dot product. One CTA owns
// kV41IdxF16BT tokens and kV41IdxF16BJ candidates; each warp owns one token
// and the packed K tile stays in shared memory across the whole token group.

constexpr int kV41IdxF16BT = 16;
constexpr int kV41IdxF16BJ = 64;
constexpr int kV41IdxF16Dim = 128;
constexpr int kV41IdxF16Stride = kV41IdxF16Dim + 8;
constexpr int kV41IdxF16N = kV41IdxF16BJ / 8;
constexpr int kV41IdxF16Threads = kV41IdxF16BT * 32;

__device__ __forceinline__ void V41IdxF16Mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b));
#else
    (void) d; (void) a0; (void) a1; (void) b;
#endif
}

// Normalize a row's magnitude to about 2^14 and return the scale that maps the
// original values into FP16. The matching inverse is ldexpf(1, -exponent).
__device__ __forceinline__ float V41IdxF16RowScale(float maximum, int &exponent) {
    if (isfinite(maximum) && maximum > 0.0f) {
        exponent = 14 - ilogb(maximum);
        if (exponent < -120) exponent = -120;
        if (exponent > 120) exponent = 120;
        return ldexpf(1.0f, exponent);
    }
    exponent = 0;
    return 1.0f;
}

// Pack Q [rows, 128] into half and keep the inverse scale per row.
// Four warps per CTA, one row per warp, four values per lane.
template <typename T>
__global__ void __launch_bounds__(128)
V41IdxF16PackQ(const T *src, __half *dst, float *invScale, int rows) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp;
    if (row >= rows) {
        return;
    }
    const T *srcRow = src + (int64_t) row * kV41IdxF16Dim;
    float values[4];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        values[i] = V41Load<T>(srcRow, lane + i * 32);
        maximum = fmaxf(maximum, fabsf(values[i]));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    }
    int exponent = 0;
    const float scale = V41IdxF16RowScale(maximum, exponent);
    __half *dstRow = dst + (int64_t) row * kV41IdxF16Dim;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        dstRow[lane + i * 32] = __float2half_rn(values[i] * scale);
    }
    if (lane == 0) {
        invScale[row] = ldexpf(1.0f, -exponent);
    }
}

__global__ void __launch_bounds__(kV41IdxF16Threads)
V41IdxF16ScoreKernel(const __half *q, const float *qInvScale, const float *weights,
                     const __nv_bfloat16 *k, const uint8_t *k8, int kMode, int kRowBytes,
                     int seqlen, int m, int ratio, int startPos, float *out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    __shared__ __align__(16) __half ks[kV41IdxF16BJ][kV41IdxF16Stride];
    __shared__ float kInvScale[kV41IdxF16BJ];
    const int j0 = blockIdx.x * kV41IdxF16BJ;
    const int i0 = blockIdx.y * kV41IdxF16BT;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;

    const int iMax = min(seqlen - 1, i0 + kV41IdxF16BT - 1);
    const int visibleMax = ratio > 0 ? min(m, (startPos + iMax + 1) / ratio) : m;
    if (j0 >= visibleMax) {
        for (int idx = tid; idx < kV41IdxF16BT * kV41IdxF16BJ; idx += kV41IdxF16Threads) {
            const int ii = idx / kV41IdxF16BJ, jj = idx % kV41IdxF16BJ;
            if (i0 + ii < seqlen && j0 + jj < m) {
                out[((int64_t) b * seqlen + i0 + ii) * m + j0 + jj] = -INFINITY;
            }
        }
        return;
    }

    // Pack K once for the whole token group. Each warp owns four candidate
    // rows so the tile-sharing benefit stays high with 16 warps.
    for (int r = 0; r < kV41IdxF16BJ / kV41IdxF16BT; r++) {
        const int jj = warp + r * kV41IdxF16BT;
        const int j = j0 + jj;
        const uint8_t *krow8 = k8 == nullptr || j >= m ? nullptr :
            k8 + ((int64_t) b * m + j) * kRowBytes;
        float values[4];
        float maximum = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int d = lane + i * 32;
            float value = 0.0f;
            if (j < m) {
                value = krow8 != nullptr ? V41LoadKvRow(krow8, kV41IdxF16Dim, d, kMode) :
                    __bfloat162float(k[((int64_t) b * m + j) * kV41IdxF16Dim + d]);
            }
            values[i] = value;
            maximum = fmaxf(maximum, fabsf(value));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
        }
        int exponent = 0;
        const float scale = V41IdxF16RowScale(maximum, exponent);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            ks[jj][lane + i * 32] = __float2half_rn(values[i] * scale);
        }
        if (lane == 0) {
            kInvScale[jj] = ldexpf(1.0f, -exponent);
        }
    }
    __syncthreads();

    const int token = i0 + warp;
    if (token >= seqlen) {
        return;
    }
    const int64_t rowBase = ((int64_t) b * seqlen + token) * 32;
    const __half *qt = q + rowBase * kV41IdxF16Dim;
    const float *qit = qInvScale + rowBase;
    const float *wt = weights + rowBase;
    const float w0 = wt[g] * qit[g];
    const float w1 = wt[g + 8] * qit[g + 8];
    const float w2 = wt[g + 16] * qit[g + 16];
    const float w3 = wt[g + 24] * qit[g + 24];

    // One mma covers 16 heads x 8 candidates; two accumulators cover all 32
    // heads. A owns head rows, B owns candidate rows, matching the scalar
    // kernel's dot-product and relu boundaries.
    float acc[2][kV41IdxF16N][4] = {};
    for (int kk = 0; kk < kV41IdxF16Dim; kk += 8) {
        const int col = kk + t * 2;
        const uint32_t a0 = *reinterpret_cast<const uint32_t *>(qt + g * kV41IdxF16Dim + col);
        const uint32_t a1 = *reinterpret_cast<const uint32_t *>(qt + (g + 8) * kV41IdxF16Dim + col);
        const uint32_t a2 = *reinterpret_cast<const uint32_t *>(qt + (g + 16) * kV41IdxF16Dim + col);
        const uint32_t a3 = *reinterpret_cast<const uint32_t *>(qt + (g + 24) * kV41IdxF16Dim + col);
#pragma unroll
        for (int n = 0; n < kV41IdxF16N; n++) {
            const uint32_t bv = *reinterpret_cast<const uint32_t *>(&ks[n * 8 + g][col]);
            V41IdxF16Mma(acc[0][n], a0, a1, bv);
            V41IdxF16Mma(acc[1][n], a2, a3, bv);
        }
    }

    float *orow = out + ((int64_t) b * seqlen + token) * m;
#pragma unroll
    for (int n = 0; n < kV41IdxF16N; n++) {
        float s0 = fmaxf(acc[0][n][0], 0.0f) * w0 + fmaxf(acc[0][n][2], 0.0f) * w1 +
                   fmaxf(acc[1][n][0], 0.0f) * w2 + fmaxf(acc[1][n][2], 0.0f) * w3;
        float s1 = fmaxf(acc[0][n][1], 0.0f) * w0 + fmaxf(acc[0][n][3], 0.0f) * w1 +
                   fmaxf(acc[1][n][1], 0.0f) * w2 + fmaxf(acc[1][n][3], 0.0f) * w3;
        s0 += __shfl_xor_sync(0xffffffffu, s0, 4);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 4);
        s0 += __shfl_xor_sync(0xffffffffu, s0, 8);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 8);
        s0 += __shfl_xor_sync(0xffffffffu, s0, 16);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
        if (g == 0) {
            const int j = j0 + n * 8 + t * 2;
            if (j < m) {
                orow[j] = s0 * kInvScale[n * 8 + t * 2];
                if (j + 1 < m) {
                    orow[j + 1] = s1 * kInvScale[n * 8 + t * 2 + 1];
                }
            }
        }
    }
#endif
}

static bool V41Sm75F16IndexerReady() {
    static std::mutex mutex;
    static std::map<int, bool> ready;
    const int device = FastllmCudaGetDevice();
    if (device < 0) {
        return false;
    }
    {
        std::lock_guard<std::mutex> guard(mutex);
        auto it = ready.find(device);
        if (it != ready.end()) {
            return it->second;
        }
    }
    int major = 0, minor = 0;
    cudaFuncAttributes scoreAttr{}, packAttr{};
    const bool ok =
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
        major == 7 && minor == 5 &&
        cudaFuncGetAttributes(&scoreAttr, V41IdxF16ScoreKernel) == cudaSuccess &&
        cudaFuncGetAttributes(&packAttr, V41IdxF16PackQ<__nv_bfloat16>) == cudaSuccess &&
        scoreAttr.ptxVersion >= 75 && packAttr.ptxVersion >= 75;
    if (!ok) {
        cudaGetLastError();
    }
    std::lock_guard<std::mutex> guard(mutex);
    ready[device] = ok;
    return ok;
}

static bool V41TryIndexerScoreSm75F16(const Data &q, const Data &weights, const Data &k,
                                      int ratio, int startPos, Data &output) {
    if (!V41IsSm75() || V41EnvOn("FASTLLM_DSV41_LEGACY_INDEXER") ||
        V41EnvOn("FASTLLM_DSV41_DISABLE_SM75_FP16_INDEXER")) {
        return false;
    }
    // Dump/reference runs compare raw score tensors; the invisible columns of
    // a partial tile are computed here instead of being written as -inf.
    if (getenv("FASTLLM_DSV41_DUMP_DIR") != nullptr ||
        V41EnvOn("FASTLLM_DSV41_REFERENCE_MATH")) {
        return false;
    }
    if (q.dims.size() != 4 || q.dims[2] != 32 || q.dims[3] != kV41IdxF16Dim ||
        k.dims.size() != 3 || k.dims[2] != kV41IdxF16Dim ||
        weights.dataType != DataType::FLOAT32 ||
        (q.dataType != DataType::BFLOAT16 && q.dataType != DataType::FLOAT32) ||
        (k.dataType != DataType::BFLOAT16 && k.dataType != DataType::INT8) ||
        !V41OnCuda(q) || !V41OnCuda(weights) || !V41OnCuda(k) ||
        FastllmCudaGraphIsCapturing()) {
        return false;
    }
    const int bsz = q.dims[0], seqlen = q.dims[1], m = k.dims[1];
    if (bsz <= 0 || seqlen < kV41IdxF16BT || m <= 0 || ratio <= 0 || startPos < 0) {
        return false;
    }
    if (output.dataType != DataType::FLOAT32 || !V41OnCuda(output) ||
        output.dims != std::vector<int>({bsz, seqlen, m})) {
        return false;
    }
    const int64_t rows = (int64_t) bsz * seqlen * 32;
    if (weights.Count(0) != (size_t) rows || rows > INT_MAX) {
        return false;
    }
    if (!V41Sm75F16IndexerReady()) {
        return false;
    }
    static std::atomic<bool> announced(false);
    if (!announced.exchange(true)) {
        fprintf(stderr, "[Fastllm] V4.1 SM75 FP16 tensor-core indexer active.\n");
    }
    int kMode = 1, kBlock = 32;
    if (k.dataType == DataType::INT8 &&
        !V41ParseKvRowHost(kV41IdxF16Dim, k.dims[2], &kMode, &kBlock)) {
        return false;
    }
    const int kRowBytes = k.dataType == DataType::INT8 ? k.dims[2] : 0;

    const size_t qBytes = (size_t) rows * kV41IdxF16Dim * sizeof(__half);
    const size_t invBytes = (size_t) rows * sizeof(float);
    const size_t bytes = qBytes + invBytes;
    if (bytes > (size_t) 512 * 1024 * 1024) {
        return false;
    }
    void *allocation = nullptr;
    if (FastllmCudaTryMalloc(&allocation, bytes) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
        return false;
    }
    struct Scratch {
        void *ptr;
        ~Scratch() {
            if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) {
                FastllmCudaFree(ptr);
            }
        }
    } scratch{allocation};
    auto *qHalf = static_cast<__half *>(allocation);
    auto *qInvScale = reinterpret_cast<float *>(static_cast<uint8_t *>(allocation) + qBytes);
    const int packBlocks = (int) ((rows + 3) / 4);
    if (q.dataType == DataType::BFLOAT16) {
        V41IdxF16PackQ<__nv_bfloat16><<<packBlocks, 128, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(q.cudaData), qHalf, qInvScale, (int) rows);
    } else {
        V41IdxF16PackQ<float><<<packBlocks, 128, 0, cudaStreamPerThread>>>(
            static_cast<const float *>(q.cudaData), qHalf, qInvScale, (int) rows);
    }
    if (!V41CheckLaunch("IndexerPackQSm75F16")) {
        return false;
    }
    const dim3 grid((m + kV41IdxF16BJ - 1) / kV41IdxF16BJ,
                    (seqlen + kV41IdxF16BT - 1) / kV41IdxF16BT, bsz);
    V41IdxF16ScoreKernel<<<grid, kV41IdxF16Threads, 0, cudaStreamPerThread>>>(
        qHalf, qInvScale, static_cast<const float *>(weights.cudaData),
        k.dataType == DataType::BFLOAT16 ? static_cast<const __nv_bfloat16 *>(k.cudaData) : nullptr,
        k.dataType == DataType::INT8 ? static_cast<const uint8_t *>(k.cudaData) : nullptr,
        kMode, kRowBytes, seqlen, m, ratio, startPos,
        static_cast<float *>(output.cudaData));
    if (!V41CheckLaunch("IndexerScoreSm75F16")) {
        return false;
    }
    return true;
}
