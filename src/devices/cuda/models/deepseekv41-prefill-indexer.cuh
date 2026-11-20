#pragma once

// Included inside the V4.1 kernel namespace. Decode/verify keep their existing
// dispatch; long SM75 prefill reuses unpacked keys and parallelizes the heads.
// Candidate keys are unpacked in bounded tiles so the scratch footprint no
// longer grows with the compressed context: a single full-width gather crossed
// the old 32 MiB budget at ~52428 rows and silently dropped every later chunk
// onto the scalar indexer (which scales with the whole context).
template<class T>
__global__ void V41IndexerUnpackKeys(const T *keys, const uint8_t *packed,
                                    float *output, int mode, int rowBytes, int rowBase) {
    const int row = blockIdx.x, d = threadIdx.x;
    const size_t source = size_t(rowBase + row);
    output[size_t(row) * 128 + d] = packed
        ? V41LoadKvRow(packed + source * rowBytes, 128, d, mode)
        : V41Load<T>(keys, source * 128 + d);
}

__global__ void V41IndexerPrefillDots(const __nv_bfloat16 *query, const float *keys,
                                     float *dots, int candidateBase, int tile,
                                     int candidates, int first, int ratio, int startPos) {
    const int token = blockIdx.z, head = blockIdx.y * 8 + threadIdx.x / 32;
    const int visible = min(candidates, (startPos + first + token + 1) / ratio);
    const int localBase = blockIdx.x * 32;
    if (candidateBase + localBase >= visible) return;
    const int local = localBase + threadIdx.x % 32;
    const int candidate = candidateBase + local;
    __shared__ float keyTile[128][33], queryTile[8][128];
    for (int i = threadIdx.x; i < 32 * 128; i += 256) {
        const int row = i / 128, d = i % 128, at = localBase + row;
        keyTile[d][row] = at < tile && candidateBase + at < visible
                              ? keys[size_t(at) * 128 + d] : 0.f;
    }
    for (int i = threadIdx.x; i < 8 * 128; i += 256)
        queryTile[i / 128][i % 128] = __bfloat162float(
            query[(size_t(token) * 32 + blockIdx.y * 8) * 128 + i]);
    __syncthreads();
    if (local >= tile || candidate >= visible) return;
    float sum = 0.f;
    // Same 128 FMAs, in the same order, as the scalar score kernel.
    #pragma unroll
    for (int d = 0; d < 128; ++d)
        sum += queryTile[threadIdx.x / 32][d] * keyTile[d][threadIdx.x % 32];
    dots[(size_t(token) * 32 + head) * tile + local] = sum;
}

__global__ void V41IndexerPrefillReduce(const float *dots, const float *weights,
                                       float *output, int candidateBase, int tile,
                                       int candidates, int first, int ratio, int startPos) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y;
    if (local >= tile) return;
    const int candidate = candidateBase + local;
    if (candidate >= (startPos + first + token + 1) / ratio) {
        output[size_t(token) * candidates + candidate] = -INFINITY;
        return;
    }
    float sum = 0.f;
    for (int h = 0; h < 32; ++h)
        sum += fmaxf(dots[(size_t(token) * 32 + h) * tile + local], 0.f) * weights[token * 32 + h];
    output[size_t(token) * candidates + candidate] = sum;
}

static bool V41TryIndexerPrefill(const Data &query, const Data &weights,
                                 const Data &keys, int mode, int ratio, int startPos,
                                 Data &output) {
    if (query.dataType != DataType::BFLOAT16 || query.dims[0] != 1 ||
        query.dims[1] < 128 || query.dims[2] != 32 || keys.dims[0] != 1 ||
        weights.Count(0) != size_t(query.dims[1]) * 32 || ratio <= 0 || startPos < 0 ||
        int64_t(startPos) + query.dims[1] > std::numeric_limits<int>::max() ||
        FastllmCudaRuntimeArch() != 75 || FastllmCudaGraphIsCapturing()) return false;
    const int rows = query.dims[1], candidates = keys.dims[1];
    if (rows <= 0 || candidates <= 0) return false;
    // Tile the candidate axis instead of materializing every key at once.
    // Tiles keep the unpacked key slab resident across all query tokens, so the
    // amount of work is unchanged while the peak stays bounded by the budget.
    constexpr int kCandidateTile = 8192;
    constexpr int kTokenBatch = 16;
    constexpr size_t budget = 32ull * 1024 * 1024;
    const int tile = std::min(kCandidateTile, candidates);
    const int batch = std::min(kTokenBatch, rows);
    const size_t scratchBytes =
        (size_t(tile) * 128 + size_t(batch) * tile * 32) * sizeof(float);
    if (scratchBytes > budget) return false;
    void *allocation = nullptr;
    if (FastllmCudaTryMalloc(&allocation, scratchBytes) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS)
        return false;
    struct Scratch {
        void *ptr;
        ~Scratch() { if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr); }
    } scratch{allocation};
    auto *decoded = static_cast<float *>(allocation);
    auto *dots = decoded + size_t(tile) * 128;
    const bool packed = keys.dataType == DataType::INT8;
    for (int base = 0; base < candidates; base += tile) {
        const int count = std::min(tile, candidates - base);
        if (packed || keys.dataType == DataType::BFLOAT16)
            V41IndexerUnpackKeys<<<count, 128, 0, cudaStreamPerThread>>>(
                packed ? nullptr : static_cast<const __nv_bfloat16 *>(keys.cudaData),
                packed ? static_cast<const uint8_t *>(keys.cudaData) : nullptr,
                decoded, mode, keys.dims[2], base);
        else
            V41IndexerUnpackKeys<<<count, 128, 0, cudaStreamPerThread>>>(
                static_cast<const float *>(keys.cudaData), nullptr, decoded, mode, 0, base);
        FastllmCudaHybridCheck(cudaGetLastError());
        for (int first = 0; first < rows; first += batch) {
            const int n = std::min(batch, rows - first);
            V41IndexerPrefillDots<<<dim3((count + 31) / 32, 4, n), 256, 0, cudaStreamPerThread>>>(
                static_cast<const __nv_bfloat16 *>(query.cudaData) + size_t(first) * 32 * 128,
                decoded, dots, base, count, candidates, first, ratio, startPos);
            V41IndexerPrefillReduce<<<dim3((count + 255) / 256, n), 256, 0, cudaStreamPerThread>>>(
                dots, static_cast<const float *>(weights.cudaData) + size_t(first) * 32,
                static_cast<float *>(output.cudaData) + size_t(first) * candidates,
                base, count, candidates, first, ratio, startPos);
            FastllmCudaHybridCheck(cudaGetLastError());
        }
    }
    return true;
}
