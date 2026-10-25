#pragma once
#include "fastllm.h"

// Resident native V4.1 GGUF MoE. GPU routes and immutable weight pointers;
// indexed GEMV for <=32 rows, grouped MMQ for supported larger batches.
// Both preserve block-32 activation quantization and FP32 expert accumulation.
bool FastllmCudaMergeMOEV41GGUFResident(
    const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &activation,
    fastllm::Data &workspace, fastllm::Data &output, fastllm::Data **weights,
    int weightsBatch, const fastllm::Data &indices, const fastllm::Data &scores,
    float sharedScale, float swigluLimit);

// Internal launcher; caller validates layouts and owns all scratch tensors.
// `accumulate` keeps the previous FLOAT32 output and adds this launch's routes,
// so a chunk can be served by several disjoint expert groups.
bool FastllmCudaRunV41GGUF(
    const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &activation,
    fastllm::Data &workspace, fastllm::Data &output, const void *weightPointers,
    const int32_t *indices, const float *scores, int gateType, int downType,
    int hidden, int inter, int experts, int topk, float swigluLimit,
    bool accumulate = false);

// Host-resident (pageable) GGUF experts on the GPU: the routed experts are
// uploaded in VRAM-bounded groups and accumulated, so the card only ever holds
// part of one layer. `indices`/`scores` are host arrays; `budgetBytes` bounds
// the staging arena (0 sizes it from the current free VRAM).
bool FastllmCudaRunV41GGUFHostStream(
    const fastllm::Data &input, fastllm::Data &output,
    fastllm::Data **weights, int weightsBatch,
    const int32_t *indices, const float *scores,
    int topk, float swigluLimit, size_t budgetBytes);

// Native CUDA activation semantics (Q8_1), distinct from NUMA's Q8_K path.
size_t FastllmCudaV41GGUFGroupedWorkspaceBytes(
    int gateType, int downType, int rows, int hidden, int inter, int experts, int topk);
bool FastllmCudaV41GGUFGroupedProjection(
    bool gate, int type, const void *inputQ8, void *outputBF16,
    const void *weightPointers, const int32_t *indices, void *workspace,
    int rows, int hidden, int inter, int experts, int topk);

// Canonical packed records restored from NUMA shards, with per-layer slot offsets.
struct FastllmCudaMoeGGUFCacheView;
bool FastllmCudaMoeV41CanonicalGGUFCacheCompute(
    const fastllm::Data &input, fastllm::Data &activation,
    const FastllmCudaMoeGGUFCacheView &view, const float *scores,
    int topk, float swigluLimit, float *perExpert);
size_t FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(int hidden, int inter);
bool FastllmCudaV41GGUFSupported(int gateType, int downType);
