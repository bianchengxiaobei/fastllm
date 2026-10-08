// INT4_GROUP expert GEMM micro-bench: SM70-AWQ vs Marlin vs FP16 dequant + cuBLAS.
//
// The three paths are selected by process-wide switches read once on first use,
// so each path needs its own process. Routed experts keep the AWQ layout because
// the batch-1 fused decode kernel needs the source tensors, which is why the
// SM70-AWQ run names the weight ".mlp.experts." like the real MoE weights do.
//
//   test_cuda_int4group_expert_gemm awq|marlin|cublas [rounds]
//
// Shapes mirror the Qwen3.5-MoE experts with moe_intermediate_size=512:
// gate/up fuses to (in=2048, out=1024) and down is (in=512, out=2048). Rows are
// the per-expert token counts, i.e. chunk_tokens * topk / 256.

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace fastllm;

extern unsigned long long FastllmCudaRealMallocCount();

namespace {

double Now() {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() * 1e-9;
}

void SetEnv(const char *key, const char *value) {
#if defined(_WIN32)
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

Data MakeInt4GroupWeight(const std::string &name, int outDim, int inDim, int groupCnt) {
    const int group = inDim / groupCnt;
    Data weight(DataType::INT4_GROUP, std::vector<int>{outDim, inDim});
    weight.name = name;
    weight.group = group;
    weight.groupCnt = groupCnt;
    weight.perChannelAxis = 0;
    weight.scales.resize((size_t)outDim * group);
    weight.mins.resize((size_t)outDim * group);
    weight.zeros.resize((size_t)outDim * group);
    weight.Allocate(true);
    uint32_t state = 12345;
    for (int y = 0; y < outDim; y++) {
        for (int g = 0; g < group; g++) {
            state = state * 1664525u + 1013904223u;
            const int zero = (int)((state >> 24) & 15);
            const float scale = 0.0015f * (1 + (int)((state >> 20) % 5));
            const size_t meta = (size_t)y * group + g;
            weight.scales[meta] = scale;
            weight.mins[meta] = -scale * (float)zero;
            weight.zeros[meta] = (uint8_t)zero;
            for (int x = g * groupCnt; x < (g + 1) * groupCnt; x++) {
                const int q = (int)((state * 7919u + (uint32_t)x * 131u) & 15);
                const size_t byteId = ((size_t)y * inDim + x) / 2;
                if ((x & 1) == 0) {
                    weight.cpuData[byteId] |= (uint8_t)(q << 4);
                } else {
                    weight.cpuData[byteId] |= (uint8_t)q;
                }
            }
        }
    }
    weight.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    return weight;
}

Data MakeHalfInput(int rows, int cols, uint32_t seed) {
    Data input(DataType::FLOAT16, std::vector<int>{rows, cols});
    input.Allocate(false);
    uint32_t state = seed;
    for (size_t i = 0; i < input.Count(0); i++) {
        state = state * 1664525u + 1013904223u;
        const float value = (float)((int)(state >> 8) - 8388608) / 8388608.0f * 0.05f;
        ((uint16_t *)input.cpuData)[i] = __half_as_ushort(__float2half_rn(value));
    }
    input.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    return input;
}

void RunCase(const char *path, const char *weightName, const char *shape,
             int inDim, int outDim, int rows, int rounds) {
    Data weight = MakeInt4GroupWeight(weightName, outDim, inDim, 128);
    Data input = MakeHalfInput(rows, inDim, 7 + (uint32_t)rows);
    Data bias, output;
    const bool marlinPickedBefore = weight.cudaData == nullptr;

    for (int i = 0; i < 3; i++) {
        Linear(input, weight, bias, output);
    }
    cudaDeviceSynchronize();

    const unsigned long long mallocBefore = FastllmCudaRealMallocCount();
    const double wallStart = Now();
    for (int i = 0; i < rounds; i++) {
        Linear(input, weight, bias, output);
    }
    const double hostMs = (Now() - wallStart) * 1000.0 / rounds;
    cudaDeviceSynchronize();
    const unsigned long long realMalloc = FastllmCudaRealMallocCount() - mallocBefore;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < rounds; i++) {
        Linear(input, weight, bias, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);
    const double deviceMs = (double)elapsed / rounds;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const double flops = 2.0 * rows * outDim * inDim;
    std::printf("[int4g-bench] path=%-6s shape=%-8s in=%4d out=%4d rows=%5d "
                "host=%8.3f ms dev=%8.3f ms tflops=%7.2f realMalloc=%llu marlin=%d\n",
                path, shape, inDim, outDim, rows, hostMs, deviceMs,
                deviceMs > 0.0 ? flops / (deviceMs * 1e-3) / 1e12 : 0.0,
                realMalloc, (int)(!marlinPickedBefore && weight.cudaData == nullptr));
    std::fflush(stdout);
}

// Per-expert activation temporaries are reallocated for every expert on every
// layer. On Windows a real cudaMalloc/cudaFree pair is not free, so measure both
// the pooled path and the raw driver call at the sizes the MoE loop uses.
void RunChurnProbe(size_t bytes, int rounds) {
    std::vector<void *> warm(8, nullptr);
    for (auto &p : warm) {
        p = FastllmCudaMalloc(bytes);
    }
    for (auto &p : warm) {
        FastllmCudaFree(p);
    }
    cudaDeviceSynchronize();

    double start = Now();
    for (int i = 0; i < rounds; i++) {
        void *p = FastllmCudaMalloc(bytes);
        FastllmCudaFree(p);
    }
    const double pooledUs = (Now() - start) * 1e6 / rounds;

    start = Now();
    for (int i = 0; i < rounds; i++) {
        void *p = nullptr;
        cudaMalloc(&p, bytes);
        cudaFree(p);
    }
    const double rawUs = (Now() - start) * 1e6 / rounds;

    std::printf("[int4g-bench] churn mb=%6.2f pooled=%8.1f us raw_cudaMalloc=%8.1f us\n",
                bytes / 1048576.0, pooledUs, rawUs);
    std::fflush(stdout);
}

// One MoE layer allocates a fresh set of activation temporaries for every expert
// it touches, and every expert gets a different token count, so the allocator sees
// a new size almost every call. Size variation is what decides whether the pool
// recycles or falls back to a real cudaMalloc, so measure both kinds of traffic.
void RunTempChurn(const char *label, size_t baseBytes, int slots, bool variable, int layers) {
    std::vector<size_t> sizes(slots);
    uint32_t state = 7;
    for (int i = 0; i < slots; i++) {
        if (!variable) {
            sizes[i] = baseBytes;
        } else {
            state = state * 1664525u + 1013904223u;
            sizes[i] = baseBytes / 2 + (state >> 8) % (baseBytes / 2 + 1);
        }
    }
    std::vector<void *> warm(slots, nullptr);
    for (int i = 0; i < slots; i++) {
        warm[i] = FastllmCudaMalloc(sizes[i]);
    }
    for (int i = 0; i < slots; i++) {
        FastllmCudaFree(warm[i]);
    }
    cudaDeviceSynchronize();

    const unsigned long long before = FastllmCudaRealMallocCount();
    const double start = Now();
    for (int layer = 0; layer < layers; layer++) {
        for (int i = 0; i < slots; i++) {
            void *p = FastllmCudaMalloc(sizes[i]);
            FastllmCudaFree(p);
        }
    }
    const double perLayerMs = (Now() - start) * 1000.0 / layers;
    const unsigned long long realMalloc = FastllmCudaRealMallocCount() - before;
    std::printf("[int4g-bench] temp-churn %-10s base=%5.2f MB slots=%d variable=%d "
                "per_layer=%8.2f ms realMalloc=%llu\n",
                label, baseBytes / 1048576.0, slots, (int)variable, perLayerMs, realMalloc);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s awq|marlin|cublas [rounds=20]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 20;
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::fprintf(stderr, "no CUDA device\n");
        return 77;
    }

    // Routed experts must not reach Marlin: their source tensors feed the fused
    // decode pointer table. Naming the weight like a real expert reproduces that.
    std::string weightName = "model.layers.0.mlp.experts.0.gate_proj.weight";
    if (path == "awq") {
        SetEnv("FASTLLM_CUDA_INT4_GROUP_PREFILL_CUBLAS", "0");
        SetEnv("FASTLLM_DISABLE_SM70_AWQ", "");
    } else if (path == "cublas") {
        SetEnv("FASTLLM_CUDA_INT4_GROUP_PREFILL_CUBLAS", "1");
        SetEnv("FASTLLM_DISABLE_SM70_AWQ", "");
    } else if (path == "marlin") {
        // Marlin is only reachable for non-routed weights, and it is skipped
        // while SM70-AWQ is available.
        SetEnv("FASTLLM_CUDA_INT4_GROUP_PREFILL_CUBLAS", "0");
        SetEnv("FASTLLM_DISABLE_SM70_AWQ", "1");
        weightName = "model.layers.0.self_attn.q_proj.weight";
    } else {
        std::fprintf(stderr, "unknown path %s\n", path.c_str());
        return 2;
    }

    static_cast<Executor *>(GetExecutor())->SetFirstDevice("cuda:0");
    std::printf("[int4g-bench] path=%s rounds=%d\n", path.c_str(), rounds);
    for (int rows : {128, 512, 1024}) {
        RunCase(path.c_str(), weightName.c_str(), "gate_up", 2048, 1024, rows, rounds);
    }
    for (int rows : {128, 512, 1024}) {
        RunCase(path.c_str(), weightName.c_str(), "down", 512, 2048, rows, rounds);
    }
    RunChurnProbe((size_t)512 * 2048 * 2, 200);
    RunChurnProbe((size_t)4096 * 2048 * 2, 200);
    // 256 experts x 2 weights x ~4 activation temporaries ~= the 1912 real
    // allocations the 16384-token server run reported per layer.
    RunTempChurn("r128-fixed", (size_t)128 * 2048 * 2, 2048, false, 3);
    RunTempChurn("r128-var", (size_t)128 * 2048 * 2, 2048, true, 3);
    RunTempChurn("r512-fixed", (size_t)512 * 2048 * 2, 2048, false, 3);
    RunTempChurn("r512-var", (size_t)512 * 2048 * 2, 2048, true, 3);
    return 0;
}
