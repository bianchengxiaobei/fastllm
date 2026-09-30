// NVFP4_BLOCK_16 fused CPU GEMM regression + micro-benchmark.
//
// Block-16 is what CPU-only NVFP4 checkpoints land on: every 16 weights are
// stored as eight packed nibble bytes plus a block scale. `NVFP4_BLOCK_16`
// keeps that scale as FP32 (12 bytes per block); the compact
// `NVFP4_BLOCK_16_E4M3_PACKED` layout keeps a per-row global multiplier plus
// the raw FP8 E4M3 block scale (9 bytes per block).
//
//   ./nvfp4Block16GemmRegression                 correctness only
//   ./nvfp4Block16GemmRegression --bench         correctness + bandwidth
//   ./nvfp4Block16GemmRegression --bench --threads 8 --gb 4
//   ./nvfp4Block16GemmRegression --bench --layout compact
//   ./nvfp4Block16GemmRegression --bench --linear   the --moe_device cpu decode path
#include "fastllm.h"
#include "devices/cpu/computeutils.h"
#include "utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace fastllm {
    bool FastllmGemmBFloat16NVFP4Block16_AVX2(
        const void *A, long lda, const void *B, long ldb,
        void *C, long ldc, int n, int m, int k, int st, int end);
    bool FastllmGemmBFloat16NVFP4Block16E4M3Packed_AVX2(
        const void *A, long lda, const void *B, long ldb,
        void *C, long ldc, int n, int m, int k, int st, int end);
    CPUInstructInfo *GetCPUInstructInfo();
}

using namespace fastllm;

static int failures = 0;

static void Check(bool ok, const std::string &what) {
    if (!ok) {
        printf("[FAIL] %s\n", what.c_str());
        failures++;
    } else {
        printf("[ OK ] %s\n", what.c_str());
    }
}

static const float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static float Bf16ToFloat(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float ret;
    memcpy(&ret, &bits, sizeof(ret));
    return ret;
}

static uint16_t FloatToBf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

// E4M3 code -> float for codes 0..126, matching fastllm's FP8 table.
static float E4M3ToFloat(uint8_t code) {
    const int sign = (code >> 7) & 1;
    const int exp = (code >> 3) & 0xF;
    const int mant = code & 0x7;
    float value;
    if (exp == 0) {
        value = (float)mant * (1.0f / 512.0f);
    } else {
        value = (1.0f + (float)mant * (1.0f / 8.0f)) * std::ldexp(1.0f, exp - 7);
    }
    return sign ? -value : value;
}

// Checkpoints store `floatScale = e4m3(byte) * global`, so a compact row is a
// global multiplier plus the original one-byte block scales.
static const float kGlobalScale = 1.0f / 448.0f;

static size_t LegacyRowBytes(int m) {
    return (size_t)((m - 1) / 16 + 1) * 12;
}

static size_t CompactRowBytes(int m) {
    return (sizeof(float) + (size_t)((m - 1) / 16 + 1) * 9 + 3) & ~size_t(3);
}

static void BuildLegacyWeights(int k, int m, std::vector<uint8_t> &out, std::mt19937 &rng) {
    const int blocks = (m - 1) / 16 + 1;
    const size_t rowBytes = LegacyRowBytes(m);
    out.assign(rowBytes * (size_t)k, 0);
    for (int row = 0; row < k; row++) {
        uint8_t *p = out.data() + (size_t)row * rowBytes;
        for (int block = 0; block < blocks; block++) {
            for (int b = 0; b < 8; b++) {
                p[(size_t)block * 12 + b] = (uint8_t)(rng() & 0xFF);
            }
            // Only E4M3-representable scales, so the same matrix can also be
            // packed in the compact layout without changing its values.
            const float scale = E4M3ToFloat((uint8_t)(rng() % 127)) * kGlobalScale;
            memcpy(p + (size_t)block * 12 + 8, &scale, sizeof(scale));
        }
    }
}

static void BuildCompactWeights(int k, int m, const std::vector<uint8_t> &legacy,
                                std::vector<uint8_t> &out) {
    const int blocks = (m - 1) / 16 + 1;
    const size_t legacyRowBytes = LegacyRowBytes(m), rowBytes = CompactRowBytes(m);
    out.assign(rowBytes * (size_t)k, 0);
    for (int row = 0; row < k; row++) {
        const uint8_t *src = legacy.data() + (size_t)row * legacyRowBytes;
        uint8_t *dst = out.data() + (size_t)row * rowBytes;
        memcpy(dst, &kGlobalScale, sizeof(float));
        for (int block = 0; block < blocks; block++) {
            const uint8_t *srcBlock = src + (size_t)block * 12;
            uint8_t *dstBlock = dst + sizeof(float) + (size_t)block * 9;
            memcpy(dstBlock, srcBlock, 8);
            const float scale = *(const float*)(srcBlock + 8);
            uint8_t best = 0;
            float bestErr = 1e30f;
            for (int code = 0; code < 128; code++) {
                const float err = std::fabs(E4M3ToFloat((uint8_t)code) * kGlobalScale - scale);
                if (err < bestErr) {
                    bestErr = err;
                    best = (uint8_t)code;
                }
            }
            dstBlock[8] = best;
        }
    }
}

static void Reference(const uint16_t *A, long ldaElems, const uint8_t *B, size_t rowBytes,
                      float *C, float *Scale, int ldcElems, int n, int m, int st, int end,
                      bool compact) {
    const int blocks = (m - 1) / 16 + 1;
    const size_t blockBytes = compact ? 9 : 12;
    for (int i = 0; i < n; i++) {
        for (int j = st; j < end; j++) {
            const uint8_t *rowStart = B + (size_t)j * rowBytes;
            double total = 0.0, magnitude = 0.0;
            for (int block = 0; block < blocks; block++) {
                const uint8_t *bs = rowStart + (compact ? sizeof(float) : 0) +
                    (size_t)block * blockBytes;
                const int base = block * 16;
                const int elems = std::min(16, m - base);
                const float scale = compact
                    ? E4M3ToFloat(bs[8]) * (*(const float*)rowStart)
                    : *(const float*)(bs + 8);
                double now = 0.0;
                for (int o = 0; o < elems; o++) {
                    const uint8_t byte = bs[o >> 1];
                    const uint8_t code = (o & 1) ? (byte >> 4) : (byte & 0xF);
                    const double term = (double)Bf16ToFloat(A[(size_t)i * ldaElems + base + o]) *
                        (double)kE2M1[code];
                    now += term;
                    magnitude += std::fabs(term) * (double)scale;
                }
                total += now * (double)scale;
            }
            C[(size_t)i * ldcElems + j] = (float)total;
            Scale[(size_t)i * ldcElems + j] = (float)magnitude;
        }
    }
}

// ---------------------------------------------------------------------------

static void Correctness(const std::vector<std::pair<int, int>> &shapes) {
    std::mt19937 rng(20260928);
    for (auto &shape : shapes) {
        const int m = shape.first, k = shape.second;
        std::vector<uint8_t> legacy, compact;
        BuildLegacyWeights(k, m, legacy, rng);
        BuildCompactWeights(k, m, legacy, compact);
        const size_t legacyLdb = LegacyRowBytes(m), compactLdb = CompactRowBytes(m);
        for (int n : {1, 2, 3, 4, 5, 6, 8, 13}) {
            std::vector<uint16_t> input((size_t)n * m);
            for (auto &v : input) {
                v = FloatToBf16(((int)(rng() % 2001) - 1000) / 1000.0f);
            }
            std::vector<float> ref((size_t)n * k, 0.0f), got((size_t)n * k, 0.0f);
            std::vector<float> mag((size_t)n * k, 0.0f);
            Reference(input.data(), m, legacy.data(), legacyLdb,
                      ref.data(), mag.data(), k, n, m, 0, k, false);
            bool ran = FastllmGemmBFloat16NVFP4Block16_AVX2(
                input.data(), (long)m * sizeof(uint16_t),
                legacy.data(), (long)legacyLdb,
                got.data(), (long)k * sizeof(float),
                n, m, k, 0, k);
            if (!ran) {
                printf("[SKIP] AVX2 block-16 kernel unavailable in this build\n");
                return;
            }
            double maxRel = 0.0;
            bool finite = true;
            for (size_t i = 0; i < ref.size(); i++) {
                finite &= std::isfinite(got[i]);
                maxRel = std::max(maxRel, std::fabs((double)got[i] - ref[i]) /
                    std::max(1e-6, (double)mag[i]));
            }
            char buf[256];
            snprintf(buf, sizeof(buf), "AVX2 block-16 m=%d k=%d n=%d maxErr/|terms|=%.3e",
                     m, k, n, maxRel);
            Check(finite && maxRel < 2e-5, buf);

            std::fill(got.begin(), got.end(), 0.0f);
            std::fill(mag.begin(), mag.end(), 0.0f);
            Reference(input.data(), m, compact.data(), compactLdb,
                      ref.data(), mag.data(), k, n, m, 0, k, true);
            FastllmGemmBFloat16NVFP4Block16E4M3Packed_AVX2(
                input.data(), (long)m * sizeof(uint16_t),
                compact.data(), (long)compactLdb,
                got.data(), (long)k * sizeof(float),
                n, m, k, 0, k);
            double compactRel = 0.0;
            bool compactFinite = true;
            for (size_t i = 0; i < ref.size(); i++) {
                compactFinite &= std::isfinite(got[i]);
                compactRel = std::max(compactRel, std::fabs((double)got[i] - ref[i]) /
                    std::max(1e-6, (double)mag[i]));
            }
            snprintf(buf, sizeof(buf),
                     "AVX2 block-16 compact m=%d k=%d n=%d maxErr/|terms|=%.3e",
                     m, k, n, compactRel);
            Check(compactFinite && compactRel < 2e-5, buf);
        }
    }
}

// Nonzero and odd column ranges, byte strides, and partial blocks; untouched
// columns and padding are guards.
static void StridedCorrectness() {
    std::mt19937 rng(20260927);
    const int shapes[][4] = {
        {2048, 1024, 3, 1023}, {512, 2048, 1, 2047},
        {2048, 211, 7, 205}, {512, 35, 7, 22},
        {1056, 21, 1, 20}, {1024, 21, 1, 20},
        {1023, 21, 1, 20}, {2049, 21, 1, 20},
        {31, 21, 1, 20}, {1, 21, 1, 20}, {17, 9, 2, 8}
    };
    for (const auto &shape : shapes) {
        const int m = shape[0], k = shape[1], st = shape[2], end = shape[3];
        const int lda = m + 11, ldc = k + 7;
        std::vector<uint8_t> dense;
        BuildLegacyWeights(1, m, dense, rng);
        const size_t rowBytes = dense.size(), ldb = rowBytes + 13;
        std::vector<uint8_t> weights((size_t)k * ldb, 0xcd);
        for (int col = 0; col < k; col++) {
            memcpy(weights.data() + col * ldb, dense.data(), rowBytes);
        }
        for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 13, 32}) {
            std::vector<uint16_t> input((size_t)n * lda, 0x7fc1);
            for (int row = 0; row < n; row++) {
                for (int d = 0; d < m; d++) {
                    input[(size_t)row * lda + d] = FloatToBf16(
                        ((int)(rng() % 2001) - 1000) / 1000.0f);
                }
            }
            constexpr float guard = -1234567.0f;
            std::vector<float> got((size_t)n * ldc, guard);
            std::vector<float> ref(got), mag(got.size(), 0.0f);
            Reference(input.data(), lda, weights.data(), ldb,
                      ref.data(), mag.data(), ldc, n, m, st, end, false);
            if (!FastllmGemmBFloat16NVFP4Block16_AVX2(
                    input.data(), (long)lda * 2, weights.data(), (long)ldb,
                    got.data(), (long)ldc * 4, n, m, k, st, end)) return;
            bool valid = true;
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < ldc; col++) {
                    const size_t i = (size_t)row * ldc + col;
                    if (col < st || col >= end) valid &= got[i] == guard;
                    else valid &= std::isfinite(got[i]) &&
                        std::fabs((double)got[i] - ref[i]) <=
                            2e-5 * std::max(1e-6, (double)mag[i]);
                }
            }
            char description[160];
            snprintf(description, sizeof(description),
                     "AVX2 block-16 strided m=%d n=%d columns=[%d,%d) padding preserved",
                     m, n, st, end);
            Check(valid, description);
        }
    }
}

// ---------------------------------------------------------------------------

struct BenchShape {
    int rows, m, k;   // n tokens, m input dim, k output dim
    const char *name;
};

// Streams a pile of routed-expert weights through `threads` workers, exactly
// like CpuMergeMOE does: every task owns a slice of one expert's output rows.
static void Bench(const BenchShape &shape, int threads, double gigabytes, bool compact) {
    const size_t rowBytes = compact ? CompactRowBytes(shape.m) : LegacyRowBytes(shape.m);
    const size_t expertBytes = rowBytes * (size_t)shape.k;
    const int experts = std::max(1, (int)(gigabytes * 1e9 / expertBytes));

    std::mt19937 rng(7);
    std::vector<std::vector<uint8_t>> weights(experts);
    for (int e = 0; e < experts; e++) {
        std::vector<uint8_t> legacy;
        BuildLegacyWeights(shape.k, shape.m, legacy, rng);
        if (compact) {
            BuildCompactWeights(shape.k, shape.m, legacy, weights[e]);
        } else {
            weights[e] = std::move(legacy);
        }
    }
    std::vector<uint16_t> input((size_t)shape.rows * shape.m);
    for (auto &v : input) v = FloatToBf16(((int)(rng() % 2001) - 1000) / 1000.0f);

    const int rowsPerTask = 208;
    const int chunks = (shape.k + rowsPerTask - 1) / rowsPerTask;
    const int tasks = experts * chunks;
    std::vector<std::vector<float>> outputs(threads,
        std::vector<float>((size_t)shape.rows * shape.k, 0.0f));

    // The first pass only warms the pages; keep the best of the timed passes so
    // a busy machine shows the kernel's own ceiling rather than a scheduling dip.
    double best = 0.0;
    for (int pass = 0; pass < 5; pass++) {
        std::atomic<int> next(0);
        auto start = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++) {
            pool.emplace_back([&, t]() {
                while (true) {
                    int task = next.fetch_add(1, std::memory_order_relaxed);
                    if (task >= tasks) break;
                    const int e = task / chunks;
                    const int chunk = task - e * chunks;
                    const int st = chunk * rowsPerTask;
                    const int end = std::min(st + rowsPerTask, shape.k);
                    if (compact) {
                        FastllmGemmBFloat16NVFP4Block16E4M3Packed_AVX2(
                            input.data(), (long)shape.m * sizeof(uint16_t),
                            weights[e].data(), (long)rowBytes,
                            outputs[t].data(), (long)shape.k * sizeof(float),
                            shape.rows, shape.m, shape.k, st, end);
                    } else {
                        FastllmGemmBFloat16NVFP4Block16_AVX2(
                            input.data(), (long)shape.m * sizeof(uint16_t),
                            weights[e].data(), (long)rowBytes,
                            outputs[t].data(), (long)shape.k * sizeof(float),
                            shape.rows, shape.m, shape.k, st, end);
                    }
                }
            });
        }
        for (auto &th : pool) th.join();
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (pass > 0) {
            best = best == 0.0 ? seconds : std::min(best, seconds);
        }
    }
    const double bytes = (double)expertBytes * experts;
    printf("%-6s %-7s threads=%2d n=%d m=%4d k=%5d experts=%4d bytes=%6.2f GB  "
           "%8.1f ms  %6.1f GB/s  %6.2f GB/s/core\n",
           shape.name, compact ? "compact" : "inline", threads, shape.rows,
           shape.m, shape.k, experts, bytes / 1e9, best * 1000.0,
           bytes / 1e9 / best, bytes / 1e9 / best / threads);
}

// The decode path with `--moe_device cpu` does *not* go through FastllmGemm:
// the expert projections are dispatched as the Linear op, i.e.
// RunLinearFloat16NVFP4 -> MultiThreadLinearBFloat16NVFP4Op.  Same weight bytes,
// different kernel, so measure both to compare machines honestly.
static void BenchLinear(const BenchShape &shape, int threads, double gigabytes) {
    constexpr int blockK = 16, blockM = 16;
    const size_t weightBytes = GetNVFP4WeightBytes(shape.k, shape.m);
    const size_t scaleBytes = GetNVFP4ScaleBytes(shape.k, shape.m, blockK, blockM);
    const size_t expertBytes = weightBytes + scaleBytes;
    const int experts = std::max(1, (int)(gigabytes * 1e9 / expertBytes));

    std::vector<std::unique_ptr<Data> > weights(experts);
    for (int e = 0; e < experts; e++) {
        std::unique_ptr<Data> weight(new Data(DataType::NVFP4, {shape.k, shape.m}));
        weight->blockK = blockK;
        weight->blockM = blockM;
        weight->Allocate(false);
        for (size_t i = 0; i < weightBytes; i++) {
            const uint8_t low = (uint8_t)((i * 5 + e) & 0xf);
            const uint8_t high = (uint8_t)((i * 11 + e * 3 + 1) & 0xf);
            weight->cpuData[i] = low | (high << 4);
        }
        uint8_t *scales = GetNVFP4ScaleData(*weight);
        if (scales == nullptr || weightBytes == 0) {
            printf("inline  linear  NVFP4 scale storage unavailable, skipping\n");
            return;
        }
        for (size_t i = 0; i < scaleBytes; i++) {
            scales[i] = (uint8_t)(119 + ((i + e) % 7));
        }
        weights[e] = std::move(weight);
    }

    // RunLinearFloat16NVFP4 takes FLOAT16 activations; 0x3c00 is exactly 1.0f,
    // so no conversion helper is needed for a pure bandwidth measurement.
    std::vector<std::vector<uint16_t> > inputs(threads,
        std::vector<uint16_t>((size_t)shape.rows * shape.m, 0x3c00));
    std::vector<std::vector<uint16_t> > outputs(threads,
        std::vector<uint16_t>((size_t)shape.rows * shape.k, 0));
    SetThreads(threads);

    // Same shape as the Gemm bench: one task owns one whole expert, workers
    // stream the expert pile, and the single-thread split matches CpuMergeMOE.
    double best = 0.0;
    for (int pass = 0; pass < 5; pass++) {
        std::atomic<int> next(0);
        auto start = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++) {
            pool.emplace_back([&, t]() {
                while (true) {
                    const int e = next.fetch_add(1, std::memory_order_relaxed);
                    if (e >= experts) break;
                    RunLinearFloat16NVFP4(inputs[t].data(), *weights[e],
                        outputs[t].data(), nullptr, shape.rows, shape.m, shape.k,
                        GetAlivePool(), t, 1);
                }
            });
        }
        for (auto &th : pool) th.join();
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (pass > 0) {
            best = best == 0.0 ? seconds : std::min(best, seconds);
        }
    }
    const double bytes = (double)expertBytes * experts;
    printf("%-6s %-7s threads=%2d n=%d m=%4d k=%5d experts=%4d bytes=%6.2f GB  "
           "%8.1f ms  %6.1f GB/s  %6.2f GB/s/core\n",
           shape.name, "linear", threads, shape.rows,
           shape.m, shape.k, experts, bytes / 1e9, best * 1000.0,
           bytes / 1e9 / best, bytes / 1e9 / best / threads);
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool bench = false;
    double gigabytes = 4.0;
    std::vector<int> threadList;
    bool compactOnly = false, inlineOnly = false, linearBench = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--bench") bench = true;
        else if (arg == "--linear") linearBench = true;
        else if (arg == "--gb" && i + 1 < argc) gigabytes = atof(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threadList.push_back(atoi(argv[++i]));
        else if (arg == "--layout" && i + 1 < argc) {
            std::string layout = argv[++i];
            compactOnly = layout == "compact";
            inlineOnly = layout == "inline";
        }
    }
    if (gigabytes <= 0 ||
        std::any_of(threadList.begin(), threadList.end(), [](int n) { return n <= 0; })) {
        fprintf(stderr, "Benchmark size and threads must be positive.\n");
        return 1;
    }
    auto *info = GetCPUInstructInfo();
    printf("hasAVX2=%d hasAVX512BF16=%d\n", (int)info->hasAVX2, (int)info->hasAVX512BF16);
    Correctness({{2048, 1024}, {512, 2048}, {512, 17}, {48, 5}, {16, 3}, {31, 2}});
    StridedCorrectness();
    if (bench) {
        if (threadList.empty()) {
            threadList = {(int)std::thread::hardware_concurrency()};
        }
        // Shapes from Qwen3.5-35B-A3B: gateup [1024, 2048] and down [2048, 512].
        const BenchShape shapes[] = {
            {1, 2048, 1024, "gateup"},
            {1, 512, 2048, "down"},
            {4, 2048, 1024, "gateup"},
            {32, 2048, 1024, "gateup"},
        };
        for (int t : threadList) {
            for (const auto &shape : shapes) {
                if (linearBench) BenchLinear(shape, t, gigabytes);
                if (!compactOnly) Bench(shape, t, gigabytes, false);
                if (!inlineOnly) Bench(shape, t, gigabytes, true);
            }
        }
    }
    printf(failures == 0 ? "ALL PASS\n" : "FAILED %d\n", failures);
    return failures == 0 ? 0 : 1;
}
