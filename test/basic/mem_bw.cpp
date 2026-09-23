#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// 纯读 / 拷贝内存带宽基准，用来判断推理是否已经打满 DRAM 带宽。
// 用法: mem_bw [sizeMB] [threads]
//   sizeMB  默认 4096，需要远大于 L3
//   threads 默认硬件线程数，测上限时给物理核数

namespace {
    double gSink = 0.0;

    void *AllocAligned(size_t bytes) {
#ifdef _WIN32
        return _aligned_malloc(bytes, 64);
#else
        void *p = nullptr;
        return posix_memalign(&p, 64, bytes) == 0 ? p : nullptr;
#endif
    }

    void FreeAligned(void *p) {
#ifdef _WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template <typename Fn>
    void ParallelRun(int threads, const Fn &fn) {
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (int t = 0; t < threads; t++) {
            pool.emplace_back([&fn, t]() { fn(t); });
        }
        for (auto &th : pool) {
            th.join();
        }
    }

    // 每线程独占一段连续区间，避免线程间互相干扰
    double ReadBandwidth(uint8_t *src, size_t bytes, int threads) {
        std::vector<double> partial(threads, 0.0);
        auto st = std::chrono::steady_clock::now();
        ParallelRun(threads, [&](int t) {
            size_t per = bytes / sizeof(float) / threads;
            const float *p = reinterpret_cast<const float *>(src) + (size_t)t * per;
            double sum = 0.0;
#if defined(__AVX2__)
            __m256 acc = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 8 <= per; i += 8) {
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(p + i));
            }
            float lane[8];
            _mm256_storeu_ps(lane, acc);
            for (int k = 0; k < 8; k++) {
                sum += lane[k];
            }
            for (; i < per; i++) {
                sum += p[i];
            }
#else
            for (size_t i = 0; i < per; i++) {
                sum += p[i];
            }
#endif
            partial[t] = sum;
        });
        double spend = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - st).count();
        for (double v : partial) {
            gSink += v;
        }
        size_t moved = bytes / sizeof(float) / threads * threads * sizeof(float);
        return spend > 0.0 ? (double)moved / spend / 1e9 : 0.0;
    }

    // 非临时写，目标块不进 cache，避免写分配再吃一遍带宽
    double CopyBandwidth(uint8_t *src, uint8_t *dst, size_t bytes, int threads) {
        const size_t block = 32;
        size_t per = bytes / block / threads;
        if (per == 0) {
            return 0.0;
        }
        auto st = std::chrono::steady_clock::now();
        ParallelRun(threads, [&](int t) {
            const uint8_t *s = src + (size_t)t * per * block;
            uint8_t *d = dst + (size_t)t * per * block;
            for (size_t i = 0; i < per; i++) {
#if defined(__AVX2__)
                _mm256_stream_si256(
                    reinterpret_cast<__m256i *>(d + i * block),
                    _mm256_load_si256(reinterpret_cast<const __m256i *>(s + i * block)));
#else
                memcpy(d + i * block, s + i * block, block);
#endif
            }
#if defined(__AVX2__)
            _mm_sfence();
#endif
        });
        double spend = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - st).count();
        size_t moved = per * threads * block;
        return spend > 0.0 ? (double)moved / spend / 1e9 : 0.0;
    }
}

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? (size_t)strtoull(argv[1], nullptr, 10) : 4096;
    int threads = argc > 2 ? atoi(argv[2]) : (int)std::thread::hardware_concurrency();
    if (mb == 0) {
        mb = 4096;
    }
    if (threads <= 0) {
        threads = 8;
    }
    size_t bytes = mb << 20;

    uint8_t *src = (uint8_t *)AllocAligned(bytes);
    uint8_t *dst = (uint8_t *)AllocAligned(bytes);
    if (src == nullptr || dst == nullptr) {
        fprintf(stderr, "allocate %zu MB failed\n", mb);
        FreeAligned(src);
        FreeAligned(dst);
        return 1;
    }
    memset(src, 1, bytes);
    memset(dst, 0, bytes);

    printf("mem_bw: size = %zu MB, threads = %d (best of 3)\n", mb, threads);
    double bestRead = 0.0, bestCopy = 0.0;
    for (int r = 0; r < 3; r++) {
        double read = ReadBandwidth(src, bytes, threads);
        double copy = CopyBandwidth(src, dst, bytes, threads);
        if (read > bestRead) {
            bestRead = read;
        }
        if (copy > bestCopy) {
            bestCopy = copy;
        }
    }
    printf("read = %.2f GB/s, copy = %.2f GB/s (sink = %.3g)\n", bestRead, bestCopy, gSink);

    FreeAligned(src);
    FreeAligned(dst);
    return 0;
}
