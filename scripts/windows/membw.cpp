#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

// 纯读带宽测试: 512MB 连续大块(远大于 E5-2699v4 的 55MB L3), 多线程各读一段,
// 8 字节步进累加. 用来判断 fastllm CPU MoE 的 15.8GB/s 是不是贴到了 DRAM 上限:
// 实测 >= 25GB/s 说明还差得远, 瓶颈在 AVX2 int4 解包算力/线程池调度;
// 实测 <= 18GB/s 说明已经贴墙, 插满内存通道才有收益.
//
// 用法: build_membw.bat  然后   membw.exe

static const size_t kBytes = 512ull << 20;
static const int kRepeat = 3;

static double NowSec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double RunOnce(unsigned char *buf, int threads) {
    std::atomic<int> ready(0);
    std::atomic<bool> go(false);
    std::vector<double> sink(threads, 0.0);
    std::vector<std::thread> pool;
    size_t chunk = kBytes / threads;
    for (int t = 0; t < threads; t++) {
        pool.emplace_back([&, t]() {
            size_t from = chunk * t, to = (t + 1 == threads) ? kBytes : chunk * (t + 1);
            const unsigned long long *p = (const unsigned long long *)(buf + from);
            size_t n = (to - from) / sizeof(unsigned long long);
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            unsigned long long acc = 0;
            for (size_t i = 0; i < n; i++) {
                acc += p[i];
            }
            sink[t] = (double)acc;
        });
    }
    while (ready.load() != threads) {
        std::this_thread::yield();
    }
    double begin = NowSec();
    go.store(true, std::memory_order_release);
    for (auto &th : pool) {
        th.join();
    }
    double end = NowSec();
    double total = 0.0;
    for (double v : sink) {
        total += v;
    }
    if (total == 12345.678) {
        printf(" ");
    }
    return end - begin;
}

int main() {
    unsigned char *buf = (unsigned char *)_aligned_malloc(kBytes, 64);
    if (buf == nullptr) {
        printf("alloc failed\n");
        return 1;
    }
    memset(buf, 1, kBytes);
    unsigned hw = std::thread::hardware_concurrency();
    printf("hw_threads=%u buffer=%zuMB\n", hw, kBytes >> 20);
    int cases[] = {1, 2, 4, 8, 12, 16, 22, 32, 44};
    for (int threads : cases) {
        if (threads > (int)hw) {
            continue;
        }
        RunOnce(buf, threads);
        double best = 1e9;
        for (int r = 0; r < kRepeat; r++) {
            double s = RunOnce(buf, threads);
            if (s < best) {
                best = s;
            }
        }
        printf("threads=%2d  read=%7.2f GB/s\n", threads,
               (double)kBytes / best / 1e9);
        fflush(stdout);
    }
    _aligned_free(buf);
    return 0;
}
