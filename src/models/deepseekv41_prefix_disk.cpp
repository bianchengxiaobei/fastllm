//
// DeepSeek-V4.1 前缀缓存落盘（跨进程复用，重启后免重 prefill）。
//
// 内存版前缀缓存（v41HistoryCache）只在进程内有效：请求结束时
// TryRecordResponseContext 把完整请求状态（每层滑窗 KV、压缩 KV、indexer
// key、engram 历史、token 序列）快照成 DeepSeekV41HistoryMemory，下一个
// 请求按 token 前缀命中即免重 prefill；进程退出即全部丢失。本文件给同一
// 份快照追加磁盘拷贝：
//   * 写：请求结束后由后台线程异步落盘；文件先写 .tmp 再原子替换，按
//     (tokens, imageKeys) 内容哈希命名，同前缀的重复记录直接覆盖旧文件。
//   * 读：重启后第一次前缀查询时懒加载（std::call_once），指纹校验通过
//     的记录按文件时间从旧到新回灌 v41HistoryCache，之后的命中/截断/
//     恢复路径与内存版完全一致。
//   * 淘汰：FASTLLM_DSV41_PREFIX_CACHE_DISK_MIB 控制目录总预算（默认
//     20480，0 不限），超限按修改时间从旧到新删除整条记录文件。
// 指纹覆盖模型几何与 dtype（层数、滑窗、压缩比、engram、KV dtype 等），
// 配置或权重变化后旧记录整体作废，不会把不匹配的状态灌回模型。
// 生效前提与内存版一致：--cache_history true。
//
#include "deepseekv41.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <direct.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace fastllm {
    static_assert(sizeof(int) == 4, "record files serialize std::vector<int> verbatim");

    namespace {
        // ---------------- 基础工具 ----------------

        uint64_t V41DiskHashBytes(const void *data, size_t n, uint64_t h) {
            const uint8_t *p = (const uint8_t *)data;
            for (size_t i = 0; i < n; i++) {
                h ^= p[i];
                h *= 1099511628211ULL;
            }
            return h;
        }

        uint64_t V41DiskHashInt(uint64_t h, uint64_t v) {
            return V41DiskHashBytes(&v, sizeof(v), h);
        }

        uint64_t V41DiskHashInts(uint64_t h, const std::vector<int> &v) {
            h = V41DiskHashInt(h, (uint64_t)v.size());
            return V41DiskHashBytes(v.data(), v.size() * sizeof(int), h);
        }

        int V41DiskEnvInt(const char *name, int fallback) {
            const char *v = std::getenv(name);
            if (v == nullptr || v[0] == '\0') {
                return fallback;
            }
            return std::atoi(v);
        }

        std::string V41DiskLower(const char *s) {
            std::string out(s);
            std::transform(out.begin(), out.end(), out.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            return out;
        }

        // FASTLLM_DSV41_PREFIX_CACHE_DIR 指定 off/0/false/none/disabled 时整体关闭
        bool V41DiskDirEnvDisabled() {
            const char *v = std::getenv("FASTLLM_DSV41_PREFIX_CACHE_DIR");
            if (v == nullptr || v[0] == '\0') {
                return false;
            }
            std::string s = V41DiskLower(v);
            return s == "off" || s == "0" || s == "false" || s == "none" || s == "disabled";
        }

        // 内存版前缀缓存总开关；deepseekv41.cpp 匿名命名空间里的同名逻辑不可见，
        // 这里逐位复刻其 V41EnvFlagOn 语义：该开关本身"默认开启"（不设或非 "0"
        // 都算开），用在 DISABLE 名字上等于前缀缓存默认关闭，须显式设 0 才启用。
        bool V41DiskPrefixCacheDisabled() {
            const char *v = std::getenv("FASTLLM_DSV41_DISABLE_PREFIX_CACHE");
            return v == nullptr || v[0] == '\0' || strcmp(v, "0") != 0;
        }

        bool V41DiskDebug() {
            const char *v = std::getenv("FASTLLM_DSV41_PREFIX_CACHE_DEBUG");
            return v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0;
        }

        // ---------------- 路径与文件系统 ----------------

        bool V41DiskIsSep(char c) {
            return c == '/' || c == '\\';
        }

        std::string V41DiskJoin(const std::string &dir, const std::string &name) {
            if (dir.empty()) {
                return name;
            }
            if (V41DiskIsSep(dir.back())) {
                return dir + name;
            }
#ifdef _WIN32
            return dir + "\\" + name;
#else
            return dir + "/" + name;
#endif
        }

        std::string V41DiskBasename(std::string path) {
            while (!path.empty() && V41DiskIsSep(path.back())) {
                path.pop_back();
            }
            size_t pos = path.find_last_of("/\\");
            return pos == std::string::npos ? path : path.substr(pos + 1);
        }

        std::string V41DiskHomeDir() {
#if defined(_WIN32) || defined(_WIN64)
            const char *profile = std::getenv("USERPROFILE");
            if (profile != nullptr && profile[0] != '\0') {
                return profile;
            }
            const char *drive = std::getenv("HOMEDRIVE");
            const char *pathEnv = std::getenv("HOMEPATH");
            if (drive != nullptr && pathEnv != nullptr) {
                return std::string(drive) + pathEnv;
            }
#else
            const char *home = std::getenv("HOME");
            if (home != nullptr && home[0] != '\0') {
                return home;
            }
#endif
            return ".";
        }

        bool V41DiskMkdirOne(const std::string &path) {
#if defined(_WIN32) || defined(_WIN64)
            return _mkdir(path.c_str()) == 0 || GetLastError() == ERROR_ALREADY_EXISTS;
#else
            return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
        }

        bool V41DiskDirExists(const std::string &path) {
#if defined(_WIN32) || defined(_WIN64)
            DWORD attr = GetFileAttributesA(path.c_str());
            return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
#else
            struct stat st;
            return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
        }

        // 逐级创建目录，以最终目录确实存在为准
        bool V41DiskMakeDirs(const std::string &path) {
            if (path.empty() || V41DiskDirExists(path)) {
                return !path.empty() && V41DiskDirExists(path);
            }
            std::string cur;
            size_t i = 0;
            while (i < path.size()) {
                size_t next = path.find_first_of("/\\", i);
                if (next == std::string::npos) {
                    next = path.size();
                }
                if (next > i) {
                    if (!cur.empty()) {
                        cur += "/";
                    }
                    cur += path.substr(i, next - i);
                    V41DiskMkdirOne(cur);
                }
                i = next + 1;
            }
            return V41DiskDirExists(path);
        }

        struct V41DiskEntry {
            std::string name;
            std::string path;
            uint64_t size = 0;
            int64_t mtime = 0;
        };

        bool V41DiskListDir(const std::string &dir, std::vector<V41DiskEntry> &out) {
            out.clear();
#if defined(_WIN32) || defined(_WIN64)
            WIN32_FIND_DATAA data;
            HANDLE handle = FindFirstFileA(V41DiskJoin(dir, "*").c_str(), &data);
            if (handle == INVALID_HANDLE_VALUE) {
                return false;
            }
            do {
                std::string name = data.cFileName;
                if (name == "." || name == ".." || (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    continue;
                }
                V41DiskEntry entry;
                entry.name = name;
                entry.path = V41DiskJoin(dir, name);
                entry.size = ((uint64_t)data.nFileSizeHigh << 32) | (uint64_t)data.nFileSizeLow;
                entry.mtime = (int64_t)(((uint64_t)data.ftLastWriteTime.dwHighDateTime << 32 |
                                         data.ftLastWriteTime.dwLowDateTime) / 10000000ULL - 11644473600ULL);
                out.push_back(entry);
            } while (FindNextFileA(handle, &data));
            FindClose(handle);
            return true;
#else
            DIR *d = opendir(dir.c_str());
            if (d == nullptr) {
                return false;
            }
            struct dirent *item;
            while ((item = readdir(d)) != nullptr) {
                std::string name = item->d_name;
                if (name == "." || name == "..") {
                    continue;
                }
                V41DiskEntry entry;
                entry.name = name;
                entry.path = V41DiskJoin(dir, name);
                struct stat st;
                if (stat(entry.path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
                    entry.size = (uint64_t)st.st_size;
                    entry.mtime = (int64_t)st.st_mtime;
                    out.push_back(entry);
                }
            }
            closedir(d);
            return true;
#endif
        }

        bool V41DiskRemove(const std::string &path) {
            return std::remove(path.c_str()) == 0;
        }

        // 原子替换：tmp 改名为 dst（覆盖已存在的 dst）
        bool V41DiskReplaceFile(const std::string &tmp, const std::string &dst) {
#if defined(_WIN32) || defined(_WIN64)
            return MoveFileExA(tmp.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
            return ::rename(tmp.c_str(), dst.c_str()) == 0;
#endif
        }

        long long V41DiskNowMs() {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        // 文件的修改时间墙上时钟秒数（steady_clock 的纪元是开机时间，不能用来比 mtime）
        int64_t V41DiskWallSec() {
            return (int64_t)std::time(nullptr);
        }

        // ---------------- 序列化：写 ----------------

        struct V41DiskOut {
            FILE *f = nullptr;
            bool ok = false;

            explicit V41DiskOut(const std::string &path) {
                f = std::fopen(path.c_str(), "wb");
                ok = f != nullptr;
            }

            ~V41DiskOut() {
                if (f != nullptr) {
                    std::fclose(f);
                }
            }

            void Raw(const void *data, size_t n) {
                if (!ok || n == 0) {
                    return;
                }
                ok = std::fwrite(data, 1, n, f) == n;
            }

            void U32(uint32_t v) { Raw(&v, 4); }
            void I32(int32_t v) { Raw(&v, 4); }
            void U64(uint64_t v) { Raw(&v, 8); }

            void Ints(const std::vector<int> &v) {
                U32((uint32_t)v.size());
                Raw(v.data(), v.size() * sizeof(int));
            }

            bool Finish() {
                if (!ok || f == nullptr) {
                    return false;
                }
                ok = std::fflush(f) == 0;
                if (std::fclose(f) != 0) {
                    ok = false;
                }
                f = nullptr;
                return ok;
            }
        };

        void V41DiskWriteTensor(V41DiskOut &out, const Data &t) {
            // 只写非空张量（dims 为空视为不存在），CPU 快照张量由调用方保证
            out.U32((uint32_t)t.dims.size());
            for (int d : t.dims) {
                out.I32(d);
            }
            out.U32((uint32_t)t.expansionDims.size());
            for (int d : t.expansionDims) {
                out.I32(d);
            }
            out.I32((int32_t)t.dataType);
            uint64_t bytes = t.GetBytes();
            out.U64(bytes);
            out.Raw(t.cpuData, (size_t)bytes);
        }

        const uint32_t V41DiskWindowKVBit = 1;
        const uint32_t V41DiskCompressedKVBit = 2;
        const uint32_t V41DiskIndexKBit = 4;
        const uint32_t V41DiskRawTailKVBit = 8;
        const uint32_t V41DiskRawTailScoreBit = 16;

        // ---------------- 序列化：读 ----------------

        struct V41DiskIn {
            FILE *f = nullptr;
            uint64_t remaining = 0;
            bool ok = false;

            explicit V41DiskIn(const std::string &path) {
                f = std::fopen(path.c_str(), "rb");
                if (f == nullptr) {
                    return;
                }
                if (std::fseek(f, 0, SEEK_END) != 0) {
                    std::fclose(f);
                    f = nullptr;
                    return;
                }
#if defined(_WIN32) || defined(_WIN64)
                long long size = _ftelli64(f);
#else
                long long size = ftello(f);
#endif
                if (size < 0) {
                    std::fclose(f);
                    f = nullptr;
                    return;
                }
                remaining = (uint64_t)size;
                std::fseek(f, 0, SEEK_SET);
                ok = true;
            }

            ~V41DiskIn() {
                if (f != nullptr) {
                    std::fclose(f);
                }
            }

            bool Raw(void *data, size_t n) {
                if (!ok) {
                    return false;
                }
                if (n == 0) {
                    return true;
                }
                if ((uint64_t)n > remaining) {
                    ok = false;
                    return false;
                }
                if (std::fread(data, 1, n, f) != n) {
                    ok = false;
                    return false;
                }
                remaining -= n;
                return true;
            }

            uint32_t U32() {
                uint32_t v = 0;
                Raw(&v, 4);
                return v;
            }

            int32_t I32() {
                int32_t v = 0;
                Raw(&v, 4);
                return v;
            }

            uint64_t U64() {
                uint64_t v = 0;
                Raw(&v, 8);
                return v;
            }

            bool Ints(std::vector<int> &v, size_t maxCount) {
                uint32_t count = U32();
                if (!ok || count > maxCount) {
                    ok = false;
                    return false;
                }
                v.assign(count, 0);
                return Raw(v.data(), (size_t)count * sizeof(int));
            }
        };

        // 读入一个张量并按 CopyFrom 的分配语义原样重建（保留预扩容容量）。
        bool V41DiskReadTensor(V41DiskIn &in, Data &t, std::string &why) {
            uint32_t ndims = in.U32();
            if (!in.ok || ndims == 0 || ndims > 8) {
                why = "bad tensor rank";
                return false;
            }
            std::vector<int> dims(ndims);
            for (uint32_t i = 0; i < ndims; i++) {
                dims[i] = in.I32();
                if (!in.ok || dims[i] <= 0) {
                    why = "bad tensor dims";
                    return false;
                }
            }
            uint32_t expCount = in.U32();
            if (!in.ok || (expCount != 0 && expCount != ndims)) {
                why = "bad expansion dims";
                return false;
            }
            std::vector<int> expDims(expCount);
            for (uint32_t i = 0; i < expCount; i++) {
                expDims[i] = in.I32();
                // 预扩容形状必须逐维不小于真实形状，否则 Expansion 分配会小于数据量
                if (!in.ok || expDims[i] < dims[i]) {
                    why = "expansion dims < dims";
                    return false;
                }
            }
            int32_t dtype = in.I32();
            if (!in.ok || dtype < 0 || dtype > 15) {
                why = "bad tensor dtype";
                return false;
            }
            uint64_t bytes = in.U64();
            if (!in.ok || bytes > in.remaining) {
                why = "tensor bytes exceed file";
                return false;
            }
            t.dataType = (DataType)dtype;
            t.UpdateUnitSize();
            if (expCount > 0 && expDims != dims) {
                t.Expansion(expDims);
                t.Resize(dims);
                t.Allocate(false);
            } else {
                t.expansionDims.clear();
                t.Resize(dims);
                t.MallocSpace(t.Count(0), false);
            }
            if (!in.ok || t.GetBytes() != bytes) {
                why = "tensor byte count mismatch";
                return false;
            }
            if (!in.Raw(t.cpuData, (size_t)bytes)) {
                why = "short tensor read";
                return false;
            }
            return true;
        }
    }

    // ---------------- 磁盘存储 ----------------

    struct DeepSeekV41PrefixDiskStore {
        struct Job {
            std::shared_ptr<DeepSeekV41HistoryMemory> memory;
            uint64_t fingerprint = 0;
            uint64_t key = 0;
        };

        // 加载期校验上下文：与模型几何不匹配的记录整条跳过
        struct LoadContext {
            uint64_t fingerprint = 0;
            int blockCnt = 0;
            bool hasEngram = false;
            int maxLen = 0;
            const std::vector<char> *isKvSource = nullptr;
        };

        std::string modelDir;
        std::string dir;
        std::atomic<bool> enabled{false};
        bool customDir = false;
        bool fallbackTried = false;
        uint64_t capBytes = 0;

        std::mutex mutex;
        std::condition_variable cv;
        std::deque<Job> jobs;
        std::thread worker;
        bool stopping = false;

        explicit DeepSeekV41PrefixDiskStore(const std::string &modelDirectory)
                : modelDir(modelDirectory) {
            int capMib = V41DiskEnvInt("FASTLLM_DSV41_PREFIX_CACHE_DISK_MIB", 20480);
            capBytes = capMib > 0 ? (uint64_t)capMib << 20 : 0;
            const char *envDir = std::getenv("FASTLLM_DSV41_PREFIX_CACHE_DIR");
            std::string want;
            if (envDir != nullptr && envDir[0] != '\0') {
                want = envDir;
                customDir = true;
            } else if (!modelDir.empty()) {
                want = V41DiskJoin(modelDir, "fastllm_prefix_cache");
            }
            if (want.empty()) {
                return; // 没有模型目录也没有显式路径：保持关闭
            }
            if (V41DiskMakeDirs(want)) {
                dir = want;
                enabled.store(true);
            } else if (!customDir) {
                TryFallbackDir();
            }
            if (enabled.load()) {
                worker = std::thread([this]() { WorkerLoop(); });
            } else {
                printf("[fastllm-dsv41-prefix-cache] disk store disabled: no writable directory.\n");
                fflush(stdout);
            }
        }

        ~DeepSeekV41PrefixDiskStore() {
            {
                std::lock_guard<std::mutex> guard(mutex);
                stopping = true;
            }
            cv.notify_all();
            if (worker.joinable()) {
                worker.join();
            }
        }

        static uint64_t RecordKey(const DeepSeekV41HistoryMemory &m) {
            uint64_t h = 1469598103934665603ULL;
            h = V41DiskHashInt(h, 0x4441535634315043ULL); // "DSV41PC"
            h = V41DiskHashInt(h, (uint64_t)m.totalLen);
            h = V41DiskHashInts(h, m.tokens);
            for (const auto &key : m.imageKeys) {
                h = V41DiskHashInts(h, key);
            }
            return h;
        }

        static std::string RecordName(int totalLen, uint64_t key) {
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "record_%d_%016llx.bin", totalLen,
                          (unsigned long long)key);
            return buffer;
        }

        void Enqueue(Job &&job) {
            if (!enabled.load() || !job.memory) {
                return;
            }
            std::lock_guard<std::mutex> guard(mutex);
            // 同一前缀的待写任务只保留最新一份
            for (auto &pending : jobs) {
                if (pending.key == job.key) {
                    pending = std::move(job);
                    cv.notify_all();
                    return;
                }
            }
            jobs.push_back(std::move(job));
            cv.notify_all();
        }

        std::vector<std::shared_ptr<DeepSeekV41HistoryMemory> >
        LoadRecords(const LoadContext &ctx, int *skippedOut) {
            std::vector<std::shared_ptr<DeepSeekV41HistoryMemory> > records;
            int skipped = 0;
            std::vector<V41DiskEntry> entries;
            if (!V41DiskListDir(dir, entries)) {
                if (skippedOut != nullptr) {
                    *skippedOut = skipped;
                }
                return records;
            }
            std::vector<const V41DiskEntry *> files;
            for (const auto &entry : entries) {
                if (entry.name.compare(0, 7, "record_") == 0 &&
                    entry.name.size() > 4 && entry.name.compare(entry.name.size() - 4, 4, ".bin") == 0) {
                    files.push_back(&entry);
                }
            }
            // 旧的先进内存缓存，flushTime 语义下最新的记录留到最后
            std::sort(files.begin(), files.end(),
                      [](const V41DiskEntry *a, const V41DiskEntry *b) {
                          if (a->mtime != b->mtime) {
                              return a->mtime < b->mtime;
                          }
                          return a->name < b->name;
                      });
            for (const V41DiskEntry *file : files) {
                std::string why;
                auto memory = LoadOne(file->path, ctx, why);
                if (memory) {
                    records.push_back(memory);
                } else {
                    skipped++;
                    if (V41DiskDebug()) {
                        printf("[fastllm-dsv41-prefix-cache] skip record %s: %s\n",
                               file->name.c_str(), why.c_str());
                        fflush(stdout);
                    }
                }
            }
            if (skippedOut != nullptr) {
                *skippedOut = skipped;
            }
            return records;
        }

    private:
        void TryFallbackDir() {
            fallbackTried = true;
            // 模型目录只读（如挂载的权重盘）时，退到用户缓存目录
            std::string base = V41DiskBasename(modelDir);
            if (base.empty()) {
                base = "model";
            }
            std::string fallback = V41DiskJoin(V41DiskJoin(V41DiskJoin(V41DiskHomeDir(), ".cache"),
                                                           "fastllm"), "dsv41_prefix_cache");
            fallback = V41DiskJoin(fallback, base);
            if (V41DiskMakeDirs(fallback)) {
                dir = fallback;
                enabled.store(true);
                printf("[fastllm-dsv41-prefix-cache] model directory is not writable, "
                       "records go to %s\n", fallback.c_str());
                fflush(stdout);
            }
        }

        void WorkerLoop() {
            while (true) {
                Job job;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [this]() { return stopping || !jobs.empty(); });
                    if (jobs.empty()) {
                        if (stopping) {
                            return;
                        }
                        continue;
                    }
                    job = std::move(jobs.front());
                    jobs.pop_front();
                }
                std::string why;
                long long start = V41DiskNowMs();
                if (!WriteRecord(job, why)) {
                    if (!fallbackTried && !customDir) {
                        // 默认目录写入失败（如 ACL 只读）时试一次回退目录
                        std::lock_guard<std::mutex> guard(mutex);
                        if (!fallbackTried) {
                            TryFallbackDir();
                        }
                    }
                    if (enabled.load() && !WriteRecord(job, why)) {
                        printf("[fastllm-dsv41-prefix-cache] record write failed: %s\n", why.c_str());
                        fflush(stdout);
                        continue;
                    }
                    if (!enabled.load()) {
                        continue;
                    }
                }
                long long ms = V41DiskNowMs() - start;
                if (ms > 100 || V41DiskDebug()) {
                    printf("[fastllm-dsv41-prefix-cache] record write took %lld ms "
                           "(tokens=%d, file=%s)\n", ms, job.memory ? job.memory->totalLen : 0,
                           RecordName(job.memory ? job.memory->totalLen : 0, job.key).c_str());
                    fflush(stdout);
                }
                EnforceCap();
            }
        }

        bool WriteRecord(const Job &job, std::string &why) {
            if (!enabled.load() || !job.memory) {
                why = "store disabled";
                return false;
            }
            const DeepSeekV41HistoryMemory &m = *job.memory;
            if (m.totalLen <= 0 || (int)m.tokens.size() != m.totalLen) {
                why = "bad memory";
                return false;
            }
            bool hasWindow = true;
            for (const auto &layer : m.layers) {
                if (layer.windowKV.dims.empty()) {
                    hasWindow = false;
                    break;
                }
            }
            if (!hasWindow) {
                // totalLen>0 时每层滑窗必然有数据；没有就说明快照不完整
                why = "missing window KV";
                return false;
            }
            std::string base = RecordName(m.totalLen, job.key);
            std::string tmp = V41DiskJoin(dir, base + ".tmp");
            std::string dst = V41DiskJoin(dir, base);
            {
                V41DiskOut out(tmp);
                out.Raw("FLDSV41C", 8);
                out.U32(1); // version
                out.U64(job.fingerprint);
                out.I32(m.totalLen);
                out.Ints(m.tokens);
                out.Ints(m.engramHistory);
                out.U32((uint32_t)m.imageKeys.size());
                for (const auto &key : m.imageKeys) {
                    out.Ints(key);
                }
                out.U32((uint32_t)m.layers.size());
                for (const auto &layer : m.layers) {
                    out.I32(layer.totalLen);
                    out.I32(layer.compressedBlocks);
                    out.I32(layer.rawTail);
                    uint32_t mask = V41DiskWindowKVBit;
                    if (!layer.compressedKV.dims.empty()) mask |= V41DiskCompressedKVBit;
                    if (!layer.indexK.dims.empty()) mask |= V41DiskIndexKBit;
                    if (!layer.rawTailKV.dims.empty()) mask |= V41DiskRawTailKVBit;
                    if (!layer.rawTailScore.dims.empty()) mask |= V41DiskRawTailScoreBit;
                    out.U32(mask);
                    V41DiskWriteTensor(out, layer.windowKV);
                    if (mask & V41DiskCompressedKVBit) V41DiskWriteTensor(out, layer.compressedKV);
                    if (mask & V41DiskIndexKBit) V41DiskWriteTensor(out, layer.indexK);
                    if (mask & V41DiskRawTailKVBit) V41DiskWriteTensor(out, layer.rawTailKV);
                    if (mask & V41DiskRawTailScoreBit) V41DiskWriteTensor(out, layer.rawTailScore);
                }
                if (!out.Finish()) {
                    V41DiskRemove(tmp);
                    why = "write error";
                    return false;
                }
            }
            if (!V41DiskReplaceFile(tmp, dst)) {
                V41DiskRemove(tmp);
                why = "rename error";
                return false;
            }
            return true;
        }

        // 目录总预算：超限时从最旧的记录文件开始删；顺带清理陈旧的 .tmp
        void EnforceCap() {
            if (capBytes == 0) {
                return;
            }
            std::vector<V41DiskEntry> entries;
            if (!V41DiskListDir(dir, entries)) {
                return;
            }
            std::vector<const V41DiskEntry *> records, tmps;
            uint64_t total = 0;
            for (const auto &entry : entries) {
                if (entry.name.compare(0, 7, "record_") == 0 &&
                    entry.name.size() > 4 && entry.name.compare(entry.name.size() - 4, 4, ".bin") == 0) {
                    records.push_back(&entry);
                    total += entry.size;
                } else if (entry.name.size() > 4 &&
                           entry.name.compare(entry.name.size() - 4, 4, ".tmp") == 0 &&
                           entry.mtime < V41DiskWallSec() - 6 * 3600) {
                    tmps.push_back(&entry);
                }
            }
            for (const V41DiskEntry *tmp : tmps) {
                V41DiskRemove(tmp->path);
            }
            if (total <= capBytes) {
                return;
            }
            std::sort(records.begin(), records.end(),
                      [](const V41DiskEntry *a, const V41DiskEntry *b) {
                          if (a->mtime != b->mtime) {
                              return a->mtime < b->mtime;
                          }
                          return a->name < b->name;
                      });
            int removed = 0;
            for (const V41DiskEntry *record : records) {
                if (total <= capBytes) {
                    break;
                }
                if (V41DiskRemove(record->path)) {
                    total -= record->size;
                    removed++;
                }
            }
            if (removed > 0) {
                printf("[fastllm-dsv41-prefix-cache] disk cap: removed %d oldest records, "
                       "total=%.1f MiB cap=%.1f MiB\n", removed, total / 1048576.0,
                       capBytes / 1048576.0);
                fflush(stdout);
            }
        }

        std::shared_ptr<DeepSeekV41HistoryMemory> LoadOne(const std::string &path,
                                                          const LoadContext &ctx,
                                                          std::string &why) {
            V41DiskIn in(path);
            if (!in.ok) {
                why = "cannot open";
                return nullptr;
            }
            char magic[8];
            if (!in.Raw(magic, 8) || memcmp(magic, "FLDSV41C", 8) != 0) {
                why = "bad magic";
                return nullptr;
            }
            uint32_t version = in.U32();
            uint64_t fingerprint = in.U64();
            if (!in.ok || version != 1) {
                why = "bad version";
                return nullptr;
            }
            if (fingerprint != ctx.fingerprint) {
                why = "fingerprint mismatch";
                return nullptr;
            }
            int32_t totalLen = in.I32();
            if (!in.ok || totalLen <= 0 || totalLen > ctx.maxLen) {
                why = "bad total length";
                return nullptr;
            }
            auto memory = std::make_shared<DeepSeekV41HistoryMemory>();
            memory->totalLen = totalLen;
            if (!in.Ints(memory->tokens, (size_t)totalLen) ||
                (int)memory->tokens.size() != totalLen) {
                why = "bad tokens";
                return nullptr;
            }
            if (!in.Ints(memory->engramHistory, ctx.hasEngram ? (size_t)totalLen : 0) ||
                (ctx.hasEngram && (int)memory->engramHistory.size() != totalLen)) {
                why = "bad engram history";
                return nullptr;
            }
            uint32_t imageRows = in.U32();
            if (!in.ok || imageRows > 4096) {
                why = "bad image keys";
                return nullptr;
            }
            memory->imageKeys.resize(imageRows);
            for (uint32_t i = 0; i < imageRows; i++) {
                // 空行会让 GetCandidates 的 storedImages[i][0] 越界，必须拒掉
                if (!in.Ints(memory->imageKeys[i], 64) || memory->imageKeys[i].empty()) {
                    why = "bad image key row";
                    return nullptr;
                }
            }
            uint32_t layerCount = in.U32();
            if (!in.ok || (int)layerCount != ctx.blockCnt) {
                why = "layer count mismatch";
                return nullptr;
            }
            memory->layers.resize(layerCount);
            for (uint32_t i = 0; i < layerCount; i++) {
                DeepSeekV41LayerCache &layer = memory->layers[i];
                int32_t layerLen = in.I32();
                int32_t compressedBlocks = in.I32();
                int32_t rawTail = in.I32();
                uint32_t mask = in.U32();
                if (!in.ok || layerLen != totalLen || compressedBlocks < 0 ||
                    compressedBlocks > totalLen || rawTail < 0 || rawTail > 4096) {
                    why = "bad layer header";
                    return nullptr;
                }
                bool kvSource = ctx.isKvSource != nullptr && i < ctx.isKvSource->size() &&
                                (*ctx.isKvSource)[i] != 0;
                uint32_t allowed = V41DiskWindowKVBit;
                if (kvSource) {
                    allowed |= V41DiskCompressedKVBit | V41DiskIndexKBit |
                               V41DiskRawTailKVBit | V41DiskRawTailScoreBit;
                }
                if ((mask & ~allowed) != 0 || (mask & V41DiskWindowKVBit) == 0) {
                    why = "bad layer tensor mask";
                    return nullptr;
                }
                layer.totalLen = layerLen;
                layer.compressedBlocks = compressedBlocks;
                layer.rawTail = rawTail;
                if (!V41DiskReadTensor(in, layer.windowKV, why) ||
                    ((mask & V41DiskCompressedKVBit) && !V41DiskReadTensor(in, layer.compressedKV, why)) ||
                    ((mask & V41DiskIndexKBit) && !V41DiskReadTensor(in, layer.indexK, why)) ||
                    ((mask & V41DiskRawTailKVBit) && !V41DiskReadTensor(in, layer.rawTailKV, why)) ||
                    ((mask & V41DiskRawTailScoreBit) && !V41DiskReadTensor(in, layer.rawTailScore, why))) {
                    return nullptr;
                }
            }
            if (in.remaining != 0) {
                why = "trailing bytes";
                return nullptr;
            }
            return memory;
        }
    };

    // ---------------- 模型侧接口 ----------------

    uint64_t DeepSeekV41Model::V41PrefixDiskFingerprint() const {
        uint64_t h = 1469598103934665603ULL;
        h = V41DiskHashBytes("deepseek_v41", 12, h);
        h = V41DiskHashInt(h, (uint64_t)this->block_cnt);
        h = V41DiskHashInt(h, (uint64_t)this->window_size);
        h = V41DiskHashInt(h, (uint64_t)this->head_dim_full);
        h = V41DiskHashInt(h, (uint64_t)this->qk_rope_head_dim);
        h = V41DiskHashInt(h, (uint64_t)this->qk_nope_head_dim);
        h = V41DiskHashInt(h, (uint64_t)this->index_head_dim);
        h = V41DiskHashInt(h, (uint64_t)this->image_token_id);
        h = V41DiskHashInts(h, this->compress_ratios);
        h = V41DiskHashInts(h, this->kv_source_layer_ids);
        h = V41DiskHashInts(h, this->index_source_layer_ids);
        h = V41DiskHashInts(h, this->engram_layer_ids);
        h = V41DiskHashInt(h, (uint64_t)this->engram_max_ngram_size);
        h = V41DiskHashInt(h, (uint64_t)this->engram_compressed_vocab_size);
        h = V41DiskHashInt(h, (uint64_t)this->dataType);
        h = V41DiskHashInt(h, (uint64_t)this->kvCacheDataType);
        return h;
    }

    void DeepSeekV41Model::V41PrefixDiskRecord(const std::shared_ptr<DeepSeekV41HistoryMemory> &memory) {
        V41PrefixDiskEnsureLoaded();
        std::shared_ptr<DeepSeekV41PrefixDiskStore> store = v41PrefixDiskStore;
        if (!store || !store->enabled.load() || !memory) {
            return;
        }
        DeepSeekV41PrefixDiskStore::Job job;
        job.memory = memory;
        job.fingerprint = V41PrefixDiskFingerprint();
        job.key = DeepSeekV41PrefixDiskStore::RecordKey(*memory);
        store->Enqueue(std::move(job));
    }

    void DeepSeekV41Model::V41PrefixDiskEnsureLoaded() {
        std::call_once(v41PrefixDiskLoadOnce, [this]() {
            v41PrefixDiskLoaded = true;
            if (V41DiskPrefixCacheDisabled() || V41DiskDirEnvDisabled()) {
                return;
            }
            std::string modelDirectory;
            auto it = this->weight.dicts.find("model_directory");
            if (it != this->weight.dicts.end()) {
                modelDirectory = it->second;
            }
            auto store = std::make_shared<DeepSeekV41PrefixDiskStore>(modelDirectory);
            if (!store->enabled.load()) {
                return;
            }
            v41PrefixDiskStore = store;
            DeepSeekV41PrefixDiskStore::LoadContext ctx;
            ctx.fingerprint = V41PrefixDiskFingerprint();
            ctx.blockCnt = this->block_cnt;
            ctx.hasEngram = !this->engram_layer_ids.empty();
            ctx.maxLen = this->max_positions > 0 ? this->max_positions : (1 << 20);
            ctx.isKvSource = &this->isKvSource;
            int skipped = 0;
            long long start = V41DiskNowMs();
            auto records = store->LoadRecords(ctx, &skipped);
            for (auto &memory : records) {
                this->v41HistoryCache.Record(memory);
            }
            long long ms = V41DiskNowMs() - start;
            printf("[fastllm-dsv41-prefix-cache] disk load: dir=%s adopted=%d retained=%d "
                   "skipped=%d took=%lld ms (in-memory cap: "
                   "FASTLLM_DSV41_PREFIX_CACHE_MAX_RECORDS)\n", store->dir.c_str(),
                   (int)records.size(), (int)this->v41HistoryCache.memorys.size(), skipped, ms);
            fflush(stdout);
        });
    }
}
