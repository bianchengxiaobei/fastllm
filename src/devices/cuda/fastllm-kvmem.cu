#include "kvmem.h"
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace fastllm {
namespace {
    void Check(cudaError_t error) {
        if (error != cudaSuccess) throw std::runtime_error(std::string("KVMem CUDA: ") + cudaGetErrorString(error));
    }

    template<class T>
    __global__ void IndexKeys(const T *keys, float *sums, int oldTokens, int tokens,
                              int pageLen, int width, int firstPage) {
        int feature = blockIdx.x * blockDim.x + threadIdx.x;
        int page = firstPage + blockIdx.y;
        if (feature >= width) return;
        int start = max(oldTokens, page * pageLen);
        int end = min(oldTokens + tokens, (page + 1) * pageLen);
        float value = 0;
        for (int t = start; t < end; ++t) value += (float)keys[(t - oldTokens) * width + feature];
        sums[(size_t)page * width + feature] += value;
    }

    template<class T>
    __global__ void ScorePages(const T *query, const float *sums, float *scores,
                               int pages, int oldTokens, int pageLen, int heads,
                               int kvHeads, int dim, float normPower) {
        int page = blockIdx.x, head = blockIdx.y;
        int kvHead = head / (heads / kvHeads);
        float value = 0, squared = 0;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            float sum = sums[((size_t)page * kvHeads + kvHead) * dim + d];
            value += (float)query[head * dim + d] * sum;
            squared += sum * sum;
        }
        __shared__ float partial[256];
        __shared__ float partialSquared[256];
        partial[threadIdx.x] = value;
        partialSquared[threadIdx.x] = squared;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
                partialSquared[threadIdx.x] += partialSquared[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            int count = min(pageLen, oldTokens - page * pageLen);
            // meanK = sumK / count, so scaling by meanK^normPower is a scale of
            // count by that power: 0 keeps the raw dot product, 1 divides by the
            // page's mean key norm and leaves the cosine similarity.
            float mean = sqrtf(partialSquared[0]) / count;
            float scale = count * sqrtf((float)dim);
            if (normPower > 0.0f) {
                scale *= powf(mean, normPower);
            }
            scores[(size_t)head * pages + page] =
                    mean > 0.0f ? partial[0] / scale : 0.0f;
        }
    }

    template<class T>
    __global__ void AppendKV(const T *k, const T *v, T *cacheK, T *cacheV,
                             const int *slots, int oldTokens, int tokens,
                             int pageLen, int heads, int dim) {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= (size_t)heads * tokens * dim) return;
        int d = i % dim, t = (i / dim) % tokens, h = i / ((size_t)tokens * dim);
        int logical = oldTokens + t;
        int slot = slots[logical / pageLen - oldTokens / pageLen];
        size_t dst = (((size_t)slot * pageLen + logical % pageLen) * heads + h) * dim + d;
        cacheK[dst] = k[i];
        cacheV[dst] = v[i];
    }

    // 换页和前缀 dump 的 D2H/H2D 中转缓冲. 目标页必须保持 pageable(它们是共享的、
    // 长期驻留的主机内存), 但 pageable 的 D2H 只有 ~250 MB/s —— 驱动得为每次拷贝
    // 走它自己的 bounce buffer. 先拷进这块固定的 pinned 缓冲才是真异步(几 GB/s),
    // 再 memcpy 进页缓冲: 4096 token 的 dump 从 ~330ms 降到几十 ms, 冷页换出和
    // 恢复走同一块缓冲, pinned 内存只多占几 MB.
    static const int KvMemStagingSlots = 8;

    // Pinned relay budget for one paging batch. A batch that fits pays a single
    // DMA round trip; what does not fit goes straight through the pageable page
    // buffers, where the extra host copy would not pay off.
    static const size_t KvMemRelayBytes = 8u << 20;

    // Grow-only pinned buffer. A lease holds the mutex for its whole lifetime,
    // so the bytes of a batch in flight cannot be reallocated (or freed) by
    // another staging user underneath it.
    struct KvMemPinnedBuffer {
        std::mutex mutex;
        void *data = nullptr;
        int device = -1;
        size_t bytes = 0;

        ~KvMemPinnedBuffer() {
            if (data != nullptr) {
                cudaFreeHost(data);
                data = nullptr;
            }
        }

        // Caller must hold mutex.
        bool Ensure(int targetDevice, size_t need) {
            if (data != nullptr && device == targetDevice && bytes >= need) {
                return true;
            }
            void *fresh = nullptr;
            if (cudaHostAlloc(&fresh, need, cudaHostAllocDefault) != cudaSuccess) {
                return false;
            }
            if (data != nullptr) {
                cudaFreeHost(data);
            }
            data = fresh;
            device = targetDevice;
            bytes = need;
            return true;
        }
    };

    // KVMem is single-request, so one buffer per device and role is enough; the
    // lease serializes any accidental overlap instead of handing out a live
    // pointer. Leases must not nest.
    class KvMemStagingLease {
    public:
        KvMemStagingLease(KvMemPinnedBuffer &buffer, int device, size_t bytes)
            : buffer(buffer), lock(buffer.mutex),
              usable(buffer.Ensure(device, bytes) && buffer.data != nullptr) {}

        KvMemStagingLease(const KvMemStagingLease &) = delete;
        KvMemStagingLease &operator=(const KvMemStagingLease &) = delete;

        bool Ok() const { return usable; }
        uint8_t *Data() const { return (uint8_t*)buffer.data; }
        uint8_t *Slot(size_t stride, size_t index) const {
            return (uint8_t*)buffer.data + stride * index;
        }

    private:
        KvMemPinnedBuffer &buffer;
        std::unique_lock<std::mutex> lock;
        bool usable = false;
    };

    // One pinned relay per device and role: a tensor-parallel request pages on
    // several devices at once, and a single shared buffer would be freed and
    // reallocated (draining its DMA queue) every time the device switched.
    KvMemPinnedBuffer &KvMemStagingFor(const char *role, int device) {
        static std::mutex locker;
        static std::map <std::string, std::map <int, KvMemPinnedBuffer*> > buffers;
        std::lock_guard <std::mutex> guard(locker);
        KvMemPinnedBuffer *&buffer = buffers[role][device];
        if (buffer == nullptr) {
            buffer = new KvMemPinnedBuffer();
        }
        return *buffer;
    }

    KvMemPinnedBuffer &KvMemPageStaging(int device) {
        return KvMemStagingFor("page", device);
    }

    KvMemPinnedBuffer &KvMemScoreStaging(int device) {
        return KvMemStagingFor("score", device);
    }

    class CudaKvMemCache final : public KvMemCache {
        KvMemConfig config;
        int device, pageLen, heads, dim;
        int previousAppendTokens = 0;
        // Retrieval scores are published by this layer's owning device so every
        // device selects the same pages; consumedStamp is this device's reader
        // state.
        std::shared_ptr<KvMemScorePlan> plan;
        uint64_t consumedStamp = 0;
        bool failed = false;
        DataType type;
        size_t onePageBytes;
        KvMemPages pages;
        PagedCacheManager poolK, poolV;
        // FP32 retrieval index [sumsCapacity, heads, dim] and the per-refresh
        // score scratch [scoreCapacity, scoreHeads]. Both grow with the
        // committed length instead of reserving maxTokens, so a short request
        // in a long-context config does not hold the whole logical context.
        // Values and layout are identical to a full-size allocation.
        void *sumsData = nullptr;
        int sumsCapacity = 0;
        void *scoreData = nullptr;
        int scoreCapacity = 0, scoreHeads = 0;
        Data writeSlots, provisionalKeys;

        // Grow geometrically so the pool is touched O(log n) times instead of
        // once per chunk.
        void EnsureSumsPages(int need) {
            if (need <= sumsCapacity) return;
            const int grown = sumsCapacity > 0 ? std::max(need, sumsCapacity * 2) : need;
            const size_t rowBytes = (size_t)heads * dim * sizeof(float);
            void *fresh = FastllmCudaMalloc((size_t)grown * rowBytes);
            if (fresh == nullptr) {
                throw std::runtime_error("KVMem: cannot allocate the retrieval index");
            }
            try {
                Check(cudaMemsetAsync(fresh, 0, (size_t)grown * rowBytes, cudaStreamPerThread));
                if (sumsData != nullptr) {
                    Check(cudaMemcpyAsync(fresh, sumsData, (size_t)sumsCapacity * rowBytes,
                                          cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                    Check(cudaStreamSynchronize(cudaStreamPerThread));
                    FastllmCudaFree(sumsData);
                }
            } catch (...) {
                FastllmCudaFree(fresh);
                throw;
            }
            sumsData = fresh;
            sumsCapacity = grown;
        }

        void EnsureScorePages(int need, int qHeads) {
            if (need <= scoreCapacity && qHeads == scoreHeads) return;
            const int grown = (scoreCapacity > 0 && qHeads == scoreHeads)
                ? std::max(need, scoreCapacity * 2) : need;
            void *fresh = FastllmCudaMalloc((size_t)grown * qHeads * sizeof(float));
            if (fresh == nullptr) {
                throw std::runtime_error("KVMem: cannot allocate retrieval scores");
            }
            if (scoreData != nullptr) {
                // Scores are rewritten on every refresh, so no copy is needed.
                FastllmCudaFree(scoreData);
            }
            scoreData = fresh;
            scoreCapacity = grown;
            scoreHeads = qHeads;
        }

        void Publish(Data &keyCache, Data &valueCache) {
            for (Data *cache : {&keyCache, &valueCache}) {
                cache->dataDevice = DataDevice::CUDA;
                cache->dataDeviceIds = {device};
                cache->isPagedKVCache = true;
                cache->pageLen = pageLen;
                cache->pagedKVCacheData = cache == &keyCache ? &poolK : &poolV;
                cache->Resize({heads, pages.Tokens(), dim});
                cache->lastPageLen = pages.Tokens() ? (pages.Tokens() - 1) % pageLen + 1 : 0;
                cache->pageIndex.clear();
                for (int p : pages.Visible()) cache->pageIndex.push_back(pages.Slot(p));
            }
        }

        template<class T> void Index(const Data &keys, int oldTokens, int tokens) {
            if (!tokens) return;
            int first = oldTokens / pageLen, last = (oldTokens + tokens - 1) / pageLen;
            IndexKeys<T><<<dim3((heads * dim + 255) / 256, last - first + 1), 256, 0, cudaStreamPerThread>>>(
                (const T*)keys.cudaData, (float*)sumsData, oldTokens, tokens, pageLen, heads * dim, first);
            Check(cudaGetLastError());
        }

        void Allocate(Data &data, DataType dtype, const std::vector<int> &shape) {
            data.dataType = dtype;
            data.UpdateUnitSize();
            data.dataDevice = DataDevice::CUDA;
            data.dataDeviceIds = {device};
            data.Resize(shape);
            data.Allocate();
        }

        // Copies bounce through the pinned relay: a pageable D2H runs at a
        // fraction of the pinned rate. One batch pays one sync, so the DMA queue
        // stays busy; only what fits in the relay takes the extra host copy.
        template<bool restore, class Page>
        void TransferPages(const std::vector<Page> &batch) {
            if (batch.empty()) return;
            const size_t perPage = onePageBytes * 2;
            const size_t fits = std::max<size_t>(1, KvMemRelayBytes / perPage);
            const int relayed = (int)std::min(fits, batch.size());
            KvMemStagingLease lease(KvMemPageStaging(device), device, perPage * (size_t)relayed);
            TransferChunk<restore>(batch, 0, relayed, relayed > 0 ? lease.Data() : nullptr);
            if (relayed < (int)batch.size()) {
                TransferChunk<restore>(batch, (size_t)relayed,
                                       (int)batch.size() - relayed, nullptr);
            }
        }

        template<bool restore, class Page>
        void TransferChunk(const std::vector<Page> &batch, size_t base, int count,
                           uint8_t *ring) {
            try {
                auto key = [&](int slot) {
                    return (uint8_t*)poolK.cudaData + (size_t)slot * onePageBytes;
                };
                auto value = [&](int slot) {
                    return (uint8_t*)poolV.cudaData + (size_t)slot * onePageBytes;
                };
                if (ring == nullptr) {
                    // No pinned memory: mirror each page straight through the
                    // pageable page buffer, which the driver has to bounce.
                    for (int i = 0; i < count; ++i) {
                        const Page &page = batch[base + i];
                        const uint8_t *src = (const uint8_t*)page.data;
                        if constexpr (restore) {
                            Check(cudaMemcpyAsync(key(page.slot), src, onePageBytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
                            Check(cudaMemcpyAsync(value(page.slot), src + onePageBytes, onePageBytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
                        } else {
                            uint8_t *dst = const_cast<uint8_t *>(src);
                            Check(cudaMemcpyAsync(dst, key(page.slot), onePageBytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                            Check(cudaMemcpyAsync(dst + onePageBytes, value(page.slot), onePageBytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                        }
                    }
                    Check(cudaStreamSynchronize(cudaStreamPerThread));
                    return;
                }
                uint8_t *ringKey = ring, *ringValue = ring + onePageBytes * (size_t)count;
                for (int start = 0; start < count; ) {
                    int end = start + 1;
                    while (end < count &&
                           batch[base + end].slot == batch[base + end - 1].slot + 1) {
                        ++end;
                    }
                    const int slot = batch[base + start].slot;
                    const size_t runBytes = (size_t)(end - start) * onePageBytes;
                    if constexpr (restore) {
                        for (int i = start; i < end; ++i) {
                            const uint8_t *page = (const uint8_t*)batch[base + i].data;
                            std::memcpy(ringKey + onePageBytes * (size_t)i, page, onePageBytes);
                            std::memcpy(ringValue + onePageBytes * (size_t)i, page + onePageBytes, onePageBytes);
                        }
                        Check(cudaMemcpyAsync(key(slot), ringKey + onePageBytes * (size_t)start,
                                              runBytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
                        Check(cudaMemcpyAsync(value(slot), ringValue + onePageBytes * (size_t)start,
                                              runBytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
                    } else {
                        Check(cudaMemcpyAsync(ringKey + onePageBytes * (size_t)start, key(slot),
                                              runBytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                        Check(cudaMemcpyAsync(ringValue + onePageBytes * (size_t)start, value(slot),
                                              runBytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                    }
                    start = end;
                }
                Check(cudaStreamSynchronize(cudaStreamPerThread));
                if constexpr (!restore) {
                    for (int i = 0; i < count; ++i) {
                        uint8_t *page = const_cast<uint8_t *>((const uint8_t*)batch[base + i].data);
                        std::memcpy(page, ringKey + onePageBytes * (size_t)i, onePageBytes);
                        std::memcpy(page + onePageBytes, ringValue + onePageBytes * (size_t)i, onePageBytes);
                    }
                }
            } catch (...) {
                // Callback buffers must outlive every submitted transfer,
                // including when a later submission in the batch fails.
                cudaStreamSynchronize(cudaStreamPerThread);
                throw;
            }
        }

        template<class T>
        void Run(const Data &rawQ, const Data &rawK, const Data &k, const Data &v,
                 Data &keyCache, Data &valueCache) {
            int oldTokens = pages.Tokens(), tokens = k.dims[1];
            if (tokens <= 0 || tokens > config.prefillTokens || tokens > config.maxTokens - oldTokens) {
                throw std::invalid_argument("KVMem: append exceeds configured context/chunk size");
            }
            int oldPages = (oldTokens + pageLen - 1) / pageLen;
            // A shared prefill selection must not see a later query row. Only
            // query[0] and already committed keys participate in retrieval.
            // Ordinary decode reuses retrieval within a committed query page.
            // Speculative verification uses a configurable committed-token
            // interval. Provisional/rejected rows never advance this interval.
            // A policy change also invalidates the previous query bucket.
            const int queryInterval = pages.InTransaction() ? config.retrievalInterval : pageLen;
            const bool prefillRefresh = !pages.InTransaction() && (tokens > 1 || previousAppendTokens > 1);
            if (oldTokens + tokens > config.residentTokens && oldPages > 0 &&
                (!plan->ValidFor(oldTokens / queryInterval, queryInterval) || prefillRefresh)) {
                if (plan->IsOwner(device)) {
                    int qHeads = rawQ.dims[2];
                    EnsureSumsPages(oldPages);
                    EnsureScorePages(oldPages, qHeads);
                    ScorePages<T><<<dim3(oldPages, qHeads), 256, 0, cudaStreamPerThread>>>(
                        (const T*)rawQ.cudaData, (const float*)sumsData, (float*)scoreData,
                        oldPages, oldTokens, pageLen, qHeads, heads, dim, config.retrievalNormPower);
                    Check(cudaGetLastError());
                    std::vector<float> logits((size_t)qHeads * oldPages);
                    const size_t logitsBytes = logits.size() * sizeof(float);
                    KvMemStagingLease lease(KvMemScoreStaging(device), device, logitsBytes);
                    if (lease.Ok()) {
                        Check(cudaMemcpyAsync(lease.Data(), scoreData, logitsBytes,
                                              cudaMemcpyDeviceToHost, cudaStreamPerThread));
                        Check(cudaStreamSynchronize(cudaStreamPerThread));
                        std::memcpy(logits.data(), lease.Data(), logitsBytes);
                    } else {
                        Check(cudaMemcpyAsync(logits.data(), scoreData, logitsBytes,
                                              cudaMemcpyDeviceToHost, cudaStreamPerThread));
                        Check(cudaStreamSynchronize(cudaStreamPerThread));
                    }
                    std::vector<float> fresh(oldPages, 0.0f);
                    for (int h = 0; h < qHeads; ++h) {
                        float *row = logits.data() + (size_t)h * oldPages;
                        float peak = *std::max_element(row, row + oldPages);
                        double total = 0;
                        for (int p = 0; p < oldPages; ++p) { row[p] = std::exp(row[p] - peak); total += row[p]; }
                        if (!(total > 0) || !std::isfinite(total)) throw std::runtime_error("KVMem: non-finite retrieval scores");
                        for (int p = 0; p < oldPages; ++p) fresh[p] += row[p] / (total * qHeads);
                    }
                    // Publish for every other device before this one continues.
                    plan->Store(fresh, oldTokens / queryInterval, queryInterval);
                    consumedStamp = plan->Stamp();
                } else {
                    // The owner runs first and reaches the same decision, so a
                    // stamp this device has not consumed must be waiting.
                    const uint64_t stamp = plan->Stamp();
                    if (stamp == consumedStamp) {
                        throw std::runtime_error("KVMem: the owning device did not publish retrieval scores");
                    }
                    consumedStamp = stamp;
                }
            }
            const std::vector<float> scores = plan->Scores();
            pages.Append(tokens, scores,
                [&](const std::vector<KvMemPages::ReadPage> &batch) { TransferPages<false>(batch); },
                [&](const std::vector<KvMemPages::WritePage> &batch) { TransferPages<true>(batch); });
            int first = oldTokens / pageLen, last = (oldTokens + tokens - 1) / pageLen;
            EnsureSumsPages(last + 1);
            std::vector<int> appendSlots;
            for (int p = first; p <= last; ++p) appendSlots.push_back(pages.Slot(p));
            Allocate(writeSlots, DataType::INT32, {(int)appendSlots.size()});
            Check(cudaMemcpyAsync(writeSlots.cudaData, appendSlots.data(), appendSlots.size() * sizeof(int),
                                  cudaMemcpyHostToDevice, cudaStreamPerThread));
            if (pages.InTransaction()) provisionalKeys.CopyFrom(rawK);
            else Index<T>(rawK, oldTokens, tokens);
            AppendKV<T><<<((size_t)heads * tokens * dim + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                (const T*)k.cudaData, (const T*)v.cudaData, (T*)poolK.cudaData, (T*)poolV.cudaData,
                (const int*)writeSlots.cudaData, oldTokens, tokens, pageLen, heads, dim);
            Check(cudaGetLastError());
            // The pageable host metadata must remain valid until its copy ends.
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            previousAppendTokens = tokens;
            Publish(keyCache, valueCache);
        }

    public:
        CudaKvMemCache(const KvMemConfig &config, const Data &k,
                       const std::shared_ptr<KvMemScorePlan> &plan)
            : config(config), device(FastllmCudaGetDevice()), pageLen(GetPageLen()),
              heads(k.dims[0]), dim(k.dims[2]), plan(plan), type(k.dataType),
              onePageBytes((size_t)pageLen * heads * dim * 2), pages(config, pageLen, onePageBytes * 2) {
            for (auto *pool : {&poolK, &poolV}) {
                pool->type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
                pool->pageLen = pageLen;
                pool->maxPages = config.residentTokens / pageLen;
                Allocate(*pool, type, {pool->maxPages, pageLen, heads, dim});
            }
        }
        ~CudaKvMemCache() override {
            int previous = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device);
            cudaStreamSynchronize(cudaStreamPerThread);
            if (sumsData != nullptr) {
                FastllmCudaFree(sumsData);
                sumsData = nullptr;
            }
            if (scoreData != nullptr) {
                FastllmCudaFree(scoreData);
                scoreData = nullptr;
            }
            FastllmCudaSetDevice(previous);
        }
        const KvMemStats& Stats() const override { return pages.Stats(); }
        std::shared_ptr<KvMemScorePlan> Plan() const override { return plan; }
        void BeginTransaction(int tokens) override {
            if (failed || FastllmCudaGetDevice() != device) {
                throw std::runtime_error("KVMem: transaction on an unusable cache or wrong device");
            }
            pages.BeginTransaction(tokens);
        }
        void FinishTransaction(int acceptedTokens, Data &keyCache, Data &valueCache) override {
            if (failed || FastllmCudaGetDevice() != device ||
                keyCache.kvMemCache.get() != this || valueCache.kvMemCache.get() != this) {
                throw std::runtime_error("KVMem: invalid transaction owner/device or failed cache");
            }
            // Validate the prefix before launching an irreversible index update.
            pages.FinishTransaction(acceptedTokens);
            try {
                const int committedBase = pages.Tokens() - acceptedTokens;
                if (type == DataType::FLOAT16) {
                    Index<half>(provisionalKeys, committedBase, acceptedTokens);
                } else {
                    Index<__nv_bfloat16>(provisionalKeys, committedBase, acceptedTokens);
                }
                Check(cudaStreamSynchronize(cudaStreamPerThread));
                provisionalKeys.FreeSpace();
                // A fully cancelled round must not retain a selection derived
                // from its uncommitted first query. Partial commits keep row 0.
                if (!acceptedTokens) {
                    plan->Reset();
                }
                previousAppendTokens = 0;
                Publish(keyCache, valueCache);
            } catch (...) {
                failed = true;
                throw;
            }
        }
        void Append(const Data &rawQ, const Data &rawK, const Data &k, const Data &v,
                    Data &keyCache, Data &valueCache) override {
            if (FastllmCudaGetDevice() != device || k.dataType != type || v.dataType != type ||
                rawQ.dataType != type || rawK.dataType != type || keyCache.dataType != type || valueCache.dataType != type ||
                k.dims.size() != 3 || k.dims != v.dims || k.dims[0] != heads || k.dims[2] != dim ||
                rawK.dims != std::vector<int>({1, k.dims[1], heads, dim}) || rawQ.dims.size() != 4 ||
                rawQ.dims[0] != 1 || rawQ.dims[1] != k.dims[1] || rawQ.dims[2] <= 0 ||
                rawQ.dims[2] % heads != 0 || rawQ.dims[3] != dim) {
                throw std::invalid_argument("KVMem: unsupported device, dtype or Q/K/V shape");
            }
            for (const Data *data : {&rawQ, &rawK, &k, &v}) {
                if (data->dataDevice != DataDevice::CUDA || !data->cudaData || data->multiDeviceData) {
                    throw std::invalid_argument("KVMem: Q/K/V must be contiguous single-device CUDA tensors");
                }
            }
            if (failed) throw std::runtime_error("KVMem: cache unusable after a failed CUDA append");
            try {
                if (type == DataType::FLOAT16) Run<half>(rawQ, rawK, k, v, keyCache, valueCache);
                else if (type == DataType::BFLOAT16) Run<__nv_bfloat16>(rawQ, rawK, k, v, keyCache, valueCache);
                else throw std::invalid_argument("KVMem: only FP16/BF16 KV is supported");
            } catch (...) {
                failed = true;
                throw;
            }
        }
        // Install a committed prefix so this cache resumes without re-prefilling
        // those tokens. Only the cold host backups and the retrieval index are
        // restored; the GPU pools stay empty and the first Append pulls the
        // selected pages back over PCIe via TransferPages<true>.
        bool Seed(const KvMemPrefixData &data) override {
            if (failed || FastllmCudaGetDevice() != device) return false;
            if (data.tokens <= 0 || data.tokens > config.maxTokens ||
                pages.Tokens() != 0) {
                return false;
            }
            const int pagesCount = (data.tokens + pageLen - 1) / pageLen;
            if ((int)data.hostPages.size() != pagesCount) return false;
            if (!pages.Seed(data.hostPages, data.tokens)) return false;
            const int sumsPages = std::min(data.sumsPages, pagesCount);
            const size_t sumsBytes = (size_t)sumsPages * heads * dim * sizeof(float);
            size_t covered = 0;
            for (const KvMemPrefixSums &slice : data.sums) covered += slice.bytes;
            if (sumsBytes > 0 && covered < sumsBytes) { failed = true; return false; }
            if (sumsBytes > 0) {
                try {
                    EnsureSumsPages(sumsPages);
                    size_t offset = 0;
                    for (const KvMemPrefixSums &slice : data.sums) {
                        if (offset >= sumsBytes) break;
                        const size_t bytes = std::min(slice.bytes, sumsBytes - offset);
                        if (slice.data == nullptr || slice.data->size() < bytes) {
                            failed = true;
                            return false;
                        }
                        Check(cudaMemcpyAsync((uint8_t*)sumsData + offset,
                                              slice.data->data(), bytes,
                                              cudaMemcpyHostToDevice, cudaStreamPerThread));
                        offset += bytes;
                    }
                    Check(cudaStreamSynchronize(cudaStreamPerThread));
                } catch (...) { failed = true; return false; }
            }
            plan->Reset();
            previousAppendTokens = 0;
            return true;
        }
        // Dump the committed prefix for cross-request reuse. Reuses the pages of
        // `reuse` (the previous, shorter dump of the same request) so successive
        // dumps only read back the newly completed pages. Does not mutate the
        // cache, so it may run at any time before release.
        bool Export(KvMemPrefixData &data,
                    const KvMemPrefixData *reuse = nullptr) const override {
            if (failed || FastllmCudaGetDevice() != device) return false;
            const int pagesCount = pages.TokenPages();
            if (pagesCount <= 0) return false;
            data.tokens = pages.Tokens();
            data.hostPages.assign(pagesCount, KvMemSharedBytes());
            // 先把不需要读回的页(与上一份 dump 共享的、已有冷备份的)填掉, 剩下的
            // resident 页走 pinned 中转批量读回.
            std::vector<int> residentPages, residentSlots;
            for (int p = 0; p < pagesCount; ++p) {
                // Committed full pages never change, so the common prefix can be
                // shared with the previous dump instead of read back again. The
                // final partial page grows with new tokens, so it is never
                // shared.
                if (reuse != nullptr && p < (int)reuse->hostPages.size() &&
                    reuse->hostPages[p] != nullptr &&
                    (int64_t)(p + 1) * pageLen <= reuse->tokens) {
                    data.hostPages[p] = reuse->hostPages[p];
                    continue;
                }
                if (const KvMemSharedBytes &cold = pages.ColdPage(p)) {
                    data.hostPages[p] = cold;
                    continue;
                }
                // Resident pages (including the partial tail) are read back now.
                const int slot = pages.PageSlot(p);
                if (slot < 0) return false;
                residentPages.push_back(p);
                residentSlots.push_back(slot);
            }
            const size_t slotBytes = onePageBytes * 2;
            for (size_t base = 0; base < residentPages.size(); ) {
                const int batch = (int)std::min((size_t)KvMemStagingSlots,
                                                residentPages.size() - base);
                KvMemStagingLease lease(KvMemPageStaging(device), device, slotBytes * (size_t)batch);
                if (!lease.Ok()) return false;
                for (int i = 0; i < batch; ++i) {
                    const int slot = residentSlots[base + i];
                    uint8_t *dst = lease.Slot(slotBytes, (size_t)i);
                    try {
                        Check(cudaMemcpyAsync(dst,
                            (uint8_t*)poolK.cudaData + (size_t)slot * onePageBytes,
                            onePageBytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                        Check(cudaMemcpyAsync(dst + onePageBytes,
                            (uint8_t*)poolV.cudaData + (size_t)slot * onePageBytes,
                            onePageBytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                    } catch (...) {
                        return false;
                    }
                }
                try {
                    Check(cudaStreamSynchronize(cudaStreamPerThread));
                } catch (...) {
                    return false;
                }
                for (int i = 0; i < batch; ++i) {
                    std::shared_ptr<KvMemBytes> out = std::make_shared<KvMemBytes>(slotBytes);
                    std::memcpy(out->data(), lease.Slot(slotBytes, (size_t)i), slotBytes);
                    data.hostPages[residentPages[base + i]] = out;
                }
                base += (size_t)batch;
            }
            // sums is per committed page and never changes once the page is
            // full, so the pages inherited from `reuse` are referenced instead
            // of copied; only the pages past them are read back.
            const size_t sumsRow = (size_t)heads * dim * sizeof(float);
            const size_t sumsBytes = (size_t)pagesCount * sumsRow;
            data.sumsPages = pagesCount;
            data.sums.clear();
            size_t sharedBytes = 0;
            if (reuse != nullptr && reuse->sumsPages > 0) {
                int reuseFullPages = (int)((int64_t)reuse->tokens / pageLen);
                reuseFullPages = std::min(reuseFullPages, reuse->sumsPages);
                reuseFullPages = std::min(reuseFullPages, pagesCount);
                const size_t wanted = (size_t)reuseFullPages * sumsRow;
                for (const KvMemPrefixSums &slice : reuse->sums) {
                    if (sharedBytes >= wanted || slice.data == nullptr) break;
                    const size_t bytes = std::min(slice.bytes, wanted - sharedBytes);
                    if (bytes == 0) break;
                    data.sums.push_back({slice.data, bytes});
                    sharedBytes += bytes;
                }
            }
            const size_t remainingSums = sumsBytes - sharedBytes;
            if (remainingSums > 0) {
                // Every appended page is indexed, so the buffer covers pagesCount.
                if (pagesCount > sumsCapacity) return false;
                // The pages this dump completes; a partial page is never shared
                // because its sum keeps growing until the page is full.
                std::shared_ptr<KvMemBytes> tail = std::make_shared<KvMemBytes>(remainingSums);
                try {
                    Check(cudaMemcpyAsync(tail->data(),
                        (const uint8_t*)sumsData + sharedBytes, remainingSums,
                        cudaMemcpyDeviceToHost, cudaStreamPerThread));
                    Check(cudaStreamSynchronize(cudaStreamPerThread));
                } catch (...) { return false; }
                data.sums.push_back({tail, remainingSums});
            }
            return true;
        }
    };
}

// The tensor-parallel parameter generation and attention op read the cache
// container, so mirror the owning storage's published page metadata onto it.
// Every device selects the same logical pages, so one device's view is
// authoritative. pagedKVCacheData and isPagedKVCache are deliberately left
// unset: the container owns no pool, and ~Data must not release slots through
// one it does not own.
static void KvMemPublishContainer(const Data &storage, Data &container) {
    container.dataType = storage.dataType;
    container.UpdateUnitSize();
    container.pageLen = storage.pageLen;
    container.lastPageLen = storage.lastPageLen;
    container.pageIndex = storage.pageIndex;
}

// One device's append. The cache owner is created lazily here, so the first
// append of a request also consumes its pending prefix seed. All devices of one
// layer share `plan`, so they select the same logical pages.
static void KvMemAppendLocal(const Data &rawQ, const Data &rawK, const Data &k, const Data &v,
                             Data &keyCache, Data &valueCache,
                             const std::shared_ptr<KvMemScorePlan> &plan) {
    if (!keyCache.kvMemConfig || !keyCache.kvMemConfig->enabled || k.dims.size() != 3) {
        throw std::invalid_argument("KVMem: cache is not configured");
    }
    if (!keyCache.kvMemCache) {
        if (!keyCache.pageIndex.empty() || !valueCache.pageIndex.empty()) {
            throw std::invalid_argument("KVMem: cannot attach to an existing KV cache");
        }
        keyCache.kvMemCache = std::make_shared<CudaKvMemCache>(*keyCache.kvMemConfig, k, plan);
        valueCache.kvMemCache = keyCache.kvMemCache;
        // Consume a pending prefix seed once, before the first Append. The seed
        // only carries cold host backups + retrieval index, so the appended
        // tokens below continue right after the restored prefix.
        if (keyCache.kvMemSeed) {
            if (!keyCache.kvMemCache->Seed(*keyCache.kvMemSeed)) {
                throw std::invalid_argument("KVMem: prefix seed rejected");
            }
            keyCache.kvMemSeed.reset();
        }
    }
    if (keyCache.kvMemCache != valueCache.kvMemCache) {
        throw std::invalid_argument("KVMem: key/value cache owners differ");
    }
    keyCache.kvMemCache->Append(rawQ, rawK, k, v, keyCache, valueCache);
}

void KvMemAppend(const Data &rawQ, const Data &rawK, const Data &k, const Data &v,
                 Data &keyCache, Data &valueCache) {
    // A tensor-parallel request carries one shard per device; a single-device
    // request yields exactly one view, referring to the tensor itself, so the
    // loop below reduces to the original single call.
    std::vector<KvMemDeviceView> qViews, rawKViews, kViews, vViews, keyViews, valueViews;
    KvMemDeviceViews(rawQ, qViews);
    KvMemDeviceViews(rawK, rawKViews);
    KvMemDeviceViews(k, kViews);
    KvMemDeviceViews(v, vViews);
    KvMemDeviceViews(keyCache, keyViews);
    KvMemDeviceViews(valueCache, valueViews);
    const size_t devices = keyViews.size();
    if (devices == 0 || qViews.size() != devices || rawKViews.size() != devices ||
        kViews.size() != devices || vViews.size() != devices || valueViews.size() != devices) {
        throw std::invalid_argument("KVMem: mismatched per-device Q/K/V and cache views");
    }
    // All devices of one layer must select the same logical pages, so they share
    // one retrieval plan. Reuse the layer's existing plan once its caches exist.
    std::shared_ptr<KvMemScorePlan> plan;
    for (const KvMemDeviceView &view : keyViews) {
        if (view.data->kvMemCache) {
            plan = view.data->kvMemCache->Plan();
            break;
        }
    }
    if (!plan) {
        plan = std::make_shared<KvMemScorePlan>();
    }
    for (size_t i = 0; i < devices; ++i) {
        if (keyViews[i].device != valueViews[i].device || keyViews[i].device != qViews[i].device ||
            keyViews[i].device != rawKViews[i].device || keyViews[i].device != kViews[i].device ||
            keyViews[i].device != vViews[i].device) {
            throw std::invalid_argument("KVMem: per-device view order mismatch");
        }
        KvMemAppendLocal(*qViews[i].data, *rawKViews[i].data, *kViews[i].data, *vViews[i].data,
                         *keyViews[i].data, *valueViews[i].data, plan);
    }
    if (devices > 1) {
        KvMemPublishContainer(*keyViews[0].data, keyCache);
        KvMemPublishContainer(*valueViews[0].data, valueCache);
    }
}
}
