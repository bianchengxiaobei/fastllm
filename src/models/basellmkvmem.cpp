#include "models/basellm.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

#include "utils.h"

namespace fastllm {
    void basellm::ConfigureKvMem(const KvMemConfig &config) {
        if (mainLoop != nullptr || autoWarmupRunning.load() || kvMemLocked ||
            GetPagedKVCacheManager(kvCacheId, true) != nullptr) {
            throw std::invalid_argument("KVMem must be configured before warmup or inference");
        }
        config.Validate(GetPageLen());
        if (!config.enabled) {
            kvMemConfig = config;
            return;
        }
#ifndef USE_CUDA
        throw std::invalid_argument("KVMem requires a CUDA build");
#else
        if (!SupportsKvMem() || !CanUseGPUForward() || !use_new_engine) {
            throw std::invalid_argument("KVMem is not supported by this model/backend");
        }
        if ((dataType != DataType::FLOAT16 && dataType != DataType::BFLOAT16) || kvCacheDataType != dataType) {
            throw std::invalid_argument("KVMem requires matching FP16/BF16 compute and KV dtypes");
        }
        std::vector<int> kvMemDevices;
        std::map<int, int> kvMemRatios;
        GetKvMemDevices(kvMemDevices, kvMemRatios);
        if (GetKVCacheInCPU()) {
            throw std::invalid_argument("KVMem requires CUDA KV caches");
        }
        if (kvMemDevices.empty()) {
            throw std::invalid_argument("KVMem requires a CUDA device map it can page on");
        }
        if (head_dim != 128 && head_dim != 256) {
            throw std::invalid_argument("KVMem: current model adapters require head_dim 128 or 256");
        }
        if (maxBatch > 1) throw std::invalid_argument("KVMem currently requires max_batch=1");
        if (config.maxTokens > max_positions) {
            throw std::invalid_argument("KVMem logical context exceeds the model's configured context");
        }
        if (const char *oldEngine = std::getenv("USE_OLD_ENGINE")) {
            std::string value(oldEngine);
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            if (value == "1" || value == "on") throw std::invalid_argument("KVMem requires the new GPU engine");
        }
        if (GetFastllmEnv().skipWarmup) throw std::invalid_argument("KVMem requires warmup");
        ValidateKvMemModel(config);
        uint64_t logicalTokens = (((uint64_t)config.maxTokens + GetPageLen() - 1) / GetPageLen()) * GetPageLen();
        uint64_t limit = config.hostBytes / logicalTokens;
        // hostBytes is the request-wide budget. Paging on several devices only
        // splits the KV heads, so the per-device per-token bytes still sum to the
        // single-device value; each device must fit it on its own.
        for (int device : kvMemDevices) {
            uint64_t bytesPerToken = 0;
            for (int i = 0; i < block_cnt; ++i) {
                if (!KvMemLayerEligible(i)) continue;
                uint64_t bytes = KvMemLayerDeviceBytesPerToken(i, device, kvMemRatios);
                if (!bytes || bytes > limit - bytesPerToken) {
                    throw std::invalid_argument("KVMem host budget cannot back all full-attention layers at max_tokens");
                }
                bytesPerToken += bytes;
            }
            if (!bytesPerToken) throw std::invalid_argument("KVMem: model has no eligible KV layers");
        }
        kvMemConfig = config;
        maxBatch = 1;
        saveHistoryChat = false;
        tokensLimit = promptLimit = config.maxTokens;
#endif
    }

    // KVMem owns the per-device KV storages. The paged path creates its locals
    // lazily inside the attention op, but KVMem needs them before the first
    // append so the dispatcher can page on every device, and before the prefix
    // restore so per-device seeds have somewhere to land. The container becomes a
    // metadata carrier holding no pages of its own and keeping isPagedKVCache
    // false, so ~Data never releases slots through a pool it does not own.
    void basellm::SplitKvMemCaches(const std::vector<std::pair<Data*, Data*>> &caches,
                                   int requestPrefixTokens) const {
        if (!kvMemConfig.enabled || (int)caches.size() != block_cnt) {
            return;
        }
        std::vector<int> devices;
        std::map<int, int> ratios;
        GetKvMemDevices(devices, ratios);
        if (devices.size() <= 1) {
            return;
        }
        const int pinned = KvMemClampPrefixTokens(kvMemConfig, GetPageLen(), requestPrefixTokens);
        uint64_t logicalTokens = (((uint64_t)kvMemConfig.maxTokens + GetPageLen() - 1) / GetPageLen()) * GetPageLen();
        auto requestConfig = [&](uint64_t hostBytes) {
            std::shared_ptr<KvMemConfig> config = std::make_shared<KvMemConfig>(kvMemConfig);
            config->hostBytes = hostBytes;
            config->prefixTokens = pinned;
            return config;
        };
        for (int i = 0; i < block_cnt; ++i) {
            if (!KvMemLayerEligible(i)) continue;
            for (Data *cache : {caches[i].first, caches[i].second}) {
                if (cache->kvMemConfig) continue;
                if (!cache->dims.empty() || !cache->pageIndex.empty()) {
                    throw std::invalid_argument("KVMem must start with an empty request cache");
                }
                if (!cache->multiDeviceData) {
                    cache->multiDeviceData = true;
                    cache->tpLayout = TP_LAYOUT_SHARDED;
                    cache->tpAxis = 0;
                    cache->cudaData = nullptr;
                    for (int device : devices) {
                        Data *local = new Data();
                        local->dataType = cache->dataType;
                        local->UpdateUnitSize();
                        local->dataDevice = DataDevice::CUDA;
                        local->dataDeviceIds = {device};
                        cache->multiDeviceDatas[device] = local;
                    }
                }
                // (Re)attach the budgets. A prefix restore that failed released the
                // previous configs but left the already-split storages in place, so
                // attach over the existing ones instead of rebuilding them.
                for (int device : devices) {
                    auto it = cache->multiDeviceDatas.find(device);
                    if (it == cache->multiDeviceDatas.end() || it->second == nullptr) {
                        throw std::invalid_argument("KVMem: per-device cache storage is missing");
                    }
                    if (it->second->kvMemConfig) continue;
                    // Reserve this storage's exact worst-case backing budget; the
                    // sum over devices and layers was checked against the request
                    // limit in ConfigureKvMem.
                    std::shared_ptr<KvMemConfig> config = requestConfig(
                            logicalTokens * KvMemLayerDeviceBytesPerToken(i, device, ratios));
                    if (!config->hostBytes) {
                        throw std::invalid_argument("KVMem: cannot size the per-device cache budget");
                    }
                    it->second->kvMemConfig = config;
                }
                cache->kvMemConfig = requestConfig(logicalTokens * KvMemLayerBytesPerToken(i));
            }
        }
    }

    void basellm::PrepareKvMemCaches(int batch, int devices,
                                    const std::vector<std::pair<Data*, Data*>> &caches,
                                    int requestPrefixTokens) {
        if (!kvMemConfig.enabled) return;
        if (batch != 1 || (int)caches.size() != block_cnt) {
            throw std::invalid_argument("KVMem currently requires one request");
        }
        ValidateKvMemModel(kvMemConfig);
        if (maxBatch > 1 || kvCacheDataType != dataType || saveHistoryChat) {
            throw std::invalid_argument("KVMem configuration changed after setup");
        }
        const int pageLen = GetPageLen();
        const int pinned = KvMemClampPrefixTokens(kvMemConfig, pageLen, requestPrefixTokens);
        // Rounding a fitting request up to whole pages is not a clamp, so only
        // report the cap actually shortening the request.
        const int wanted = requestPrefixTokens > 0
                ? (requestPrefixTokens + pageLen - 1) / pageLen * pageLen : 0;
        if (pinned < wanted) {
            static bool reported = false;
            if (!reported) {
                reported = true;
                printf("[Fastllm] KVMem prefix pin clamped to %d tokens (requested %d)\n",
                       pinned, wanted);
                fflush(stdout);
            }
        }
        std::vector<int> kvMemDevices;
        std::map<int, int> kvMemRatios;
        GetKvMemDevices(kvMemDevices, kvMemRatios);
        if ((int)kvMemDevices.size() != devices || kvMemDevices.empty()) {
            throw std::invalid_argument("KVMem: forward devices do not match the configured KVMem devices");
        }
        SplitKvMemCaches(caches, requestPrefixTokens);
        uint64_t logicalTokens = (((uint64_t)kvMemConfig.maxTokens + GetPageLen() - 1) / GetPageLen()) * GetPageLen();
        for (int i = 0; i < block_cnt; ++i) {
            if (!KvMemLayerEligible(i)) continue;
            // A split layer already has one storage per device, each carrying its
            // own config from SplitKvMemCaches. An unsplit layer keeps one config
            // shared by key and value, as before.
            std::shared_ptr<KvMemConfig> singleConfig;
            for (Data *cache : {caches[i].first, caches[i].second}) {
                std::vector<KvMemDeviceView> views;
                KvMemDeviceViews(*cache, views);
                if (views.size() > 1) {
                    continue;
                }
                for (const KvMemDeviceView &view : views) {
                    Data &storage = *view.data;
                    if (storage.kvMemConfig) continue;
                    if (!storage.dims.empty() || !storage.pageIndex.empty()) {
                        throw std::invalid_argument("KVMem must start with an empty request cache");
                    }
                    if (!singleConfig) {
                        singleConfig = std::make_shared<KvMemConfig>(kvMemConfig);
                        uint64_t bytes = KvMemLayerBytesPerToken(i);
                        if (!bytes) {
                            throw std::invalid_argument("KVMem: cannot size the cache budget");
                        }
                        singleConfig->hostBytes = logicalTokens * bytes;
                        singleConfig->prefixTokens = pinned;
                    }
                    storage.kvMemConfig = singleConfig;
                }
            }
        }
    }

    // Default resolver: a model that does not override this never splits KV
    // heads, so KVMem stays on the single CUDA device of a plain `cuda` /
    // `cuda:N` map. Tensor-parallel models override it to return every device
    // the forward itself pages on.
    void basellm::GetKvMemDevices(std::vector<int> &devices, std::map<int, int> &ratios) const {
        devices.clear();
        ratios.clear();
#ifdef USE_CUDA
        if (GetKVCacheInCPU() || deviceMap.size() != 1) {
            return;
        }
        std::string spec = deviceMap.begin()->first;
        std::transform(spec.begin(), spec.end(), spec.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        if (spec != "cuda" && !StartWith(spec, "cuda:")) {
            return;
        }
        std::map<int, int> parsedRatios;
        std::vector<int> parsed = ParseDeviceIds(spec, "cuda", parsedRatios);
        if (parsed.empty()) {
            parsed.push_back(0);
        }
        for (int device : parsed) {
            if (device < 0 || device == 99999 || ratios.find(device) != ratios.end()) {
                continue;
            }
            devices.push_back(device);
            ratios[device] = 1;
        }
        // The default path cannot page on more than one device; a model-level
        // split has to come from an override.
        if (devices.size() != 1) {
            devices.clear();
            ratios.clear();
        }
#endif
    }

    uint64_t basellm::KvMemLayerDeviceBytesPerToken(int layer, int device,
                                                   const std::map<int, int> &ratios) const {
        uint64_t bytes = KvMemLayerBytesPerToken(layer);
        auto it = ratios.find(device);
        if (bytes == 0 || it == ratios.end() || it->second <= 0) {
            return 0;
        }
        uint64_t total = 0;
        for (auto &entry : ratios) {
            if (entry.second > 0) {
                total += (uint64_t)entry.second;
            }
        }
        if (total == 0) {
            return 0;
        }
        return bytes * (uint64_t)it->second / total;
    }

    void basellm::WarmupKvMem() {
        ResponseContext context;
        context.Init(block_cnt, dataType, kvCacheDataType);
        std::vector<std::pair<Data*, Data*>> caches;
        for (auto &kv : context.pastKeyValues) caches.emplace_back(&kv.first, &kv.second);
        Data input(DataType::FLOAT32, {1, 2}, std::vector<float>{1, 1});
        Data positions(DataType::FLOAT32, {1, 2}, std::vector<float>{0, 1});
        std::vector<Data*> masks{nullptr}, positionIds{&positions};
        std::vector<int> lengths{2};
        std::vector<GenerationConfig> configs{GenerationConfig()};
        LastTokensManager lastTokens;
        ForwardGPU(1, input, masks, positionIds, lengths, caches, configs, lastTokens);
        tokensLimit = promptLimit = kvMemConfig.maxTokens;
        maxBatch = 1;
        kvMemLocked = true;
        printf("[Fastllm] KVMem enabled: logical=%d, resident=%d, host=%llu MiB; single-request eager decoding.\n",
               kvMemConfig.maxTokens, kvMemConfig.residentTokens,
               (unsigned long long)(kvMemConfig.hostBytes >> 20));
    }
}
