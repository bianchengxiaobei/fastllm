#ifndef FASTLLM_KVMEM_H
#define FASTLLM_KVMEM_H

#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace fastllm {
    // Model-local and opt-in. All token budgets except maxTokens are page aligned.
    struct KvMemConfig {
        bool enabled = false;
        int maxTokens = 32768;
        int residentTokens = 8192;
        int sinkTokens = 128;
        // Leading tokens that stay in every attention view regardless of their
        // retrieval score, so a chat template's system block is never dropped.
        // Zero lets the caller decide per request instead.
        int prefixTokens = 0;
        int recentTokens = 2048;
        int retrievalTokens = 4096;
        int prefillTokens = 512;
        // Retrieval score is (q . sumK) / (count * meanK^normPower * sqrt(dim)):
        // 0 scores the raw dot product (favours high-norm pages), 1 scores pure
        // cosine similarity. Applies to every recall, not per request.
        float retrievalNormPower = 0.0f;
        // Speculative retrieval refresh interval in committed tokens; 1 refreshes
        // every round. Independent of physical page size and token budgets.
        int retrievalInterval = 64;
        uint64_t hostBytes = 8ULL << 30;

        void Validate(int pageLen) const;
    };

    // Page-aligned prefix length one request may pin. A non-positive request
    // keeps the configured value; anything the resident budget cannot hold is
    // reduced to the cap, so a request can never make the page store unusable.
    int KvMemClampPrefixTokens(const KvMemConfig &config, int pageLen, int requested);

    struct KvMemStats {
        uint64_t hostBytes = 0, evictedPages = 0, restoredPages = 0;
        // Includes cached pages outside the current attention view.
        int tokens = 0, residentPages = 0;
    };

    // Immutable byte storage that is not value-initialized: a page buffer is
    // filled by a device copy right after allocation, so zeroing it first (as
    // std::vector would) only costs a full memset per page.
    class KvMemBytes {
    public:
        KvMemBytes() = default;
        explicit KvMemBytes(size_t size) : storage(new uint8_t[size]), size_(size) {}

        uint8_t *data() { return storage.get(); }
        const uint8_t *data() const { return storage.get(); }
        size_t size() const { return size_; }

    private:
        std::unique_ptr<uint8_t[]> storage;
        size_t size_ = 0;
    };

    using KvMemSharedBytes = std::shared_ptr<const KvMemBytes>;

    // One immutable slice of the page-major retrieval index. bytes is the
    // referenced prefix of data, so a shorter slice may point into a longer
    // slice's buffer.
    struct KvMemPrefixSums {
        KvMemSharedBytes data;
        size_t bytes = 0;
    };

    // Portable snapshot of one committed prefix, used to reuse KV across
    // requests. tokens is the exact committed length; hostPages[p] is the [K|V]
    // backup of logical page p (immutable once committed), shared between
    // snapshots so successive dumps only add the newly completed pages. sums
    // holds the retrieval index of the leading sumsPages pages as consecutive
    // slices; a slice inherited from a shorter dump is referenced, not copied.
    struct KvMemPrefixData {
        int tokens = 0;
        std::vector<KvMemSharedBytes> hostPages;
        std::vector<KvMemPrefixSums> sums;
        int sumsPages = 0;
    };

    // Backend-independent append-only page store. Logical IDs retain absolute
    // positions; slots are recycled only when needed. Unselected resident pages
    // are cached by recency; they do not enter the attention view. The view is
    // chronological and contains the current query and its recent causal history.
    // A full immutable page is backed up only once. Batch callbacks must finish
    // all transfers before returning (also on failure). Backups complete before
    // any slot is reused. This object belongs to one request/layer.
    class KvMemPages {
    public:
        struct ReadPage { int slot; uint8_t *data; };
        struct WritePage { int slot; const uint8_t *data; };
        using ReadPages = std::function<void(const std::vector<ReadPage>&)>;
        using WritePages = std::function<void(const std::vector<WritePage>&)>;

        KvMemPages(const KvMemConfig &config, int pageLen, size_t pageBytes);
        const std::vector<int>& Append(int tokens, const std::vector<float> &scores,
                                      const ReadPages &read, const WritePages &write);
        // Install a committed prefix as cold host backups. tokens is the exact
        // committed length (need not be page aligned); hostPages[p] is the
        // immutable [K|V] backup of logical page p for p in [0, ceil(tokens/
        // pageLen)). The final page may be partial. Backups are SHARED (the
        // store keeps them for later requests) and every slot stays non-resident,
        // so the first Append restores the selected pages on demand. Returns
        // false (state untouched) on mismatch.
        bool Seed(const std::vector<KvMemSharedBytes> &hostPages,
                  int tokens);
        // Export helpers for cross-request reuse. TokenPages() is ceil(Tokens()/
        // pageLen). ColdPage returns the shared cold backup of a page, or nullptr
        // when the page is GPU resident (the caller then reads it back via
        // PageSlot()). PageBytes is the [K|V] byte size of one page.
        int TokenPages() const {
            return (stats.tokens + pageLen - 1) / pageLen;
        }
        const KvMemSharedBytes &ColdPage(int logicalPage) const;
        int PageSlot(int logicalPage) const;
        size_t PageBytes() const { return pageBytes; }
        // One provisional append per transaction. Finish keeps an accepted
        // prefix (0 rolls back); the writer/recent tail stays resident.
        void BeginTransaction(int tokens);
        void FinishTransaction(int acceptedTokens);
        bool InTransaction() const { return transactionBase >= 0; }
        const std::vector<int>& Visible() const { return visible; }
        int Slot(int logicalPage) const;
        int Tokens() const { return stats.tokens; }
        const KvMemStats& Stats() const { return stats; }

    private:
        KvMemConfig config;
        int pageLen, capacity;
        size_t pageBytes;
        bool failed = false;
        int transactionBase = -1, transactionTokens = 0;
        std::vector<int> slots, visible;
        std::vector<uint64_t> lastUsed;
        uint64_t useSequence = 0;
        std::vector<KvMemSharedBytes> host;
        KvMemStats stats;
    };

    class Data;
    // Detach a borrowed sparse view without releasing its slots into a global
    // page pool. Returns false for an ordinary cache; dimensions stay caller-owned.
    bool ReleaseKvMemCache(Data &cache);

    // One device's view of a possibly tensor-parallel tensor. A tensor that is
    // not split yields exactly one view referring to the tensor itself, with
    // device == -1 (a flat tensor has no split identity); a single-device caller
    // therefore iterates once with byte-identical behavior. For a split tensor
    // the views carry real device ids, ordered by ascending device id, letting
    // callers pair the Q/K/V arguments with the key/value caches positionally.
    struct KvMemDeviceView {
        int device = -1;
        Data *data = nullptr;
    };
    void KvMemDeviceViews(const Data &cache, std::vector<KvMemDeviceView> &views);

    // Device-wise copy of `src` into `dst`. A tensor-parallel source has no
    // single backing buffer (Data::CopyFrom only handles flat tensors), so its
    // per-device sharding is mirrored into `dst` with real per-device copies; a
    // plain source is an ordinary Data::CopyFrom. `dst` must be fresh, and owns
    // the per-device storages it creates.
    void KvMemCopyDeviceData(const Data &src, Data &dst);

    // Speculative transactions fan out over the cache's device views: every
    // device reserves, then commits or rolls back the same length, and a cache
    // that is not split behaves exactly like the single-device call. The caller
    // owns the all-or-nothing policy across layers and devices.
    void KvMemBeginTransaction(Data &keyCache, int tokens);
    void KvMemFinishTransaction(Data &keyCache, Data &valueCache, int acceptedTokens);

    // One request's layer shares this plan so that every device selects the same
    // logical pages: the tensor-parallel attention op replicates a single
    // pageIndex across devices. The first device to append owns the retrieval
    // computation and publishes the scores; the others read them. A reader that
    // needs a refresh must find a stamp it has not consumed, which the owner's
    // identical, same-pass refresh decision guarantees. Views are ordered by
    // ascending device id, so the owner always runs first.
    class KvMemScorePlan {
    public:
        bool IsOwner(int device) {
            std::lock_guard<std::mutex> guard(mutex);
            if (owner < 0) {
                owner = device;
            }
            return owner == device;
        }

        bool ValidFor(int bucket, int interval) {
            std::lock_guard<std::mutex> guard(mutex);
            return valid && this->bucket == bucket && this->interval == interval;
        }

        std::vector<float> Scores() {
            std::lock_guard<std::mutex> guard(mutex);
            return scores;
        }

        uint64_t Stamp() {
            std::lock_guard<std::mutex> guard(mutex);
            return stamp;
        }

        void Store(const std::vector<float> &value, int bucket, int interval) {
            std::lock_guard<std::mutex> guard(mutex);
            scores = value;
            this->bucket = bucket;
            this->interval = interval;
            valid = true;
            ++stamp;
        }

        void Reset() {
            std::lock_guard<std::mutex> guard(mutex);
            scores.clear();
            bucket = -1;
            interval = 0;
            valid = false;
        }

    private:
        std::mutex mutex;
        int owner = -1;
        bool valid = false;
        int bucket = -1, interval = 0;
        uint64_t stamp = 0;
        std::vector<float> scores;
    };

    class KvMemCache {
    public:
        virtual ~KvMemCache() = default;
        virtual const KvMemStats& Stats() const = 0;
        // Retrieval plan shared by this layer's per-device caches; null when the
        // backend keeps its scores private (single device).
        virtual std::shared_ptr<KvMemScorePlan> Plan() const { return nullptr; }
        virtual void BeginTransaction(int tokens) = 0;
        // Commit only accepted normalized keys to the retrieval index and
        // republish both attention views. May finish with 0 before Append.
        virtual void FinishTransaction(int acceptedTokens, Data &keyCache, Data &valueCache) = 0;
        // rawQ/rawK: normalized pre-RoPE [1,T,H,D]; q/k/v: post-RoPE
        // [H,T,D]. KV and query types must match (FP16 or BF16).
        virtual void Append(const Data &rawQ, const Data &rawK,
                            const Data &k, const Data &v,
                            Data &keyCache, Data &valueCache) = 0;
        // Cross-request prefix reuse. Seed installs a committed prefix as cold
        // host backups (GPU pool stays empty; the first Append restores the
        // selected pages). Export dumps the cold backups plus the retrieval
        // index; when reuse is non-null its pages are shared for the common
        // prefix so successive dumps stay incremental. Both must run on a fresh
        // cache before any Append.
        virtual bool Seed(const KvMemPrefixData &data) { (void)data; return false; }
        virtual bool Export(KvMemPrefixData &data, const KvMemPrefixData *reuse = nullptr) const {
            (void)data;
            (void)reuse;
            return false;
        }
    };

#ifdef USE_CUDA
    // Publishes a non-owning compact paged-attention view into key/valueCache.
    // RoPE must use absolute positions. Supports ordinary causal MHA/GQA;
    // sliding-window attention and position-dependent biases need another view.
    void KvMemAppend(const Data &rawQ, const Data &rawK, const Data &k, const Data &v,
                     Data &keyCache, Data &valueCache);
#endif
}
#endif
