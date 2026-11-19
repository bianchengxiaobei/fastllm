#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "utils/op_trace_archive.h"
#include "utils/op_trace_types.h"

namespace fastllm {
namespace op_trace {

struct ThreadPhase {
    int phase = (int)PhaseKind::None;
    int request = -1;
    int token = -1;
    int layer = -1;
    const char *region = "";
    uint64_t lastEndNs = 0;
    int phaseDepth = 0;
    int opDepth = 0;
};

inline ThreadPhase &TlsPhase() {
    static thread_local ThreadPhase state;
    return state;
}

inline bool Enabled() noexcept;
inline bool GapTrackingEnabled() noexcept;
inline void FlushGap(uint64_t nowNs) noexcept;

// Host-side code region used to label the wall-clock holes between traced
// operators.  Entering/leaving a region flushes the current hole so the
// untraced time is attributed to the block that actually spent it.
class RegionScope {
public:
    explicit RegionScope(const char *name) noexcept {
        ThreadPhase &state = TlsPhase();
        previousRegion = state.region;
        active = name != nullptr && name[0] != '\0' && Enabled() &&
                 GapTrackingEnabled() &&
                 state.phase != (int)PhaseKind::None;
        if (!active) {
            return;
        }
        FlushGap(NowNs());
        state.region = name;
    }

    ~RegionScope() {
        if (!active) {
            return;
        }
        FlushGap(NowNs());
        TlsPhase().region = previousRegion;
    }

    RegionScope(const RegionScope &) = delete;
    RegionScope &operator=(const RegionScope &) = delete;

private:
    const char *previousRegion = "";
    bool active = false;
};

// Keeps the layer field on synthetic gap events in sync with the model loop.
class LayerScope {
public:
    explicit LayerScope(int value) noexcept {
        ThreadPhase &state = TlsPhase();
        previousLayer = state.layer;
        active = Enabled() && GapTrackingEnabled() &&
                 state.phase != (int)PhaseKind::None;
        if (!active) {
            return;
        }
        FlushGap(NowNs());
        state.layer = value;
    }

    ~LayerScope() {
        if (!active) {
            return;
        }
        FlushGap(NowNs());
        TlsPhase().layer = previousLayer;
    }

    LayerScope(const LayerScope &) = delete;
    LayerScope &operator=(const LayerScope &) = delete;

private:
    int previousLayer = -1;
    bool active = false;
};

// One Executor::Run call.  Marks the operator interval so region scopes
// nested inside device code do not emit overlapping gap events.
class HostOpScope {
public:
    explicit HostOpScope(uint64_t startNs) noexcept {
        if (!Enabled()) {
            return;
        }
        FlushGap(startNs);
        ++TlsPhase().opDepth;
        active = true;
    }

    ~HostOpScope() {
        if (!active) {
            return;
        }
        ThreadPhase &state = TlsPhase();
        if (state.opDepth > 0) {
            --state.opDepth;
        }
    }

    HostOpScope(const HostOpScope &) = delete;
    HostOpScope &operator=(const HostOpScope &) = delete;

private:
    bool active = false;
};

class PhaseScope {
public:
    PhaseScope(PhaseKind phase, int token = -1, int request = -1) noexcept {
        ThreadPhase &state = TlsPhase();
        previousPhase = state.phase;
        previousRequest = state.request;
        previousToken = state.token;
        previousLayer = state.layer;
        previousRegion = state.region;
        if (state.phaseDepth == 0) {
            state.lastEndNs = NowNs();
            state.layer = -1;
            state.region = "";
            state.opDepth = 0;
        }
        state.phase = (int)phase;
        state.request = request;
        state.token = token;
        ++state.phaseDepth;
    }

    ~PhaseScope() {
        ThreadPhase &state = TlsPhase();
        FlushGap(NowNs());
        if (state.phaseDepth > 0) {
            --state.phaseDepth;
        }
        if (state.phaseDepth == 0) {
            state.lastEndNs = 0;
            state.layer = -1;
            state.region = "";
            state.opDepth = 0;
        }
        state.phase = previousPhase;
        state.request = previousRequest;
        state.token = previousToken;
        state.layer = previousLayer;
        state.region = previousRegion;
    }

    PhaseScope(const PhaseScope &) = delete;
    PhaseScope &operator=(const PhaseScope &) = delete;

private:
    int previousPhase = (int)PhaseKind::None;
    int previousRequest = -1;
    int previousToken = -1;
    int previousLayer = -1;
    const char *previousRegion = "";
};


inline size_t ParseEnvCapacity() {
    const char *value = std::getenv("FASTLLM_OP_TRACE_CAPACITY");
    if (value == nullptr || value[0] == '\0') {
        return 65536;
    }
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value) {
        return 65536;
    }
    if (parsed < 64) {
        parsed = 64;
    }
    if (parsed > 1000000ULL) {
        parsed = 1000000ULL;
    }
    return (size_t)parsed;
}

// Prefill events are pinned out of the ring before eviction so a long decode
// cannot push them out of the live view.  0 disables pinning.
inline size_t ParseEnvPrefillPin() {
    const char *value = std::getenv("FASTLLM_OP_TRACE_PREFILL_PIN");
    if (value == nullptr || value[0] == '\0') {
        return 131072;
    }
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value) {
        return 131072;
    }
    if (parsed > 1000000ULL) {
        parsed = 1000000ULL;
    }
    return (size_t)parsed;
}

inline bool GapTrackingEnabled() noexcept {
    static const bool enabled = []() {
        const char *value = std::getenv("FASTLLM_OP_TRACE_GAPS");
        if (value == nullptr || value[0] == '\0') {
            return true;
        }
        return ParseEnvEnabled(value);
    }();
    return enabled;
}

inline uint64_t GapThresholdNs() noexcept {
    static const uint64_t threshold = []() -> uint64_t {
        const char *value = std::getenv("FASTLLM_OP_TRACE_GAP_US");
        if (value == nullptr || value[0] == '\0') {
            return 20000ULL;
        }
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end == value) {
            return 20000ULL;
        }
        return parsed * 1000ULL;
    }();
    return threshold;
}

inline bool TensorIoEnabled() noexcept {
    static const bool enabled = []() {
        const char *value = std::getenv("FASTLLM_OP_TRACE_IO");
        if (value == nullptr || value[0] == '\0') {
            return true;
        }
        return ParseEnvEnabled(value);
    }();
    return enabled;
}

inline ArchiveConfig MakeArchiveConfig() {
    return ArchiveConfigFromEnv(TensorIoEnabled(), GapTrackingEnabled(),
                                GapThresholdNs() / 1000ULL);
}

class Store {
public:
    Store() : capacity(ParseEnvCapacity()),
              pinnedCapacity(ParseEnvPrefillPin()),
              enabled(ParseEnvEnabled(std::getenv("FASTLLM_OP_TRACE"))) {
        ArchiveConfigure(MakeArchiveConfig());
    }

    bool Enabled() const {
        return enabled.load(std::memory_order_relaxed);
    }

    void SetEnabled(bool value) {
        bool previous = enabled.exchange(value, std::memory_order_relaxed);
        if (value) {
            ArchiveEnsureSession();
        } else if (previous) {
            ArchiveFlush(true);
        }
    }

    size_t Capacity() const {
        return capacity;
    }

    size_t PinnedCapacity() const {
        return pinnedCapacity;
    }

    void Clear() noexcept {
        try {
            std::lock_guard<std::mutex> guard(mutex);
            events.clear();
            pinned.clear();
            dropped = 0;
            nextSeq = 1;
            ++epoch;
            AggregateClear();
            if (enabled.load(std::memory_order_relaxed)) {
                ArchiveRotateSession();
            }
        } catch (...) {
        }
    }

    void Record(int layer,
                const std::string &op,
                const std::string &device,
                const std::string &weight,
                const std::string &io,
                uint64_t startNs,
                uint64_t durationNs,
                const Timing &timing = Timing(),
                bool gap = false) noexcept {
        if (!Enabled()) {
            return;
        }
        try {
            Event event;
            event.startNs = startNs;
            event.durationNs = durationNs;
            event.threadId = (uint64_t)std::hash<std::thread::id>{}(
                std::this_thread::get_id());
            ThreadPhase &phase = TlsPhase();
            event.phase = PhaseName(phase.phase);
            event.request = phase.request;
            event.token = phase.token;
            event.layer = layer;
            event.gap = gap;
            event.op = op;
            event.device = device;
            event.weight = weight;
            event.io = io;
            event.transferNs = timing.transferNs;
            event.reshapeNs = timing.reshapeNs;
            event.runNs = timing.runNs;
            event.syncNs = timing.syncNs;
            const uint64_t endNs = startNs + durationNs;
            if (endNs > phase.lastEndNs) {
                phase.lastEndNs = endNs;
            }

            std::lock_guard<std::mutex> guard(mutex);
            if (events.size() >= capacity) {
                Event &oldest = events.front();
                if (pinnedCapacity > 0 && pinned.size() < pinnedCapacity &&
                    oldest.phase == "prefill") {
                    pinned.push_back(std::move(oldest));
                } else {
                    ++dropped;
                }
                events.pop_front();
            }
            event.seq = nextSeq++;
            events.push_back(std::move(event));
            const Event &stored = events.back();
            if (enabled.load(std::memory_order_relaxed)) {
                ArchiveAppend(stored);
                AggregateRecord(stored);
            }
        } catch (...) {
        }
    }

    std::string SnapshotJson(uint64_t afterSeq, size_t maxEvents) noexcept {
        try {
            return SnapshotJsonImpl(afterSeq, maxEvents);
        } catch (...) {
            return "{\"schema\":1,\"error\":\"snapshot failed\",\"events\":[]}";
        }
    }

private:
    std::string SnapshotJsonImpl(uint64_t afterSeq, size_t maxEvents) {
        std::vector<Event> selected;
        uint64_t firstSeq = 0, lastSeq = 0, currentEpoch = 0;
        uint64_t currentDropped = 0, currentSize = 0, currentPinned = 0, lost = 0;
        std::string sessionId;
        std::string archiveStatus;
        {
            std::lock_guard<std::mutex> guard(mutex);
            currentEpoch = epoch;
            currentDropped = dropped;
            currentSize = (uint64_t)events.size();
            currentPinned = (uint64_t)pinned.size();
            sessionId = ArchiveSessionId();
            archiveStatus = ArchiveStatusJson();
            lastSeq = nextSeq > 0 ? nextSeq - 1 : 0;
            if (!pinned.empty()) {
                firstSeq = pinned.front().seq;
            } else if (!events.empty()) {
                firstSeq = events.front().seq;
            } else {
                firstSeq = nextSeq;
            }
            if (maxEvents == 0) {
                maxEvents = 4096;
            }
            selected.reserve(std::min(maxEvents, pinned.size() + events.size()));

            auto pinnedIt = std::upper_bound(
                pinned.begin(), pinned.end(), afterSeq,
                [](uint64_t value, const Event &event) { return value < event.seq; });
            for (; pinnedIt != pinned.end() && selected.size() < maxEvents;
                 ++pinnedIt) {
                selected.push_back(*pinnedIt);
            }
            auto eventsIt = std::upper_bound(
                events.begin(), events.end(), afterSeq,
                [](uint64_t value, const Event &event) { return value < event.seq; });
            for (; eventsIt != events.end() && selected.size() < maxEvents;
                 ++eventsIt) {
                selected.push_back(*eventsIt);
            }
            if (!selected.empty()) {
                uint64_t expected = afterSeq + 1;
                for (const Event &event : selected) {
                    if (event.seq > expected) {
                        lost = 1;
                        break;
                    }
                    expected = event.seq + 1;
                }
            }
        }

        std::string out;
        out.reserve(256 + selected.size() * 256);
        out += "{\"schema\":1,\"epoch\":";
        out += std::to_string(currentEpoch);
        out += ",\"enabled\":";
        out += Enabled() ? "true" : "false";
        out += ",\"capacity\":";
        out += std::to_string(capacity);
        out += ",\"size\":";
        out += std::to_string(currentSize);
        out += ",\"pinned\":";
        out += std::to_string(currentPinned);
        out += ",\"pinned_capacity\":";
        out += std::to_string(pinnedCapacity);
        out += ",\"count\":";
        out += std::to_string((uint64_t)selected.size());
        out += ",\"first_seq\":";
        out += std::to_string(firstSeq);
        out += ",\"last_seq\":";
        out += std::to_string(lastSeq);
        out += ",\"dropped\":";
        out += std::to_string(currentDropped);
        out += ",\"lost\":";
        out += lost ? "true" : "false";
        out += ",\"session_id\":";
        AppendJsonString(out, sessionId);
        out += ",\"archive\":";
        out += archiveStatus;
        out += ",\"events\":[";
        for (size_t i = 0; i < selected.size(); ++i) {
            const Event &event = selected[i];
            if (i) {
                out += ',';
            }
            out += "{\"seq\":";
            out += std::to_string(event.seq);
            out += ",\"t_us\":";
            out += std::to_string(event.startNs / 1000);
            out += ",\"dur_ns\":";
            out += std::to_string(event.durationNs);
            out += ",\"dur_us\":";
            out += std::to_string(event.durationNs / 1000);
            out += ",\"tid\":";
            out += std::to_string(event.threadId);
            out += ",\"layer\":";
            out += std::to_string(event.layer);
            out += ",\"phase\":";
            AppendJsonString(out, event.phase);
            if (event.gap) {
                out += ",\"gap\":1";
            }
            out += ",\"req\":";
            out += std::to_string(event.request);
            out += ",\"token\":";
            out += std::to_string(event.token);
            out += ",\"op\":";
            AppendJsonString(out, event.op);
            out += ",\"dev\":";
            AppendJsonString(out, event.device);
            out += ",\"weight\":";
            AppendJsonString(out, event.weight);
            out += ",\"io\":";
            AppendJsonString(out, event.io);
            out += ",\"xfer_us\":";
            out += std::to_string(event.transferNs / 1000);
            out += ",\"reshape_us\":";
            out += std::to_string(event.reshapeNs / 1000);
            out += ",\"run_us\":";
            out += std::to_string(event.runNs / 1000);
            out += ",\"sync_us\":";
            out += std::to_string(event.syncNs / 1000);
            out += '}';
        }
        out += "]}";
        return out;
    }

private:
    size_t capacity;
    size_t pinnedCapacity;
    std::atomic<bool> enabled;
    std::mutex mutex;
    std::deque<Event> events;
    std::deque<Event> pinned;
    uint64_t nextSeq = 1;
    uint64_t dropped = 0;
    uint64_t epoch = 0;
};

inline Store &Instance() {
    static Store store;
    return store;
}

inline bool Enabled() noexcept {
    try {
        return Instance().Enabled();
    } catch (...) {
        return false;
    }
}

inline void SetEnabled(bool enabled) noexcept {
    try {
        Instance().SetEnabled(enabled);
    } catch (...) {
    }
}

inline void Clear() noexcept {
    try {
        Instance().Clear();
    } catch (...) {
    }
}

inline std::string SnapshotJson(uint64_t afterSeq = 0, size_t maxEvents = 4096) noexcept {
    try {
        return Instance().SnapshotJson(afterSeq, maxEvents);
    } catch (...) {
        return "{\"schema\":1,\"error\":\"snapshot failed\",\"events\":[]}";
    }
}

inline void RecordGapEvent(const std::string &name, int layer,
                           uint64_t startNs, uint64_t endNs) noexcept {
    try {
        Instance().Record(layer, name, "host", "", "", startNs,
                          endNs > startNs ? endNs - startNs : 0,
                          Timing(), true);
    } catch (...) {
    }
}

// Closes the wall-clock hole since the last traced event.  Small holes are
// intentionally left open so they accumulate into the next labelled region
// instead of creating one event per operator.
inline void FlushGap(uint64_t nowNs) noexcept {
    try {
        if (!Enabled() || !GapTrackingEnabled()) {
            return;
        }
        ThreadPhase &state = TlsPhase();
        if (state.phase == (int)PhaseKind::None || state.phaseDepth <= 0 ||
            state.opDepth > 0 || state.lastEndNs == 0 ||
            nowNs <= state.lastEndNs) {
            return;
        }
        const uint64_t holeNs = nowNs - state.lastEndNs;
        if (holeNs < GapThresholdNs()) {
            return;
        }
        std::string name = "[gap] ";
        if (state.region != nullptr && state.region[0] != '\0') {
            name += state.region;
        } else {
            name += "host";
        }
        RecordGapEvent(name, state.layer, state.lastEndNs, nowNs);
        state.lastEndNs = nowNs;
    } catch (...) {
    }
}

inline void Record(int layer,
                   const std::string &op,
                   const std::string &device,
                   const std::string &weight,
                   const std::string &io,
                   uint64_t startNs,
                   uint64_t endNs,
                   const Timing &timing = Timing(),
                   bool gap = false) noexcept {
    try {
        Instance().Record(layer, op, device, weight, io, startNs,
                          endNs > startNs ? endNs - startNs : 0, timing, gap);
    } catch (...) {
    }
}

} // namespace op_trace
} // namespace fastllm
