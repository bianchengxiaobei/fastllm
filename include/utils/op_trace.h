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

namespace fastllm {
namespace op_trace {

using Clock = std::chrono::steady_clock;

struct Event {
    uint64_t seq = 0;
    uint64_t startNs = 0;
    uint64_t durationNs = 0;
    uint64_t threadId = 0;
    int layer = -1;
    int request = -1;
    int token = -1;
    std::string phase;
    std::string op;
    std::string device;
    std::string weight;
    std::string io;
};

enum class PhaseKind : uint8_t {
    None = 0,
    Prefill = 1,
    Decode = 2,
    Mixed = 3,
};

inline const char *PhaseName(int phase) {
    switch (phase) {
        case (int)PhaseKind::Prefill: return "prefill";
        case (int)PhaseKind::Decode: return "decode";
        case (int)PhaseKind::Mixed: return "mixed";
        default: return "";
    }
}

struct ThreadPhase {
    int phase = (int)PhaseKind::None;
    int request = -1;
    int token = -1;
};

inline ThreadPhase &TlsPhase() {
    static thread_local ThreadPhase state;
    return state;
}

class PhaseScope {
public:
    PhaseScope(PhaseKind phase, int token = -1, int request = -1) noexcept {
        ThreadPhase &state = TlsPhase();
        previousPhase = state.phase;
        previousRequest = state.request;
        previousToken = state.token;
        state.phase = (int)phase;
        state.request = request;
        state.token = token;
    }

    ~PhaseScope() {
        ThreadPhase &state = TlsPhase();
        state.phase = previousPhase;
        state.request = previousRequest;
        state.token = previousToken;
    }

    PhaseScope(const PhaseScope &) = delete;
    PhaseScope &operator=(const PhaseScope &) = delete;

private:
    int previousPhase = (int)PhaseKind::None;
    int previousRequest = -1;
    int previousToken = -1;
};


inline bool ParseEnvEnabled(const char *value) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 ||
           std::strcmp(value, "on") == 0 ||
           std::strcmp(value, "ON") == 0 ||
           std::strcmp(value, "yes") == 0 ||
           std::strcmp(value, "YES") == 0;
}

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

inline void AppendJsonString(std::string &out, const std::string &value) {
    out += '"';
    for (unsigned char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    out += buffer;
                } else {
                    out += (char)c;
                }
                break;
        }
    }
    out += '"';
}

class Store {
public:
    Store() : capacity(ParseEnvCapacity()),
              enabled(ParseEnvEnabled(std::getenv("FASTLLM_OP_TRACE"))) {}

    bool Enabled() const {
        return enabled.load(std::memory_order_relaxed);
    }

    void SetEnabled(bool value) {
        enabled.store(value, std::memory_order_relaxed);
    }

    size_t Capacity() const {
        return capacity;
    }

    void Clear() noexcept {
        try {
            std::lock_guard<std::mutex> guard(mutex);
            events.clear();
            dropped = 0;
            nextSeq = 1;
            ++epoch;
        } catch (...) {
        }
    }

    void Record(int layer,
                const std::string &op,
                const std::string &device,
                const std::string &weight,
                const std::string &io,
                uint64_t startNs,
                uint64_t durationNs) noexcept {
        if (!Enabled()) {
            return;
        }
        try {
            Event event;
            event.startNs = startNs;
            event.durationNs = durationNs;
            event.threadId = (uint64_t)std::hash<std::thread::id>{}(
                std::this_thread::get_id());
            const ThreadPhase &phase = TlsPhase();
            event.phase = PhaseName(phase.phase);
            event.request = phase.request;
            event.token = phase.token;
            event.layer = layer;
            event.op = op;
            event.device = device;
            event.weight = weight;
            event.io = io;

            std::lock_guard<std::mutex> guard(mutex);
            if (events.size() >= capacity) {
                events.pop_front();
                ++dropped;
            }
            event.seq = nextSeq++;
            events.push_back(std::move(event));
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
        uint64_t currentDropped = 0, currentSize = 0, lost = 0;
        {
            std::lock_guard<std::mutex> guard(mutex);
            currentEpoch = epoch;
            currentDropped = dropped;
            currentSize = (uint64_t)events.size();
            lastSeq = nextSeq > 0 ? nextSeq - 1 : 0;
            firstSeq = events.empty() ? nextSeq : events.front().seq;
            if (afterSeq > 0 && firstSeq > afterSeq + 1) {
                lost = 1;
            }
            if (maxEvents == 0) {
                maxEvents = 4096;
            }
            selected.reserve(std::min(maxEvents, events.size()));
            for (const Event &event : events) {
                if (event.seq <= afterSeq) {
                    continue;
                }
                selected.push_back(event);
                if (selected.size() >= maxEvents) {
                    break;
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
            out += '}';
        }
        out += "]}";
        return out;
    }

private:
    size_t capacity;
    std::atomic<bool> enabled;
    std::mutex mutex;
    std::deque<Event> events;
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

inline uint64_t NowNs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}

inline void Record(int layer,
                   const std::string &op,
                   const std::string &device,
                   const std::string &weight,
                   const std::string &io,
                   uint64_t startNs,
                   uint64_t endNs) noexcept {
    try {
        Instance().Record(layer, op, device, weight, io, startNs,
                          endNs > startNs ? endNs - startNs : 0);
    } catch (...) {
    }
}

} // namespace op_trace
} // namespace fastllm
