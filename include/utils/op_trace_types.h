#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace fastllm {
namespace op_trace {

using Clock = std::chrono::steady_clock;

inline uint64_t NowNs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}

struct Event {
    uint64_t seq = 0;
    uint64_t startNs = 0;
    uint64_t durationNs = 0;
    uint64_t threadId = 0;
    int layer = -1;
    int request = -1;
    int token = -1;
    bool gap = false;
    std::string phase;
    std::string op;
    std::string device;
    std::string weight;
    std::string io;
    uint64_t transferNs = 0;
    uint64_t reshapeNs = 0;
    uint64_t runNs = 0;
    uint64_t syncNs = 0;
};

struct Timing {
    uint64_t transferNs = 0;
    uint64_t reshapeNs = 0;
    uint64_t runNs = 0;
    uint64_t syncNs = 0;
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

inline uint8_t PhaseCode(const std::string &phase) {
    if (phase == "prefill") {
        return (uint8_t)PhaseKind::Prefill;
    }
    if (phase == "decode") {
        return (uint8_t)PhaseKind::Decode;
    }
    if (phase == "mixed") {
        return (uint8_t)PhaseKind::Mixed;
    }
    return (uint8_t)PhaseKind::None;
}

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

} // namespace op_trace
} // namespace fastllm
