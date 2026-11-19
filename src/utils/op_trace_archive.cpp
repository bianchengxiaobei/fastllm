#include "utils/op_trace_archive.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fastllm {
namespace op_trace {
namespace {

namespace fs = std::filesystem;

constexpr uint64_t kMiB = 1024ull * 1024ull;
constexpr size_t kMaxTidEntries = 254;
constexpr size_t kMaxStringEntries = 65535;
constexpr size_t kMaxSummaryTokens = 100000;
constexpr size_t kMaxSummaryOutput = 8192;
constexpr size_t kMaxSummaryOps = 64;
constexpr size_t kMaxSummaryGaps = 32;
constexpr size_t kMaxChunkOps = 32;
constexpr size_t kMaxChunkGaps = 32;
constexpr size_t kMaxChunkDevices = 64;

uint64_t EnvU64(const char *name, uint64_t fallback, uint64_t low, uint64_t high) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    if (parsed < low) {
        parsed = low;
    }
    if (parsed > high) {
        parsed = high;
    }
    return (uint64_t)parsed;
}

std::string ToUtf8Path(const fs::path &path) {
    auto text = path.u8string();
    return std::string(text.begin(), text.end());
}

fs::path FromUtf8Path(const std::string &text) {
    return fs::u8path(text);
}

uint64_t WallNs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

uint32_t CurrentPid() {
#ifdef _WIN32
    return (uint32_t)_getpid();
#else
    return (uint32_t)getpid();
#endif
}

std::string HostName() {
#ifdef _WIN32
    const char *value = std::getenv("COMPUTERNAME");
    return (value != nullptr && value[0] != '\0') ? std::string(value)
                                                  : std::string("unknown");
#else
    char buffer[256] = {0};
    if (gethostname(buffer, sizeof(buffer) - 1) == 0) {
        return std::string(buffer);
    }
    return std::string("unknown");
#endif
}

std::string LocalStamp() {
    std::time_t now = std::time(nullptr);
    std::tm value{};
#ifdef _WIN32
    localtime_s(&value, &now);
#else
    localtime_r(&now, &value);
#endif
    char stamp[32] = {0};
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &value);
    char result[64] = {0};
    std::snprintf(result, sizeof(result), "%s-%u", stamp, CurrentPid());
    return std::string(result);
}

void PutU8(std::string &out, uint8_t value) {
    out.push_back((char)value);
}

void PutU16(std::string &out, uint16_t value) {
    out.push_back((char)(value & 0xFF));
    out.push_back((char)((value >> 8) & 0xFF));
}

void PutU32(std::string &out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back((char)((value >> shift) & 0xFF));
    }
}

void PutU64(std::string &out, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        out.push_back((char)((value >> shift) & 0xFF));
    }
}

void PutVarint(std::string &out, uint64_t value) {
    while (value >= 0x80) {
        out.push_back((char)((value & 0x7F) | 0x80));
        value >>= 7;
    }
    out.push_back((char)value);
}

void PutZigzag(std::string &out, int64_t value) {
    uint64_t encoded = (uint64_t)value << 1;
    if (value < 0) {
        encoded = ~encoded;
    }
    PutVarint(out, encoded);
}

const char *PhaseLabel(uint8_t code) {
    return code == 0 ? "none" : PhaseName(code);
}

uint64_t TokenKey(int request, int token) {
    return ((uint64_t)(uint32_t)request << 32) | (uint64_t)(uint32_t)token;
}

struct OpAgg {
    uint64_t totalNs = 0;
    uint64_t count = 0;
    uint64_t maxNs = 0;
};

struct TokenAgg {
    int request = -1;
    int token = -1;
    uint64_t firstSeq = 0;
    uint64_t lastSeq = 0;
    uint64_t events = 0;
    uint64_t totalNs = 0;
    uint64_t runNs = 0;
    uint64_t transferNs = 0;
    uint64_t syncNs = 0;
    uint64_t otherOpNs = 0;
    uint64_t gapEventNs = 0;
    uint64_t cudaNs = 0;
    uint64_t cpuNs = 0;
    uint64_t otherDeviceNs = 0;
    uint64_t startNs = UINT64_MAX;
    uint64_t endNs = 0;
    bool hasSplit = false;
};

void AddTokenEvent(TokenAgg &target, const Event &event) {
    if (target.events == 0) {
        target.request = event.request;
        target.token = event.token;
        target.firstSeq = event.seq;
        target.startNs = event.startNs;
    }
    target.lastSeq = event.seq;
    target.events++;
    target.totalNs += event.durationNs;
    if (event.gap) {
        target.gapEventNs += event.durationNs;
    } else {
        target.hasSplit = true;
        target.runNs += event.runNs;
        target.transferNs += event.transferNs;
        target.syncNs += event.syncNs;
        uint64_t split = event.runNs + event.transferNs + event.syncNs;
        target.otherOpNs += event.durationNs > split ? event.durationNs - split : 0;
        if (event.device == "cuda") {
            target.cudaNs += event.durationNs;
        } else if (event.device == "cpu") {
            target.cpuNs += event.durationNs;
        } else {
            target.otherDeviceNs += event.durationNs;
        }
    }
    uint64_t end = event.startNs + event.durationNs;
    if (event.startNs < target.startNs) {
        target.startNs = event.startNs;
    }
    if (end > target.endNs) {
        target.endNs = end;
    }
}

struct ChunkStrings {
    std::vector<std::string> items;
    std::unordered_map<std::string, uint32_t> ids;

    uint32_t Add(const std::string &value) {
        auto found = ids.find(value);
        if (found != ids.end()) {
            return found->second;
        }
        uint32_t id = (uint32_t)items.size();
        items.push_back(value);
        ids.emplace(value, id);
        return id;
    }

    bool Serialize(std::string &out) const {
        if (items.size() > kMaxStringEntries) {
            return false;
        }
        PutU16(out, (uint16_t)items.size());
        for (const std::string &item : items) {
            if (item.size() > 65535) {
                return false;
            }
            PutU16(out, (uint16_t)item.size());
            out += item;
        }
        return true;
    }
};

struct ChunkTids {
    std::vector<uint64_t> values;
    std::unordered_map<uint64_t, uint8_t> ids;

    uint8_t Add(uint64_t value) {
        auto found = ids.find(value);
        if (found != ids.end()) {
            return found->second;
        }
        if (values.size() < kMaxTidEntries) {
            uint8_t id = (uint8_t)values.size();
            values.push_back(value);
            ids.emplace(value, id);
            return id;
        }
        return (uint8_t)kMaxTidEntries;
    }

    void Serialize(std::string &out) const {
        PutU8(out, (uint8_t)values.size());
        for (uint64_t value : values) {
            PutU64(out, value);
        }
    }
};

struct ChunkStats {
    uint64_t firstSeq = 0;
    uint64_t lastSeq = 0;
    uint64_t t0Ns = UINT64_MAX;
    uint64_t t1Ns = 0;
    uint64_t phases[4] = {0, 0, 0, 0};
    uint64_t phaseNs[4] = {0, 0, 0, 0};
    std::map<std::pair<int, int>, TokenAgg> tokens;
    std::map<int, std::pair<uint64_t, uint64_t>> requests;
    std::map<std::pair<uint8_t, std::string>, OpAgg> ops;
    std::map<std::pair<uint8_t, std::string>, OpAgg> gaps;
    std::map<std::string, std::pair<uint64_t, uint64_t>> devices;
};

void AddChunkEvent(ChunkStats &stats, const Event &event) {
    if (stats.firstSeq == 0) {
        stats.firstSeq = event.seq;
    }
    stats.lastSeq = event.seq;
    uint8_t phase = PhaseCode(event.phase);
    stats.phases[phase]++;
    stats.phaseNs[phase] += event.durationNs;
    if (event.startNs < stats.t0Ns) {
        stats.t0Ns = event.startNs;
    }
    uint64_t end = event.startNs + event.durationNs;
    if (end > stats.t1Ns) {
        stats.t1Ns = end;
    }
    if (phase == (uint8_t)PhaseKind::Decode && event.token >= 0) {
        TokenAgg &token = stats.tokens[std::make_pair(event.request, event.token)];
        AddTokenEvent(token, event);
    }
    auto &request = stats.requests[event.request];
    if (request.first == 0) {
        request.first = event.seq;
    }
    request.second = event.seq;
    auto &target = event.gap ? stats.gaps : stats.ops;
    OpAgg &op = target[std::make_pair(phase, event.op)];
    op.totalNs += event.durationNs;
    op.count++;
    if (event.durationNs > op.maxNs) {
        op.maxNs = event.durationNs;
    }
    auto &device = stats.devices[event.device];
    device.first++;
    device.second += event.durationNs;
}

bool SerializeChunk(const std::vector<Event> &events, std::string &out,
                    ChunkStats &stats) {
    if (events.empty()) {
        return false;
    }
    stats = ChunkStats();
    ChunkStrings strings;
    ChunkTids tids;
    std::vector<uint16_t> opIds(events.size());
    std::vector<uint16_t> devIds(events.size());
    std::vector<uint32_t> weightIds(events.size());
    std::vector<uint32_t> ioIds(events.size());
    std::vector<uint8_t> tidIds(events.size());

    for (size_t i = 0; i < events.size(); ++i) {
        const Event &event = events[i];
        AddChunkEvent(stats, event);
        uint32_t opId = strings.Add(event.op);
        uint32_t devId = strings.Add(event.device);
        uint32_t weightId = event.weight.empty() ? 0xFFFFFFFFu
                                                 : strings.Add(event.weight);
        uint32_t ioId = event.io.empty() ? 0xFFFFFFFFu : strings.Add(event.io);
        if (opId > 0xFFFF || devId > 0xFFFF) {
            return false;
        }
        opIds[i] = (uint16_t)opId;
        devIds[i] = (uint16_t)devId;
        weightIds[i] = weightId;
        ioIds[i] = ioId;
        tidIds[i] = tids.Add(event.threadId);
    }

    std::string stringTable;
    if (!strings.Serialize(stringTable)) {
        return false;
    }

    std::string payload;
    tids.Serialize(payload);

    uint64_t previousSeq = events.front().seq;
    for (size_t i = 0; i < events.size(); ++i) {
        uint64_t delta = i == 0 ? 0 : events[i].seq - previousSeq;
        PutVarint(payload, delta);
        previousSeq = events[i].seq;
    }

    const int64_t baseUs = (int64_t)(events.front().startNs / 1000);
    int64_t previousUs = 0;
    for (size_t i = 0; i < events.size(); ++i) {
        int64_t us = (int64_t)(events[i].startNs / 1000) - baseUs;
        PutZigzag(payload, i == 0 ? us : us - previousUs);
        previousUs = us;
    }

    for (const Event &event : events) {
        PutVarint(payload, event.durationNs);
    }
    for (uint8_t id : tidIds) {
        PutU8(payload, id);
    }
    for (const Event &event : events) {
        PutZigzag(payload, event.layer);
    }
    for (const Event &event : events) {
        PutZigzag(payload, event.request);
    }
    for (const Event &event : events) {
        PutZigzag(payload, event.token);
    }
    for (const Event &event : events) {
        PutU8(payload, event.gap ? 1 : 0);
    }
    for (const Event &event : events) {
        PutU8(payload, PhaseCode(event.phase));
    }
    for (uint16_t id : opIds) {
        PutU16(payload, id);
    }
    for (uint16_t id : devIds) {
        PutU16(payload, id);
    }
    for (uint32_t id : weightIds) {
        PutU32(payload, id);
    }
    for (uint32_t id : ioIds) {
        PutU32(payload, id);
    }
    for (const Event &event : events) {
        PutVarint(payload, event.transferNs);
    }
    for (const Event &event : events) {
        PutVarint(payload, event.reshapeNs);
    }
    for (const Event &event : events) {
        PutVarint(payload, event.runNs);
    }
    for (const Event &event : events) {
        PutVarint(payload, event.syncNs);
    }

    if (stringTable.size() > 0xFFFFFFFFull || payload.size() > 0xFFFFFFFFull) {
        return false;
    }

    out.clear();
    out.reserve(36 + stringTable.size() + payload.size());
    out.append("FOTR");
    PutU16(out, 1);
    PutU16(out, 0);
    PutU32(out, (uint32_t)events.size());
    PutU64(out, events.front().seq);
    PutU64(out, events.front().startNs);
    PutU32(out, (uint32_t)stringTable.size());
    PutU32(out, (uint32_t)payload.size());
    out += stringTable;
    out += payload;
    return true;
}

void AppendChunkOpList(std::string &out,
                       const std::map<std::pair<uint8_t, std::string>, OpAgg> &source,
                       size_t limit) {
    std::vector<std::pair<std::pair<uint8_t, std::string>, OpAgg>> items(
        source.begin(), source.end());
    std::sort(items.begin(), items.end(), [](const auto &left, const auto &right) {
        return left.second.totalNs > right.second.totalNs;
    });
    if (items.size() > limit) {
        items.resize(limit);
    }
    bool first = true;
    for (const auto &entry : items) {
        if (!first) {
            out += ',';
        }
        first = false;
        out += "{\"phase\":";
        AppendJsonString(out, PhaseLabel(entry.first.first));
        out += ",\"op\":";
        AppendJsonString(out, entry.first.second);
        out += ",\"total_us\":";
        out += std::to_string(entry.second.totalNs / 1000);
        out += ",\"count\":";
        out += std::to_string(entry.second.count);
        out += ",\"max_us\":";
        out += std::to_string(entry.second.maxNs / 1000);
        out += '}';
    }
}

std::string BuildIndexLine(const std::string &file, const ChunkStats &stats,
                           uint64_t bytes) {
    std::string out;
    out.reserve(512 + stats.tokens.size() * 256);
    out += "{\"file\":";
    AppendJsonString(out, file);
    out += ",\"first_seq\":";
    out += std::to_string(stats.firstSeq);
    out += ",\"last_seq\":";
    out += std::to_string(stats.lastSeq);
    uint64_t t0 = stats.t0Ns == UINT64_MAX ? 0 : stats.t0Ns / 1000;
    out += ",\"t0_us\":";
    out += std::to_string(t0);
    out += ",\"t1_us\":";
    out += std::to_string(stats.t1Ns / 1000);
    out += ",\"phases\":{\"prefill\":";
    out += std::to_string(stats.phases[(size_t)PhaseKind::Prefill]);
    out += ",\"decode\":";
    out += std::to_string(stats.phases[(size_t)PhaseKind::Decode]);
    out += ",\"mixed\":";
    out += std::to_string(stats.phases[(size_t)PhaseKind::Mixed]);
    out += ",\"none\":";
    out += std::to_string(stats.phases[(size_t)PhaseKind::None]);
    out += "},\"phase_us\":{\"prefill\":";
    out += std::to_string(stats.phaseNs[(size_t)PhaseKind::Prefill] / 1000);
    out += ",\"decode\":";
    out += std::to_string(stats.phaseNs[(size_t)PhaseKind::Decode] / 1000);
    out += ",\"mixed\":";
    out += std::to_string(stats.phaseNs[(size_t)PhaseKind::Mixed] / 1000);
    out += ",\"none\":";
    out += std::to_string(stats.phaseNs[(size_t)PhaseKind::None] / 1000);
    out += "},\"reqs\":[";
    bool first = true;
    for (const auto &entry : stats.requests) {
        if (!first) {
            out += ',';
        }
        first = false;
        out += "{\"req\":";
        out += std::to_string(entry.first);
        out += ",\"first_seq\":";
        out += std::to_string(entry.second.first);
        out += ",\"last_seq\":";
        out += std::to_string(entry.second.second);
        out += '}';
    }
    out += "],\"tokens\":[";
    first = true;
    for (const auto &entry : stats.tokens) {
        const TokenAgg &token = entry.second;
        if (!first) {
            out += ',';
        }
        first = false;
        uint64_t latencyNs = token.endNs > token.startNs
                                 ? token.endNs - token.startNs
                                 : 1;
        uint64_t gapNs = latencyNs > token.totalNs
                             ? latencyNs - token.totalNs
                             : 0;
        out += "{\"req\":";
        out += std::to_string(token.request);
        out += ",\"token\":";
        out += std::to_string(token.token);
        out += ",\"first_seq\":";
        out += std::to_string(token.firstSeq);
        out += ",\"last_seq\":";
        out += std::to_string(token.lastSeq);
        out += ",\"events\":";
        out += std::to_string(token.events);
        out += ",\"total_us\":";
        out += std::to_string(token.totalNs / 1000);
        out += ",\"other_op_us\":";
        out += std::to_string(token.otherOpNs / 1000);
        out += ",\"latency_us\":";
        out += std::to_string(latencyNs / 1000);
        out += ",\"run_us\":";
        out += std::to_string(token.runNs / 1000);
        out += ",\"xfer_us\":";
        out += std::to_string(token.transferNs / 1000);
        out += ",\"sync_us\":";
        out += std::to_string(token.syncNs / 1000);
        out += ",\"gap_us\":";
        out += std::to_string(gapNs / 1000);
        out += ",\"gap_attr_us\":";
        out += std::to_string(token.gapEventNs / 1000);
        out += ",\"cuda_us\":";
        out += std::to_string(token.cudaNs / 1000);
        out += ",\"cpu_us\":";
        out += std::to_string(token.cpuNs / 1000);
        out += ",\"other_us\":";
        out += std::to_string(token.otherDeviceNs / 1000);
        out += ",\"start_us\":";
        out += std::to_string(token.startNs / 1000);
        out += ",\"end_us\":";
        out += std::to_string(token.endNs / 1000);
        out += ",\"has_split\":";
        out += token.hasSplit ? "true" : "false";
        out += '}';
    }
    out += "],\"ops\":[";
    AppendChunkOpList(out, stats.ops, kMaxChunkOps);
    out += "],\"gaps\":[";
    AppendChunkOpList(out, stats.gaps, kMaxChunkGaps);
    out += "],\"devices\":[";
    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> devices(
        stats.devices.begin(), stats.devices.end());
    std::sort(devices.begin(), devices.end(), [](const auto &left, const auto &right) {
        return left.second.second > right.second.second;
    });
    if (devices.size() > kMaxChunkDevices) {
        devices.resize(kMaxChunkDevices);
    }
    first = true;
    for (const auto &entry : devices) {
        if (!first) {
            out += ',';
        }
        first = false;
        out += "{\"dev\":";
        AppendJsonString(out, entry.first);
        out += ",\"events\":";
        out += std::to_string(entry.second.first);
        out += ",\"us\":";
        out += std::to_string(entry.second.second / 1000);
        out += '}';
    }
    out += "],\"bytes\":";
    out += std::to_string(bytes);
    out += '}';
    return out;
}

struct ArchiveState {
    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable flushCv;
    ArchiveConfig config;
    bool configured = false;
    bool sessionActive = false;
    bool sealed = false;
    bool stopRequested = false;
    bool flushRequested = false;
    bool flushSeal = false;
    bool flushDone = true;
    bool workerStarted = false;
    std::thread worker;
    std::string model;
    std::string sessionId;
    fs::path dir;
    std::string host;
    uint32_t pid = 0;
    uint64_t createdAtNs = 0;
    uint64_t wallBaseNs = 0;
    uint64_t endedAtNs = 0;
    bool ended = false;
    std::string state = "idle";
    std::string stoppedReason;
    uint64_t committedSeq = 0;
    uint64_t chunks = 0;
    uint64_t bytes = 0;
    uint64_t dropped = 0;
    std::vector<Event> current;
    std::deque<std::vector<Event>> pending;
    std::string lastError;
    uint64_t lastMetaNs = 0;
    uint64_t lastDiskNs = 0;
    ~ArchiveState();
};

ArchiveState &State() {
    static ArchiveState state;
    return state;
}

bool WriteFileReplace(const fs::path &temp, const fs::path &target,
                      const std::string &data) {
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return false;
        }
        out.write(data.data(), (std::streamsize)data.size());
        out.flush();
        if (!out.good()) {
            return false;
        }
        out.close();
        if (!out.good()) {
            return false;
        }
    }
    std::error_code error;
    fs::rename(temp, target, error);
    if (error) {
        std::error_code ignored;
        fs::remove(target, ignored);
        error.clear();
        fs::rename(temp, target, error);
    }
    return !error;
}

bool AppendLine(const fs::path &path, const std::string &line) {
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) {
        return false;
    }
    out.write(line.data(), (std::streamsize)line.size());
    out.put('\n');
    out.flush();
    return out.good();
}

std::string MetaBodyLocked(const ArchiveState &state) {
    std::string out;
    out.reserve(768);
    out += "\"schema\":1,\"session\":";
    AppendJsonString(out, state.sessionId);
    out += ",\"pid\":";
    out += std::to_string(state.pid);
    out += ",\"host\":";
    AppendJsonString(out, state.host);
    out += ",\"created_at_ns\":";
    out += std::to_string(state.createdAtNs);
    out += ",\"wall_base_ns\":";
    out += std::to_string(state.wallBaseNs);
    out += ",\"ended_at_ns\":";
    if (state.ended) {
        out += std::to_string(state.endedAtNs);
    } else {
        out += "null";
    }
    out += ",\"sealed\":";
    out += state.sealed ? "true" : "false";
    out += ",\"model\":";
    AppendJsonString(out, state.model);
    out += ",\"config\":{\"chunk_events\":";
    out += std::to_string(state.config.chunkEvents);
    out += ",\"flush_ms\":";
    out += std::to_string(state.config.flushMs);
    out += ",\"io\":";
    out += state.config.io ? "true" : "false";
    out += ",\"gaps\":";
    out += state.config.gaps ? "true" : "false";
    out += ",\"gap_us\":";
    out += std::to_string(state.config.gapUs);
    out += "},\"archive\":{\"committed_seq\":";
    out += std::to_string(state.committedSeq);
    out += ",\"chunks\":";
    out += std::to_string(state.chunks);
    out += ",\"bytes\":";
    out += std::to_string(state.bytes);
    out += ",\"dropped\":";
    out += std::to_string(state.dropped);
    out += ",\"state\":";
    AppendJsonString(out, state.state);
    out += ",\"stopped_reason\":";
    AppendJsonString(out, state.stoppedReason);
    out += ",\"packed\":false,\"ftrace\":null,\"pack_error\":\"\"}";
    return out;
}

bool WriteMetaLocked(ArchiveState &state) {
    if (state.dir.empty()) {
        return false;
    }
    std::string text = "{";
    text += MetaBodyLocked(state);
    text += '}';
    if (!WriteFileReplace(state.dir / "meta.json.tmp", state.dir / "meta.json",
                          text)) {
        state.lastError = "meta_write_failed";
        return false;
    }
    state.lastMetaNs = NowNs();
    return true;
}

bool HasDiskSpace(const fs::path &dir, uint64_t minimum) {
    if (minimum == 0) {
        return true;
    }
    std::error_code error;
    fs::space_info info = fs::space(dir, error);
    if (error) {
        return true;
    }
    return info.available >= minimum;
}

void FinishFlushLocked(ArchiveState &state) {
    if (!state.flushRequested) {
        return;
    }
    if (!state.pending.empty() || !state.current.empty()) {
        return;
    }
    if (state.flushSeal) {
        state.sealed = true;
        state.ended = true;
        state.endedAtNs = WallNs();
        if (state.state == "recording") {
            state.state = "sealed";
        }
        WriteMetaLocked(state);
        state.sessionActive = false;
    }
    state.flushRequested = false;
    state.flushSeal = false;
    state.flushDone = true;
    state.flushCv.notify_all();
}

void WorkerMain(ArchiveState *state);

ArchiveState::~ArchiveState() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopRequested = true;
        flushRequested = true;
        flushSeal = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

void WorkerMain(ArchiveState *state) {
    for (;;) {
        std::vector<Event> chunk;
        bool shouldStop = false;
        fs::path dir;
        uint64_t chunkIndex = 0;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            uint64_t flushMs = state->config.flushMs;
            if (flushMs == 0) {
                flushMs = 200;
            }
            state->cv.wait_for(lock, std::chrono::milliseconds((int64_t)flushMs),
                               [state] {
                                   return state->stopRequested ||
                                          state->flushRequested ||
                                          !state->pending.empty();
                               });
            if (!state->sessionActive) {
                if (state->stopRequested) {
                    break;
                }
                FinishFlushLocked(*state);
                continue;
            }
            if (state->state != "recording") {
                state->dropped += state->current.size();
                state->current.clear();
                for (const auto &pending : state->pending) {
                    state->dropped += pending.size();
                }
                state->pending.clear();
                FinishFlushLocked(*state);
                if (state->stopRequested) {
                    break;
                }
                continue;
            }
            if (!state->current.empty() &&
                (state->flushRequested || state->pending.empty())) {
                state->pending.push_back(std::move(state->current));
                state->current = std::vector<Event>();
                state->current.reserve(state->config.chunkEvents);
            }

            if (!state->pending.empty()) {
                chunk = std::move(state->pending.front());
                state->pending.pop_front();
            }
            if (chunk.empty()) {
                FinishFlushLocked(*state);
                if (state->stopRequested) {
                    break;
                }
                continue;
            }
            dir = state->dir;
            chunkIndex = state->chunks + 1;
        }

        std::string data;
        ChunkStats stats;
        bool ok = SerializeChunk(chunk, data, stats);
        if (ok) {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->state != "recording") {
                ok = false;
            } else {
                uint64_t now = NowNs();
                if (now - state->lastDiskNs > 1000000000ull) {
                    state->lastDiskNs = now;
                    if (!HasDiskSpace(state->dir, state->config.minFreeBytes)) {
                        state->state = "stopped";
                        state->stoppedReason = "low_disk";
                        state->lastError = "low_disk";
                        std::fprintf(stderr, "[op_trace] archiving stopped: low_disk\n");
                        ok = false;
                    }
                }
                if (ok && state->config.sessionMaxBytes > 0 &&
                    state->bytes + data.size() > state->config.sessionMaxBytes) {
                    state->state = "truncated";
                    state->stoppedReason = "session_max_bytes";
                    state->lastError = "session_max_bytes";
                    std::fprintf(stderr, "[op_trace] archiving stopped: session_max_bytes\n");
                    ok = false;
                }
            }
            if (!ok) {
                state->dropped += chunk.size();
            }
        }

        std::string fileName;
        if (ok) {
            char name[32] = {0};
            std::snprintf(name, sizeof(name), "%05llu.bin",
                          (unsigned long long)chunkIndex);
            fileName = name;
            fs::path temp = dir / "chunks" / (fileName + ".tmp");
            fs::path target = dir / "chunks" / fileName;
            ok = WriteFileReplace(temp, target, data);
        }

        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->state == "recording") {
                if (ok) {
                    state->committedSeq = stats.lastSeq;
                    state->chunks++;
                    state->bytes += data.size();
                    if (!AppendLine(state->dir / "chunks.idx",
                                    BuildIndexLine("chunks/" + fileName, stats,
                                                   data.size()))) {
                        state->state = "stopped";
                        state->stoppedReason = "index_write_failed";
                        state->lastError = "index_write_failed";
                        std::fprintf(stderr, "[op_trace] archiving stopped: index_write_failed\n");
                    }
                    if (NowNs() - state->lastMetaNs > 1000000000ull ||
                        state->flushRequested || state->state != "recording") {
                        WriteMetaLocked(*state);
                    }
                } else {
                    state->state = "stopped";
                    state->stoppedReason = "chunk_write_failed";
                    state->lastError = "chunk_write_failed";
                    std::fprintf(stderr, "[op_trace] archiving stopped: chunk_write_failed\n");
                    state->dropped += chunk.size();
                }
            }
            FinishFlushLocked(*state);
            if (state->stopRequested && state->pending.empty() &&
                state->current.empty()) {
                shouldStop = true;
            }
        }
        if (shouldStop) {
            break;
        }
    }
}

bool EnsureSessionLocked(ArchiveState &state) {
    if (!state.config.enabled || state.sessionActive) {
        return false;
    }
    if (!state.workerStarted) {
        state.stopRequested = false;
        state.worker = std::thread(WorkerMain, &state);
        state.workerStarted = true;
    }
    fs::path base = state.config.dir.empty()
                        ? fs::current_path() / ".fastllm" / "op_trace"
                        : FromUtf8Path(state.config.dir);
    std::error_code error;
    std::string id = LocalStamp();
    fs::path dir = base / id;
    bool unique = !fs::exists(dir, error);
    if (!unique) {
        for (int suffix = 1; suffix < 10000; ++suffix) {
            std::string candidate = id + "-" + std::to_string(suffix);
            fs::path alternate = base / candidate;
            if (!fs::exists(alternate, error)) {
                id = candidate;
                dir = alternate;
                unique = true;
                break;
            }
        }
    }
    if (!unique) {
        state.state = "stopped";
        state.stoppedReason = "session_exists";
        state.lastError = "session_exists";
        return false;
    }
    fs::create_directories(dir / "chunks", error);

    if (error) {
        state.state = "stopped";
        state.stoppedReason = "mkdir_failed";
        state.lastError = error.message();
        std::fprintf(stderr, "[op_trace] cannot create %s: %s\n",
                     ToUtf8Path(dir).c_str(), error.message().c_str());
        return false;
    }
    {
        std::ofstream index(dir / "chunks.idx",
                            std::ios::binary | std::ios::trunc);
        if (!index) {
            state.state = "stopped";
            state.stoppedReason = "index_create_failed";
            state.lastError = "index_create_failed";
            return false;
        }
    }
    state.sessionId = id;
    state.dir = dir;
    state.host = HostName();
    state.pid = CurrentPid();
    state.createdAtNs = WallNs();
    state.wallBaseNs = state.createdAtNs;
    state.endedAtNs = 0;
    state.ended = false;
    state.sealed = false;
    state.state = "recording";
    state.stoppedReason.clear();
    state.lastError.clear();
    state.committedSeq = 0;
    state.chunks = 0;
    state.bytes = 0;
    state.dropped = 0;
    state.current.clear();
    state.current.reserve(state.config.chunkEvents);
    state.pending.clear();
    state.flushRequested = false;
    state.flushSeal = false;
    state.flushDone = true;
    state.lastMetaNs = 0;
    state.lastDiskNs = 0;
    state.sessionActive = true;
    WriteMetaLocked(state);
    state.cv.notify_all();
    std::fprintf(stderr, "[op_trace] recording %s -> %s\n", state.sessionId.c_str(),
                 ToUtf8Path(state.dir).c_str());
    return true;
}

struct AggregateState {
    std::mutex mutex;
    std::unordered_map<uint64_t, TokenAgg> tokens;
    std::map<uint64_t, uint64_t> tokenOrder;
    std::map<std::pair<uint8_t, std::string>, OpAgg> ops;
    std::map<std::pair<uint8_t, std::string>, OpAgg> gaps;
    uint64_t phaseEvents[4] = {0, 0, 0, 0};
    uint64_t phaseNs[4] = {0, 0, 0, 0};
    uint64_t phaseTokens[4] = {0, 0, 0, 0};
    uint64_t tokensSeen = 0;
    bool evicted = false;
};

AggregateState &Aggregate() {
    static AggregateState state;
    return state;
}

void AppendTokenJson(std::string &out, const TokenAgg &token) {
    uint64_t latencyNs = token.endNs > token.startNs ? token.endNs - token.startNs : 1;
    uint64_t gapNs = latencyNs > token.totalNs ? latencyNs - token.totalNs : 0;
    out += "{\"req\":";
    out += std::to_string(token.request);
    out += ",\"token\":";
    out += std::to_string(token.token);
    out += ",\"first_seq\":";
    out += std::to_string(token.firstSeq);
    out += ",\"last_seq\":";
    out += std::to_string(token.lastSeq);
    out += ",\"events\":";
    out += std::to_string(token.events);
    out += ",\"total_us\":";
    out += std::to_string(token.totalNs / 1000);
    out += ",\"latency_us\":";
    out += std::to_string(latencyNs / 1000);
    out += ",\"run_us\":";
    out += std::to_string(token.runNs / 1000);
    out += ",\"xfer_us\":";
    out += std::to_string(token.transferNs / 1000);
    out += ",\"sync_us\":";
    out += std::to_string(token.syncNs / 1000);
    out += ",\"other_op_us\":";
    out += std::to_string(token.otherOpNs / 1000);
    out += ",\"gap_us\":";
    out += std::to_string(gapNs / 1000);
    out += ",\"gap_attr_us\":";
    out += std::to_string(token.gapEventNs / 1000);
    out += ",\"cuda_us\":";
    out += std::to_string(token.cudaNs / 1000);
    out += ",\"cpu_us\":";
    out += std::to_string(token.cpuNs / 1000);
    out += ",\"other_us\":";
    out += std::to_string(token.otherDeviceNs / 1000);
    out += ",\"start_us\":";
    out += std::to_string(token.startNs / 1000);
    out += ",\"end_us\":";
    out += std::to_string(token.endNs / 1000);
    out += ",\"has_split\":";
    out += token.hasSplit ? "true" : "false";
    out += '}';
}

void AppendOpListJson(std::string &out,
                      std::vector<std::pair<std::pair<uint8_t, std::string>, OpAgg>>
                          &items,
                      size_t limit) {
    std::sort(items.begin(), items.end(), [](const auto &left, const auto &right) {
        return left.second.totalNs > right.second.totalNs;
    });
    if (items.size() > limit) {
        items.resize(limit);
    }
    out += '[';
    bool first = true;
    for (const auto &entry : items) {
        if (!first) {
            out += ',';
        }
        first = false;
        out += "{\"phase\":";
        AppendJsonString(out, PhaseLabel(entry.first.first));
        out += ",\"op\":";
        AppendJsonString(out, entry.first.second);
        out += ",\"total_us\":";
        out += std::to_string(entry.second.totalNs / 1000);
        out += ",\"count\":";
        out += std::to_string(entry.second.count);
        out += ",\"max_us\":";
        out += std::to_string(entry.second.maxNs / 1000);
        out += '}';
    }
    out += ']';
}

} // namespace

ArchiveConfig ArchiveConfigFromEnv(bool io, bool gaps, uint64_t gapUs) noexcept {
    ArchiveConfig config;
    try {
        config.io = io;
        config.gaps = gaps;
        config.gapUs = gapUs;
        const char *archive = std::getenv("FASTLLM_OP_TRACE_ARCHIVE");
        config.enabled = (archive == nullptr || archive[0] == '\0')
                             ? true
                             : ParseEnvEnabled(archive);
        const char *dir = std::getenv("FASTLLM_OP_TRACE_DIR");
        fs::path path = (dir != nullptr && dir[0] != '\0')
                            ? FromUtf8Path(dir)
                            : fs::current_path() / ".fastllm" / "op_trace";
        config.dir = ToUtf8Path(path);
        config.chunkEvents =
            (size_t)EnvU64("FASTLLM_OP_TRACE_CHUNK_EVENTS", 4096, 64, 65536);
        config.flushMs =
            (size_t)EnvU64("FASTLLM_OP_TRACE_FLUSH_MS", 200, 50, 60000);
        config.pendingChunks = 8;
        config.minFreeBytes =
            EnvU64("FASTLLM_OP_TRACE_MIN_FREE_MB", 2048, 0, 1048576) * kMiB;
        config.sessionMaxBytes =
            EnvU64("FASTLLM_OP_TRACE_SESSION_MAX_MB", 0, 0, 1048576) * kMiB;
    } catch (...) {
    }
    return config;
}

void ArchiveConfigure(const ArchiveConfig &config) noexcept {
    try {
        ArchiveState &state = State();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.config = config;
        state.config.chunkEvents =
            std::min<size_t>(65536, std::max<size_t>(64, config.chunkEvents));
        state.config.flushMs =
            std::min<size_t>(60000, std::max<size_t>(50, config.flushMs));
        state.config.pendingChunks =
            std::min<size_t>(64, std::max<size_t>(1, config.pendingChunks));
        if (state.config.dir.empty()) {
            state.config.dir =
                ToUtf8Path(fs::current_path() / ".fastllm" / "op_trace");
        }
        state.configured = true;
        if (state.config.enabled) {
            std::fprintf(stderr,
                         "[op_trace] archive dir: %s (chunk=%llu, flush=%llums)\n",
                         state.config.dir.c_str(),
                         (unsigned long long)state.config.chunkEvents,
                         (unsigned long long)state.config.flushMs);
        }
    } catch (...) {
    }
}

void ArchiveSetModel(const std::string &model) noexcept {
    try {
        ArchiveState &state = State();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.model = model;
        if (state.sessionActive) {
            WriteMetaLocked(state);
        }
    } catch (...) {
    }
}

void ArchiveEnsureSession() noexcept {
    try {
        ArchiveState &state = State();
        bool created = false;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            created = EnsureSessionLocked(state);
        }
        if (created) {
            AggregateClear();
        }
    } catch (...) {
    }
}

void ArchiveAppend(const Event &event) noexcept {
    try {
        ArchiveState &state = State();
        bool created = false;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!state.config.enabled) {
                return;
            }
            if (!state.sessionActive) {
                created = EnsureSessionLocked(state);
                if (!state.sessionActive) {
                    return;
                }
            }
            if (state.state != "recording") {
                state.dropped++;
                return;
            }
            state.current.push_back(event);
            if (state.current.size() >= state.config.chunkEvents) {
                if (state.pending.size() >= state.config.pendingChunks) {
                    state.dropped += state.current.size();
                    state.current.clear();
                } else {
                    state.pending.push_back(std::move(state.current));
                    state.current = std::vector<Event>();
                    state.current.reserve(state.config.chunkEvents);
                }
                state.cv.notify_one();
            }
        }
        if (created) {
            AggregateClear();
        }
    } catch (...) {
    }
}

bool ArchiveFlush(bool seal) noexcept {
    try {
        ArchiveState &state = State();
        std::unique_lock<std::mutex> lock(state.mutex);
        if (!state.sessionActive) {
            if (seal && !state.sessionId.empty()) {
                state.sealed = true;
                if (state.state == "recording") {
                    state.state = "sealed";
                }
            }
            return true;
        }

        state.flushRequested = true;
        state.flushSeal = state.flushSeal || seal;
        state.flushDone = false;
        state.cv.notify_all();
        bool completed = state.flushCv.wait_for(
            lock, std::chrono::seconds(5), [&state] { return state.flushDone; });
        if (!completed) {
            state.lastError = "flush_timeout";
            state.state = "stopped";
            state.stoppedReason = "flush_timeout";
            std::fprintf(stderr, "[op_trace] flush timeout for session %s\n",
                         state.sessionId.c_str());
        }

        return completed;
    } catch (...) {
        return false;
    }
}

void ArchiveRotateSession() noexcept {
    try {
        if (!ArchiveFlush(true)) {
            return;
        }
        ArchiveEnsureSession();
    } catch (...) {
    }
}

void ArchiveShutdown() noexcept {
    try {
        ArchiveFlush(true);
        ArchiveState &state = State();
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.stopRequested = true;
        }
        state.cv.notify_all();
        if (state.worker.joinable()) {
            state.worker.join();
        }
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.workerStarted = false;
        }
    } catch (...) {
    }
}


std::string ArchiveSessionId() noexcept {
    try {
        ArchiveState &state = State();
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.sessionId;
    } catch (...) {
        return std::string();
    }
}

std::string ArchiveStatusJson() noexcept {
    try {
        ArchiveState &state = State();
        std::lock_guard<std::mutex> lock(state.mutex);
        std::string out;
        out += "{\"enabled\":";
        out += state.config.enabled ? "true" : "false";
        out += ",\"dir\":";
        AppendJsonString(out, state.dir.empty() ? std::string()
                                               : ToUtf8Path(state.dir));
        out += ",\"committed_seq\":";
        out += std::to_string(state.committedSeq);
        out += ",\"chunks\":";
        out += std::to_string(state.chunks);
        out += ",\"bytes\":";
        out += std::to_string(state.bytes);
        out += ",\"dropped\":";
        out += std::to_string(state.dropped);
        out += ",\"state\":";
        AppendJsonString(out, state.state);
        out += ",\"sealed\":";
        out += state.sealed ? "true" : "false";
        out += ",\"packed\":false}";
        return out;
    } catch (...) {
        return "{\"enabled\":false}";
    }
}

std::string ArchiveSessionJson() noexcept {
    try {
        ArchiveState &state = State();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.sessionId.empty()) {
            return "{\"schema\":1,\"session\":\"\",\"dir\":\"\",\"active\":false}";
        }
        std::string out = "{";
        out += MetaBodyLocked(state);
        out += ",\"dir\":";
        AppendJsonString(out, ToUtf8Path(state.dir));
        out += ",\"active\":";
        out += state.sessionActive ? "true" : "false";
        out += '}';
        return out;
    } catch (...) {
        return "{\"schema\":1,\"session\":\"\",\"dir\":\"\",\"active\":false}";
    }
}

void AggregateRecord(const Event &event) noexcept {
    try {
        AggregateState &state = Aggregate();
        std::lock_guard<std::mutex> lock(state.mutex);
        uint8_t phase = PhaseCode(event.phase);
        state.phaseEvents[phase]++;
        state.phaseNs[phase] += event.durationNs;
        if (phase == (uint8_t)PhaseKind::Decode && event.token >= 0) {
            uint64_t key = TokenKey(event.request, event.token);
            auto found = state.tokens.find(key);
            if (found == state.tokens.end()) {
                if (state.tokens.size() >= kMaxSummaryTokens &&
                    !state.tokenOrder.empty()) {
                    auto oldest = state.tokenOrder.begin();
                    state.tokens.erase(oldest->second);
                    state.tokenOrder.erase(oldest);
                    state.evicted = true;
                }
                TokenAgg token;
                AddTokenEvent(token, event);
                state.tokenOrder.emplace(token.firstSeq, key);
                state.tokens.emplace(key, std::move(token));
                state.tokensSeen++;
                state.phaseTokens[phase]++;
            } else {
                AddTokenEvent(found->second, event);
            }
        }
        auto &target = event.gap ? state.gaps : state.ops;
        OpAgg &op = target[std::make_pair(phase, event.op)];
        op.totalNs += event.durationNs;
        op.count++;
        if (event.durationNs > op.maxNs) {
            op.maxNs = event.durationNs;
        }
    } catch (...) {
    }
}

void AggregateClear() noexcept {
    try {
        AggregateState &state = Aggregate();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.tokens.clear();
        state.tokenOrder.clear();
        state.ops.clear();
        state.gaps.clear();
        for (int i = 0; i < 4; ++i) {
            state.phaseEvents[i] = 0;
            state.phaseNs[i] = 0;
            state.phaseTokens[i] = 0;
        }
        state.tokensSeen = 0;
        state.evicted = false;
    } catch (...) {
    }
}

std::string ArchiveSummaryJson(size_t maxTokens) noexcept {
    try {
        if (maxTokens == 0) {
            maxTokens = kMaxSummaryOutput;
        }
        maxTokens = std::min(maxTokens, kMaxSummaryOutput);
        std::string session = ArchiveSessionId();

        uint64_t phaseEvents[4] = {0, 0, 0, 0};
        uint64_t phaseNs[4] = {0, 0, 0, 0};
        uint64_t phaseTokens[4] = {0, 0, 0, 0};
        uint64_t tokensSeen = 0;
        bool truncated = false;
        std::vector<TokenAgg> tokens;
        std::vector<std::pair<std::pair<uint8_t, std::string>, OpAgg>> ops;
        std::vector<std::pair<std::pair<uint8_t, std::string>, OpAgg>> gaps;

        {
            AggregateState &state = Aggregate();
            std::lock_guard<std::mutex> lock(state.mutex);
            for (int i = 0; i < 4; ++i) {
                phaseEvents[i] = state.phaseEvents[i];
                phaseNs[i] = state.phaseNs[i];
                phaseTokens[i] = state.phaseTokens[i];
            }
            tokensSeen = state.tokensSeen;
            struct HeapItem {
                uint64_t endNs;
                uint64_t key;
            };
            auto compare = [](const HeapItem &left, const HeapItem &right) {
                return left.endNs > right.endNs;
            };
            std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(compare)>
                heap(compare);
            for (const auto &entry : state.tokens) {
                heap.push({entry.second.endNs, entry.first});
                if (heap.size() > maxTokens) {
                    heap.pop();
                }
            }
            tokens.reserve(heap.size());
            while (!heap.empty()) {
                auto found = state.tokens.find(heap.top().key);
                if (found != state.tokens.end()) {
                    tokens.push_back(found->second);
                }
                heap.pop();
            }
            std::sort(tokens.begin(), tokens.end(),
                      [](const TokenAgg &left, const TokenAgg &right) {
                          return left.endNs > right.endNs;
                      });
            ops.assign(state.ops.begin(), state.ops.end());
            gaps.assign(state.gaps.begin(), state.gaps.end());
            truncated = state.evicted || state.tokens.size() > tokens.size();
        }

        std::string out;
        out.reserve(512 + tokens.size() * 320 + ops.size() * 96);
        out += "{\"schema\":1,\"session\":";
        AppendJsonString(out, session);
        out += ",\"phases\":{\"prefill\":{\"events\":";
        out += std::to_string(phaseEvents[(size_t)PhaseKind::Prefill]);
        out += ",\"us\":";
        out += std::to_string(phaseNs[(size_t)PhaseKind::Prefill] / 1000);
        out += ",\"tokens\":";
        out += std::to_string(phaseTokens[(size_t)PhaseKind::Prefill]);
        out += "},\"decode\":{\"events\":";
        out += std::to_string(phaseEvents[(size_t)PhaseKind::Decode]);
        out += ",\"us\":";
        out += std::to_string(phaseNs[(size_t)PhaseKind::Decode] / 1000);
        out += ",\"tokens\":";
        out += std::to_string(phaseTokens[(size_t)PhaseKind::Decode]);
        out += "},\"mixed\":{\"events\":";
        out += std::to_string(phaseEvents[(size_t)PhaseKind::Mixed]);
        out += ",\"us\":";
        out += std::to_string(phaseNs[(size_t)PhaseKind::Mixed] / 1000);
        out += ",\"tokens\":";
        out += std::to_string(phaseTokens[(size_t)PhaseKind::Mixed]);
        out += "},\"none\":{\"events\":";
        out += std::to_string(phaseEvents[(size_t)PhaseKind::None]);
        out += ",\"us\":";
        out += std::to_string(phaseNs[(size_t)PhaseKind::None] / 1000);
        out += ",\"tokens\":";
        out += std::to_string(phaseTokens[(size_t)PhaseKind::None]);
        out += "}},\"tokens\":[";
        bool first = true;
        for (const TokenAgg &token : tokens) {
            if (!first) {
                out += ',';
            }
            first = false;
            AppendTokenJson(out, token);
        }
        out += "],\"tokens_total\":";
        out += std::to_string(tokensSeen);
        out += ",\"tokens_returned\":";
        out += std::to_string(tokens.size());
        out += ",\"tokens_truncated\":";
        out += truncated ? "true" : "false";
        out += ",\"ops\":";
        AppendOpListJson(out, ops, kMaxSummaryOps);
        out += ",\"gaps\":";
        AppendOpListJson(out, gaps, kMaxSummaryGaps);
        out += '}';
        return out;
    } catch (...) {
        return "{\"schema\":1,\"phases\":{},\"tokens\":[]}";
    }
}

} // namespace op_trace
} // namespace fastllm





