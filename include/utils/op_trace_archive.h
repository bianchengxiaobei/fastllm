#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "utils/op_trace_types.h"

namespace fastllm {
namespace op_trace {

struct ArchiveConfig {
    bool enabled = true;
    std::string dir;
    size_t chunkEvents = 4096;
    size_t flushMs = 200;
    size_t pendingChunks = 8;
    uint64_t minFreeBytes = 2048ull * 1024ull * 1024ull;
    uint64_t sessionMaxBytes = 0;
    bool io = true;
    bool gaps = true;
    uint64_t gapUs = 20;
};

ArchiveConfig ArchiveConfigFromEnv(bool io, bool gaps, uint64_t gapUs) noexcept;
void ArchiveConfigure(const ArchiveConfig &config) noexcept;
void ArchiveSetModel(const std::string &model) noexcept;
void ArchiveEnsureSession() noexcept;
void ArchiveAppend(const Event &event) noexcept;
bool ArchiveFlush(bool seal) noexcept;
void ArchiveRotateSession() noexcept;
void ArchiveShutdown() noexcept;

std::string ArchiveSessionId() noexcept;
std::string ArchiveStatusJson() noexcept;
std::string ArchiveSessionJson() noexcept;

void AggregateRecord(const Event &event) noexcept;
void AggregateClear() noexcept;
std::string ArchiveSummaryJson(size_t maxTokens) noexcept;

} // namespace op_trace
} // namespace fastllm
