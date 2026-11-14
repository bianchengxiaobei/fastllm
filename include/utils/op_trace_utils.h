#pragma once

#include "device.h"
#include "op_trace.h"

namespace fastllm {
namespace op_trace {

inline int InferLayerFromName(const std::string &name) {
    static const char *patterns[] = {"layers.", "layer."};
    for (const char *pattern : patterns) {
        size_t pos = name.find(pattern);
        if (pos == std::string::npos) {
            continue;
        }
        pos += std::strlen(pattern);
        int layer = 0;
        bool any = false;
        while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
            layer = layer * 10 + (name[pos] - '0');
            ++pos;
            any = true;
        }
        if (any) {
            return layer;
        }
    }
    return -1;
}

inline int LayerIndex(const DataDict &datas, const IntDict &intParams) {
    auto it = intParams.find("layer");
    if (it != intParams.end()) {
        return it->second;
    }
    auto weightIt = datas.find("weight");
    if (weightIt != datas.end() && weightIt->second != nullptr &&
        intParams.find("weight___batch") == intParams.end()) {
        int layer = InferLayerFromName(weightIt->second->name);
        if (layer >= 0) {
            return layer;
        }
    }
    for (auto &entry : datas) {
        if (entry.second == nullptr ||
            intParams.find(entry.first + "___batch") != intParams.end()) {
            continue;
        }
        int layer = InferLayerFromName(entry.second->name);
        if (layer >= 0) {
            return layer;
        }
    }
    return -1;
}

inline std::string WeightName(const DataDict &datas,
                              const IntDict &intParams) {
    if (intParams.find("weight___batch") != intParams.end()) {
        return "<batch>";
    }
    auto it = datas.find("weight");
    if (it == datas.end() || it->second == nullptr) {
        return "";
    }
    return it->second->name;
}

inline std::string TensorSummary(const DataDict &datas,
                                 const IntDict &intParams) {
    std::string result;
    int written = 0;
    for (auto &it : datas) {
        if (written >= 16 || result.size() >= 800) {
            break;
        }
        if (it.first == "weight") {
            continue;
        }
        if (intParams.find(it.first + "___batch") != intParams.end()) {
            if (!result.empty()) {
                result += '|';
            }
            result += it.first;
            result += "=<batch>";
            ++written;
            continue;
        }
        const Data *data = it.second;
        if (data == nullptr) {
            continue;
        }
        if (!result.empty()) {
            result += '|';
        }
        result += it.first;
        result += '=';
        result += data->name;
        result += '[';
        for (size_t i = 0; i < data->dims.size() && i < 8; ++i) {
            if (i) {
                result += ',';
            }
            result += std::to_string(data->dims[i]);
        }
        result += "] ";
        result += GetDataTypeName(data->dataType);
        ++written;
    }
    return result;
}

inline void RecordOp(const std::string &opType,
                     const std::string &deviceType,
                     const DataDict &datas,
                     const IntDict &intParams,
                     uint64_t startNs,
                     uint64_t endNs,
                     const Timing &timing = Timing()) {
    Record(LayerIndex(datas, intParams), opType, deviceType,
           WeightName(datas, intParams), TensorSummary(datas, intParams),
           startNs, endNs, timing);
}

} // namespace op_trace
} // namespace fastllm
