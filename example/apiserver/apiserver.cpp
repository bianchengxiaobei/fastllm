// Provide by Jacques CHEN (http://whchen.net/index.php/About.html)
// HTML file reference from ChatGLM-MNN （https://github.com/wangzhaode/ChatGLM-MNN)

#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <stdlib.h>
#include <string>
#include <mutex>

/*
 * Headers
 */

#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif //_CRT_SECURE_NO_WARNINGS

#ifndef _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_NONSTDC_NO_DEPRECATE
#endif //_CRT_NONSTDC_NO_DEPRECATE

#if defined(_MSC_VER)
#if _MSC_VER < 1900
#error Sorry, Visual Studio versions prior to 2015 are not supported
#endif

#pragma comment(lib, "ws2_32.lib")

#ifdef _WIN64
using ssize_t = __int64;
#else
using ssize_t = long;
#endif
#endif // _MSC_VER

#ifndef S_ISREG
#define S_ISREG(m) (((m)&S_IFREG) == S_IFREG)
#endif // S_ISREG

#ifndef S_ISDIR
#define S_ISDIR(m) (((m)&S_IFDIR) == S_IFDIR)
#endif // S_ISDIR

#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX

#include <io.h>
#include <process.h> // _spawnl：探测python时要避免经过cmd的引号处理
#include <winsock2.h>
#include <ws2tcpip.h>

#ifndef WSA_FLAG_NO_HANDLE_INHERIT
#define WSA_FLAG_NO_HANDLE_INHERIT 0x80
#endif

#ifndef strcasecmp
#define strcasecmp _stricmp
#endif // strcasecmp

using socket_t = SOCKET;
#ifdef CPPHTTPLIB_USE_POLL
#define poll(fds, nfds, timeout) WSAPoll(fds, nfds, timeout)
#endif

#else // not _WIN32

#include <arpa/inet.h>
#ifndef _AIX
#include <ifaddrs.h>
#endif
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#ifdef __linux__
#include <resolv.h>
#endif
#include <netinet/tcp.h>
#ifdef CPPHTTPLIB_USE_POLL
#include <poll.h>
#endif
#include <csignal>
#include <dlfcn.h> // 动态查CUDA驱动，判断计算能力
#include <pthread.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using socket_t = int;
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#endif //_WIN32

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <exception>
#include "model.h"
#include "models/bert.h"

#ifdef _WIN32
// Windows下socket只能用send/recv/closesocket：CRT的write/read/close(即_write/_read/_close)
// 只认CRT文件描述符表，直接把SOCKET传进去会触发CRT非法参数处理并终止进程
static int SocketWrite(socket_t fd, const char *buf, int len) {
    return send(fd, buf, len, 0);
}

static int SocketRead(socket_t fd, char *buf, int len) {
    return recv(fd, buf, len, 0);
}

static void SocketClose(socket_t fd) {
    closesocket(fd);
}

// 先shutdown写方向，把FIN正常发出去再close，
// 直接close在还有未发数据时可能给对方RST，客户端会当成流被截断
static void SocketShutdownWrite(socket_t fd) {
    shutdown(fd, SD_SEND);
}
#else
static int SocketWrite(socket_t fd, const char *buf, int len) {
    return (int)write(fd, buf, len);
}

static int SocketRead(socket_t fd, char *buf, int len) {
    return (int)read(fd, buf, len);
}

static void SocketClose(socket_t fd) {
    close(fd);
}

static void SocketShutdownWrite(socket_t fd) {
    shutdown(fd, SHUT_WR);
}
#endif

long long _GetCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

std::string GenerateRandomID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    for (int i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            ss << '-';
        }
        ss << std::hex << dis(gen);
    }
    return ss.str();
}

// prefill期间只能靠实测速率推算进度：库没有进度回调，cacheLen在prefill期间也不增长
static std::atomic<double> gPrefillTokenPerSec(0.0);

// prefill 是阻塞调用且库里没有进度回调，这里用后台线程定期打心跳，
// 便于在长 prompt 下判断服务是还在跑还是已经卡死；
// prefill结束后线程不退出，继续负责生成期间的空闲保活(对应ftllm的keep-alive注释帧)
class PrefillHeartbeat {
public:
    PrefillHeartbeat(int intervalSeconds, socket_t keepAliveClient, int totalTokens, int missedTokens)
        : interval(intervalSeconds), keepAliveClient(keepAliveClient),
          totalTokens(totalTokens), missedTokens(missedTokens),
          keepAliveEnabled(false), finished(false), prefillDone(false) {}

    // 响应头写出去之后才能开始发保活帧
    void EnableKeepAlive() {
        keepAliveEnabled = true;
    }

    // chunked帧是"长度行+数据+空行"三次send，主线程和保活线程共用同一个socket，
    // 交错写会产出非法分块，所以所有帧都必须走这里
    void WriteFrame(const std::string &payload) {
        std::lock_guard <std::mutex> guard(frameLocker);
        SendFrameLocked(payload);
    }

    // LaunchResponseTokens只是提交请求，真正耗时的是第一次FetchResponseTokens，
    // 所以在第一次fetch之前Start，返回后Stop
    void Start() {
        finished = false;
        start = std::chrono::steady_clock::now();
        lastFrame = start;
        worker = std::thread([this] () {
            while (!finished) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (finished) {
                    break;
                }
                int elapsed = (int)std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - start).count();
                if (!prefillDone) {
                    if (elapsed > 0 && elapsed % interval == 0) {
                        double rate = gPrefillTokenPerSec.load();
                        if (rate > 0.0 && missedTokens > 0) {
                            int percent = (int)std::min(99.0, 100.0 * elapsed * rate / missedTokens);
                            printf("[fastllm-api] prefill: %d tokens (needs %d), %d s elapsed, ~%d%%",
                                   totalTokens, missedTokens, elapsed, percent);
                        } else {
                            printf("[fastllm-api] prefill: %d tokens (needs %d), %d s elapsed",
                                   totalTokens, missedTokens, elapsed);
                        }
                        // 流式响应在首个token之前没有任何数据，客户端(或中间代理)的
                        // 首字节/空闲超时会先到并重连，这里发SSE注释帧保活(解析器会忽略)
                        SendCommentFrame(": prefill " + std::to_string(elapsed) + "s\n\n");
                        printf("\n");
                        fflush(stdout);
                    }
                    continue;
                }
                // 生成期间正常每出一个token就刷新一次lastFrame，只有chunk迟迟不来
                // 才补注释帧：单次decode可能到秒级，客户端空闲超时会当成流中断去重试
                if (IdleSeconds() >= interval) {
                    SendCommentFrame(": keep-alive\n\n");
                }
            }
        });
    }

    // prefill结束：停掉进度上报，线程继续跑做生成期的空闲保活。
    // 这里必须立刻取耗时，Stop()要等线程(最多再睡1s)会把prefill耗时算多
    int StopPrefill() {
        int spendMs = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
        if (spendMs > 0 && missedTokens > 0) {
            gPrefillTokenPerSec.store(missedTokens * 1000.0 / spendMs);
        }
        prefillDone = true;
        return spendMs;
    }

    // 返回从Start到现在的毫秒数
    int Stop() {
        finished = true;
        if (worker.joinable()) {
            worker.join();
        }
        int spendMs = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
        // 已经用StopPrefill报过prefill速率时不能再覆盖，否则后续请求的进度推算会用整条流的均速
        if (!prefillDone && spendMs > 0 && missedTokens > 0) {
            gPrefillTokenPerSec.store(missedTokens * 1000.0 / spendMs);
        }
        return spendMs;
    }

private:
    // 调用前必须持有frameLocker
    void SendFrameLocked(const std::string &payload) {
        char header[32];
        sprintf(header, "%zx\r\n", payload.size());
        SocketWrite(keepAliveClient, header, strlen(header));
        SocketWrite(keepAliveClient, payload.data(), payload.size());
        SocketWrite(keepAliveClient, "\r\n", 2);
        lastFrame = std::chrono::steady_clock::now();
    }

    void SendCommentFrame(const std::string &payload) {
        if (keepAliveClient == INVALID_SOCKET || !keepAliveEnabled) {
            return;
        }
        std::lock_guard <std::mutex> guard(frameLocker);
        SendFrameLocked(payload);
    }

    int IdleSeconds() {
        std::lock_guard <std::mutex> guard(frameLocker);
        return (int)std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - lastFrame).count();
    }

    int interval;
    socket_t keepAliveClient;
    int totalTokens;
    int missedTokens;
    std::atomic<bool> keepAliveEnabled;
    std::atomic<bool> finished;
    std::atomic<bool> prefillDone;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point lastFrame;
    std::mutex frameLocker;
    std::thread worker;
};

std::map <std::string, fastllm::DataType> dataTypeDict = {
    {"float32", fastllm::DataType::FLOAT32},
    {"half", fastllm::DataType::FLOAT16},
    {"float16", fastllm::DataType::FLOAT16},
    {"int8", fastllm::DataType::INT8},
    {"int4", fastllm::DataType::INT4_NOZERO},
    {"int4z", fastllm::DataType::INT4},
    {"int4g", fastllm::DataType::INT4_GROUP},
    {"bfloat16", fastllm::DataType::BFLOAT16},
    {"bf16", fastllm::DataType::BFLOAT16},
    {"fp8_e4m3", fastllm::DataType::FP8_E4M3},
    {"fp8", fastllm::DataType::FP8_E4M3},
    {"fp4", fastllm::DataType::FP4_E2M1}
};

// 去掉设备名里的引号、空格等装饰字符
static std::string StripDeviceToken(const std::string &text) {
    std::string ret;
    for (char c : text) {
        if (isalnum((unsigned char)c) || c == '_' || c == ':' || c == '-' || c == '.') {
            ret += c;
        }
    }
    return ret;
}

// --device / --moe_device 既支持单个设备名（cuda、cuda:0、cpu、numa），
// 也支持 ftllm 风格的比例分配（"{'cuda':1,'numa':8}"），逗号分隔的多个
// 设备名按等比例串行处理。
static std::map <std::string, int> ParseDeviceMap(const std::string &spec) {
    std::map <std::string, int> ret;
    std::string text = spec;
    size_t begin = text.find('{');
    size_t end = text.rfind('}');
    if (begin == std::string::npos || end == std::string::npos || end <= begin) {
        size_t pos = 0;
        while (pos <= text.size()) {
            size_t comma = text.find(',', pos);
            if (comma == std::string::npos) {
                comma = text.size();
            }
            std::string name = StripDeviceToken(text.substr(pos, comma - pos));
            if (!name.empty()) {
                ret[name] = 1;
            }
            pos = comma + 1;
        }
        return ret;
    }
    text = text.substr(begin + 1, end - begin - 1);
    size_t pos = 0;
    while (pos < text.size()) {
        size_t colon = text.find(':', pos);
        if (colon == std::string::npos) {
            break;
        }
        size_t comma = text.find(',', colon + 1);
        std::string name = StripDeviceToken(text.substr(pos, colon - pos));
        std::string value = text.substr(colon + 1, comma == std::string::npos ?
                                        std::string::npos : comma - colon - 1);
        if (!name.empty() && atoi(value.c_str()) > 0) {
            ret[name] = atoi(value.c_str());
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return ret;
}

// ftllm里 --cache_history 这类参数是字符串开关，"" 表示走模型默认值
static bool ParseBoolStr(const std::string &text) {
    std::string v;
    for (char c : text) {
        v += (char)tolower((unsigned char)c);
    }
    return v == "true" || v == "1" || v == "on" || v == "yes";
}

// --kv_cache_limit/--moe_cuda_cache/--moe_cpu_cache 支持 "10k"/"512m"/"3g" 这类写法
static long long ParseMemorySize(const std::string &text) {
    if (text.empty()) {
        return 0;
    }
    double scale = 1.0;
    std::string num = text;
    char last = (char)tolower((unsigned char)text[text.size() - 1]);
    if (last == 'k') {
        scale = 1e3;
        num = text.substr(0, text.size() - 1);
    } else if (last == 'm') {
        scale = 1e6;
        num = text.substr(0, text.size() - 1);
    } else if (last == 'g') {
        scale = 1e9;
        num = text.substr(0, text.size() - 1);
    }
    return (long long)(atof(num.c_str()) * scale);
}

// ftllm的_memory_size_bytes用二进制单位(1k=1024)，--moe_cuda_cache/--image_embedding_cache 走这个
static long long ParseBinaryMemorySize(const std::string &text) {
    std::string lower;
    for (char c : text) {
        lower += (char)tolower((unsigned char)c);
    }
    const char *suffixes[] = {"kib", "mib", "gib", "ki", "mi", "gi", "kb", "mb", "gb", "k", "m", "g"};
    long long factors[] = {1LL << 10, 1LL << 20, 1LL << 30, 1LL << 10, 1LL << 20, 1LL << 30,
                           1LL << 10, 1LL << 20, 1LL << 30, 1LL << 10, 1LL << 20, 1LL << 30};
    for (int i = 0; i < 12; i++) {
        size_t len = strlen(suffixes[i]);
        if (lower.size() > len && lower.compare(lower.size() - len, len, suffixes[i]) == 0) {
            return (long long)(atof(lower.substr(0, lower.size() - len).c_str()) * factors[i]);
        }
    }
    return (long long)atof(lower.c_str());
}

// 库里有部分开关只从环境变量读取，必须在建模型之前设置
static void SetEnvVar(const std::string &name, const std::string &value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

struct APIConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    std::string host = "0.0.0.0"; // 监听地址
    std::string modelName = "fastllm";
    std::string apiKey = ""; // 非空时校验 Authorization: Bearer <apiKey>

    int threads = 4; // 使用的线程数
    bool lowMemMode = false; // 是否使用低内存模式
    bool cudaEmbedding = false; // 是否使用cudaEmbedding
    int port = 8080; // 端口号
    int tokens = -1; // token容量限制
    int batch = 256; // 并发请求数上限

    fastllm::DataType dtype = fastllm::DataType::FLOAT16;
    fastllm::DataType atype = fastllm::DataType::FLOAT32;
    bool useAtype = false; // 仅在显式传入--atype时调用SetDataType
    int groupCnt = -1;
    std::string dtypeConfig = ""; // 权重类型配置文件
    std::string lora = ""; // lora路径

    std::string device = ""; // 主计算设备，例如 cuda、cuda:0、cpu、numa
    std::string moeDevice = ""; // MoE 专家层设备，例如 cpu、numa
    int moeDeviceLayers = -1; // 仅最后 N 层 MoE 使用 moeDevice，-1 表示全部
    fastllm::DataType moeDtype = fastllm::DataType::FLOAT32;
    int moeGroupCnt = -1;
    bool useMoeDtype = false;
    std::string ngramDevice = ""; // ngram表存放位置(cpu/disk)

    std::string moeAtypeStr = ""; // MoE激活类型，空表示不设置
    std::string kvCacheDtypeStr = ""; // KV Cache类型，空表示不设置
    long long moeCudaCache = 0; // MoE专家CUDA显存缓存，0关闭
    long long moeCpuCache = 0; // MoE专家内存缓存上限，0关闭

    int moeExperts = -1; // MoE使用的专家数
    int maxBatch = -1; // 引擎侧最大batch
    long long kvCacheLimit = 0; // KV Cache上限(字节)，0表示用库默认
    int chunkedPrefillSize = -1; // 分块prefill切片大小
    int moePinnedStagingSlots = -1; // MoE专家流式的中转槽数
    int moePinnedStagingSlotMb = -1; // MoE专家流式的中转单槽宽(MB)
    int maxContextLength = -1; // 单会话输入+输出最大token数
    std::string ropeScaling = ""; // RoPE扩展配置(yarn或JSON)
    int pageSize = -1; // paged cache每页token数
    float gpuMemRatio = -1.0f; // GPU显存使用比例，<0表示用库默认
    int cudaSlabMB = 0; // CUDA权重slab大小(MB)，0关闭
    std::string cudaSharedExpert = ""; // 是否用cuda执行共享专家，空表示不设置
    std::string enableAmx = ""; // 是否开启amx，空表示不设置

    std::string cacheHistory = ""; // 是否缓存历史对话
    std::string cacheFast = ""; // 是否启用快速缓存
    bool lowGpuMem = false; // 降低显存占用：强制关闭CUDA embedding，优先于--cuda_embedding
    bool fastPrefill = false; // DeepSeek-V4.1近似prefill
    std::string prefixCache = ""; // 前缀缓存开关，对应 FASTLLM_PREFIX_CACHE
    int prefixCacheSnapshotIntervalPages = -1;
    int prefixCacheSnapshotMaxPerRequest = -1;
    int prefixCacheSnapshotMaxRecords = -1;
    std::string visionDevice = ""; // Qwen3.5视觉编码器设备
    long long imageEmbeddingCache = -1; // 图片embedding的CPU缓存上限(字节)，-1表示不设置

    // 与ftllm的default_generation_config一致，CLI传了则覆盖，请求里传了再覆盖
    float temperature = 1.0f;
    float topP = 0.8f;
    int topK = 1;
    float repeatPenalty = 1.0f;
    // ftllm里CLI参数优先于模型自带的generation_config.json，这几个标记记录谁被显式指定过
    bool cliTemperature = false;
    bool cliTopP = false;
    bool cliTopK = false;
    bool cliRepeatPenalty = false;
    bool hideInput = false; // 不打印请求体
    bool devMode = false; // 开发模式(对话列表/主动停止)
    std::string toolCallParser = "auto"; // auto/qwen3_coder/hermes/none
    std::string embeddingModel = ""; // 可选的embedding/rerank模型(Bert)，用于/v1/embed与/v1/rerank
    int mtp = 0; // MTP草稿token数
    std::string mtpFp8DraftHead = ""; // 空表示不设置
    std::string tp = ""; // 线程张量并行配置，对应 FASTLLM_TP
    std::string dspark = ""; // DSpark草稿模型目录
    int draftTokens = -1;
    int speculativeNumDraftTokens = -1;
    std::string speculativeAlgorithm = "";
    std::string speculativeDraftModelPath = "";
    int speculativeDsparkBlockSize = -1;
    float speculativeDsparkConfidenceThreshold = -1.0f;
    bool triton = false; // 启用Triton CUDA算子
    std::string tritonPython = ""; // triton用的python解释器，不指定则自动探测
};
APIConfig config;

// /v1/embed与/v1/rerank用的Bert模型；不指定--embedding_model时这两个接口返回400
static std::unique_ptr<fastllm::BertModel> gEmbeddingModel;

void ToNext(char * &cur, const std::string &target, std::string &v) {
    v = "";
    while (*cur != 0) {
        bool stop = true;
        for (int i = 0; i < target.size(); i++) {
            if (cur[i] != target[i]) {
                stop = false;
                break;
            }
        }
        if (stop && target.size() > 0) {
            cur += target.size();
            break;
        } else {
            v += *(cur++);
        }
    }
}

// HTTP头名大小写不敏感，取值时统一按忽略大小写查找
static std::string GetHeaderValue(const std::unordered_map <std::string, std::string> &headers,
                                  const std::string &name) {
    for (auto &it : headers) {
        if (it.first.size() != name.size()) {
            continue;
        }
        bool same = true;
        for (size_t i = 0; i < name.size(); i++) {
            if (tolower((unsigned char)it.first[i]) != tolower((unsigned char)name[i])) {
                same = false;
                break;
            }
        }
        if (same) {
            return it.second;
        }
    }
    return "";
}

struct HttpRequest {
    std::string method;
    std::string route;
    std::string type;
    std::unordered_map <std::string, std::string> headers;
    std::string body;

    std::string GetHeader(const std::string &name) const {
        return GetHeaderValue(headers, name);
    }

    // 兼容 "Bearer <key>"、直接给 key，以及Anthropic风格的 x-api-key 头
    bool CheckApiKey(const std::string &apiKey) const {
        if (GetHeader("x-api-key") == apiKey) {
            return true;
        }
        std::string value = GetHeader("Authorization");
        size_t begin = value.find_first_not_of(" \t");
        if (begin == std::string::npos) {
            return false;
        }
        value = value.substr(begin);
        const char *prefix = "Bearer ";
        bool isBearer = value.size() > 7;
        for (int i = 0; isBearer && i < 7; i++) {
            if (tolower((unsigned char)value[i]) != tolower((unsigned char)prefix[i])) {
                isBearer = false;
            }
        }
        if (isBearer) {
            value = value.substr(7);
        }
        return value == apiKey;
    }

    void Init (char *buffer) {
        char *old = buffer;
        headers.clear();
        ToNext(buffer, " ", method);
        ToNext(buffer, " ", route);
        ToNext(buffer, "\r\n", type);
        while (true) {
            if (buffer[0] == 0 || ((long long)(buffer - old)) > 1024 * 1024) {
                break;
            }
            if (buffer[0] == '\r' && buffer[1] == '\n') {
                buffer += 2;
                ToNext(buffer, "", body);
                break;
            } else {
                std::string key;
                ToNext(buffer, ":", key);
                ToNext(buffer, "\r\n", headers[key]);
            }
        }
    }

    bool IsValid (char *buffer, int size) {
        char *old = buffer;
        headers.clear();
        ToNext(buffer, " ", method);
        ToNext(buffer, " ", route);
        ToNext(buffer, "\r\n", type);
        while (true) {
            if (buffer[0] == 0 || ((long long)(buffer - old)) > 1024 * 1024) {
                break;
            }
            if (buffer[0] == '\r' && buffer[1] == '\n') {
                // 头部结束：没有Content-Length(如GET、chunked请求)说明请求已经完整，
                // 这里必须返回，否则buffer不再前进会造成死循环
                std::string contentLength = GetHeaderValue(headers, "Content-Length");
                if (contentLength.empty()) {
                    return true;
                }
                return size - ((long long)(buffer - old)) - 2 >= (long long)atoi(contentLength.c_str());
            } else {
                std::string key;
                ToNext(buffer, ":", key);
                ToNext(buffer, "\r\n", headers[key]);
            }
        }
        return false;
    }

    void Print() {
        for (auto &it : headers) {
            printf("%s: %s\n", it.first.c_str(), it.second.c_str());
        }
        printf("body: %s\n", body.c_str());
    }
} httpChecker;

struct WorkNode {
    socket_t client;
    HttpRequest request;
    json11::Json config;
    std::string error;

    void Init(char *buffer, socket_t client) {
        this->client = client;
        request.Init(buffer);
        config = json11::Json::parse(request.body, this->error);
    }
};

// 词表上限，用于提前校验token
static int GetVocabSize(fastllm::basellm *model) {
    static int cached = -1;
    if (cached < 0) {
        int mx = 0;
        for (auto &it : model->weight.tokenizer.tokenToStringDict) {
            mx = std::max(mx, it.first);
        }
        cached = mx + 1;
    }
    return cached;
}

// 把Encode的结果转成token ids。库在token越界时会走ErrorInFastLLM，里面是getchar()，
// 服务进程会永久卡在那里等按键，所以这里必须先校验并拦掉。
static bool BuildTokens(fastllm::basellm *model, const fastllm::Data &inputs,
                        std::vector<int> &tokens, std::string &error) {
    int vocabSize = GetVocabSize(model);
    for (int i = 0; i < inputs.Count(0); i++) {
        float raw = ((float *) inputs.cpuData)[i];
        int id = (int) (raw + 1e-9); // 与库内getToken保持一致的取整方式
        tokens.push_back(id);
        if (error.empty() && (id < 0 || id >= vocabSize)) {
            error = "prompt token out of range: index " + std::to_string(i) +
                    ", token " + std::to_string(id) +
                    ", raw " + std::to_string(raw) +
                    ", vocabSize " + std::to_string(vocabSize);
        }
    }
    printf("[fastllm-api] tokens: count=%d vocab=%d head=", (int)tokens.size(), vocabSize);
    for (int i = 0; i < std::min(8, (int)tokens.size()); i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n");
    fflush(stdout);
    return error.empty();
}

// 多token的stop串没法交给引擎，只能在解码文本上截断(对应ftllm的_truncate_at_stop)
static bool TruncateAtStop(std::string &text, const std::vector <std::string> &stopStrings) {
    size_t stopPos = std::string::npos;
    for (auto &stop : stopStrings) {
        size_t pos = text.find(stop);
        if (pos != std::string::npos && (stopPos == std::string::npos || pos < stopPos)) {
            stopPos = pos;
        }
    }
    if (stopPos == std::string::npos) {
        return false;
    }
    text = text.substr(0, stopPos);
    return true;
}

// ---------------- tool calls ----------------
// 以下实现对应ftllm的openai_server/tool_schema.py与qwen3coder_tool_parser.py，
// 覆盖Qwen3-Coder的XML调用格式：<tool_call><function=NAME><parameter=P>V</parameter></function></tool_call>

struct ToolCallInfo {
    std::string name;
    std::string arguments; // 参数对象的JSON字符串
};

enum ToolCallParserType {
    TOOL_CALL_PARSER_NONE = 0,
    TOOL_CALL_PARSER_QWEN3_CODER = 1,
    TOOL_CALL_PARSER_HERMES = 2
};

// 启动时解析一次，请求里直接用
static ToolCallParserType gToolCallParser = TOOL_CALL_PARSER_NONE;

static std::string ToLowerString(const std::string &text) {
    std::string ret;
    for (char c : text) {
        ret += (char)tolower((unsigned char)c);
    }
    return ret;
}

static std::string TrimString(const std::string &text) {
    size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

static bool IsBlankString(const std::string &text) {
    return text.find_first_not_of(" \t\r\n") == std::string::npos;
}

static size_t SkipWhitespace(const std::string &text, size_t cursor) {
    while (cursor < text.size() && (text[cursor] == ' ' || text[cursor] == '\t' ||
                                    text[cursor] == '\r' || text[cursor] == '\n')) {
        cursor++;
    }
    return cursor;
}

static bool StartsWithAt(const std::string &text, size_t cursor, const std::string &token) {
    return text.size() >= cursor + token.size() && text.compare(cursor, token.size(), token) == 0;
}

// 输出可能正好停在标记中间，这种情况要按"不完整"处理而不是解析失败
static bool IsPartialTokenAt(const std::string &text, size_t cursor, const std::string &token) {
    if (cursor > text.size()) {
        return false;
    }
    size_t available = text.size() - cursor;
    return available < token.size() &&
           token.compare(0, available, text, cursor, available) == 0;
}

static bool ContainsString(const std::vector <std::string> &values, const std::string &value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

static json11::Json ResolveSchemaRef(const json11::Json &root, const std::string &ref) {
    if (ref.size() < 3 || ref.compare(0, 2, "#/") != 0) {
        return json11::Json();
    }
    json11::Json current = root;
    size_t pos = 2;
    while (pos <= ref.size()) {
        size_t slash = ref.find('/', pos);
        std::string key = ref.substr(pos, slash == std::string::npos ? std::string::npos : slash - pos);
        if (!current.is_object()) {
            return json11::Json();
        }
        current = current[key];
        if (slash == std::string::npos) {
            break;
        }
        pos = slash + 1;
    }
    return current;
}

// 对应ftllm的schema_types：按JSON Schema取出允许的类型，保持稳定顺序
static std::vector <std::string> SchemaTypes(const json11::Json &schema, const json11::Json &root, int depth = 0) {
    std::vector <std::string> result;
    if (depth > 16 || !schema.is_object()) {
        return result;
    }
    json11::Json candidate = schema;
    if (candidate["$ref"].is_string()) {
        json11::Json resolved = ResolveSchemaRef(root, candidate["$ref"].string_value());
        if (resolved.is_object()) {
            candidate = resolved;
        }
    }
    auto appendType = [&result](const std::string &type) {
        if (!type.empty() && !ContainsString(result, type)) {
            result.push_back(type);
        }
    };
    auto normalizeType = [](const json11::Json &value) {
        std::string type = value.is_string() ? value.string_value() : "";
        if (type == "integer" || type == "number" || type == "string" || type == "boolean" ||
            type == "array" || type == "object" || type == "null") {
            return type;
        }
        return std::string("");
    };
    json11::Json declared = candidate["type"];
    if (declared.is_array()) {
        for (auto &it : declared.array_items()) {
            appendType(normalizeType(it));
        }
    } else if (!declared.is_null()) {
        appendType(normalizeType(declared));
    }
    if (candidate["nullable"].is_bool() && candidate["nullable"].bool_value()) {
        appendType("null");
    }
    for (const char *keyword : {"anyOf", "oneOf", "allOf"}) {
        for (auto &alternative : candidate[keyword].array_items()) {
            for (auto &type : SchemaTypes(alternative, root, depth + 1)) {
                appendType(type);
            }
        }
    }
    if (declared.is_null()) {
        if (candidate["properties"].is_object()) {
            appendType("object");
        } else if (!candidate["items"].is_null()) {
            appendType("array");
        }
    }
    return result;
}

static bool ParseStrictInteger(const std::string &text, long long &value) {
    if (text.empty()) {
        return false;
    }
    size_t i = 0;
    bool negative = false;
    if (text[0] == '+' || text[0] == '-') {
        negative = (text[0] == '-');
        i = 1;
    }
    if (i >= text.size()) {
        return false;
    }
    long long result = 0;
    for (; i < text.size(); i++) {
        if (!isdigit((unsigned char)text[i])) {
            return false;
        }
        result = result * 10 + (text[i] - '0');
    }
    value = negative ? -result : result;
    return true;
}

static json11::Json MakeJsonNumber(double value) {
    // json11没有long long构造，整数在int范围内就按int走
    if (value == (double)(long long)value && value >= -2147483648.0 && value <= 2147483647.0) {
        return json11::Json((int)value);
    }
    return json11::Json(value);
}

// 对应ftllm的convert_text_value：按schema转换，转换失败保留原字符串
static json11::Json ConvertTextValue(const std::string &value, const json11::Json &schema, const json11::Json &root) {
    std::vector <std::string> types = SchemaTypes(schema, root, 0);
    if (types.empty()) {
        return json11::Json(value);
    }
    std::string stripped = TrimString(value);
    std::string lowered = ToLowerString(stripped);
    if (lowered == "null" && ContainsString(types, "null")) {
        return json11::Json();
    }
    for (auto &type : types) {
        if (type == "null") {
            continue;
        }
        if (type == "string") {
            return json11::Json(value);
        } else if (type == "integer") {
            long long number = 0;
            if (ParseStrictInteger(stripped, number)) {
                return MakeJsonNumber((double)number);
            }
        } else if (type == "number") {
            if (!stripped.empty()) {
                char *tail = nullptr;
                double number = strtod(stripped.c_str(), &tail);
                if (tail != nullptr && *tail == 0) {
                    return MakeJsonNumber(number);
                }
            }
        } else if (type == "boolean") {
            if (lowered == "true") {
                return json11::Json(true);
            }
            if (lowered == "false") {
                return json11::Json(false);
            }
        } else if (type == "array" || type == "object") {
            std::string parseError;
            json11::Json parsed = json11::Json::parse(stripped, parseError);
            if (parseError.empty()) {
                if (type == "array" && parsed.is_array()) {
                    return parsed;
                }
                if (type == "object" && parsed.is_object()) {
                    return parsed;
                }
            }
        }
    }
    return json11::Json(value);
}

static json11::Json ToolParameters(const std::string &functionName, const json11::Json &tools) {
    for (auto &tool : tools.array_items()) {
        json11::Json function = tool["function"];
        if (function["name"].string_value() != functionName) {
            continue;
        }
        json11::Json parameters = function["parameters"];
        return parameters.is_object() ? parameters : json11::Json::object {};
    }
    return json11::Json::object {};
}

static json11::Json ParameterSchema(const std::string &parameterName, const json11::Json &parameters) {
    json11::Json properties = parameters["properties"];
    if (properties.is_object() && !properties[parameterName].is_null()) {
        return properties[parameterName];
    }
    json11::Json additional = parameters["additionalProperties"];
    if (additional.is_object()) {
        return additional;
    }
    return json11::Json();
}

static bool ParseNamedOpening(const std::string &text, size_t cursor, const std::string &prefix,
                              std::string &name, size_t &next) {
    if (!StartsWithAt(text, cursor, prefix)) {
        return false;
    }
    size_t nameStart = cursor + prefix.size();
    size_t nameEnd = text.find('>', nameStart);
    if (nameEnd == std::string::npos) {
        return false;
    }
    std::string raw = text.substr(nameStart, nameEnd - nameStart);
    if (raw.find('\n') != std::string::npos || raw.find('\r') != std::string::npos) {
        return false;
    }
    name = TrimString(raw);
    next = nameEnd + 1;
    return !name.empty();
}

// </parameter>只有在后面能继续wire语法(另一个parameter或function结束)时才算结构分隔符，
// 否则参数值里出现的</parameter>字样会被当成普通文本
static bool FindParameterEnd(const std::string &text, size_t valueStart, size_t &end) {
    static const std::string parameterEnd = "</parameter>";
    static const std::string parameterPrefix = "<parameter=";
    static const std::string functionEnd = "</function>";
    size_t searchFrom = valueStart;
    while (true) {
        size_t found = text.find(parameterEnd, searchFrom);
        if (found == std::string::npos) {
            return false;
        }
        size_t continuation = SkipWhitespace(text, found + parameterEnd.size());
        if (StartsWithAt(text, continuation, parameterPrefix) ||
            StartsWithAt(text, continuation, functionEnd) ||
            StartsWithAt(text, continuation, "</tool_call>")) {
            end = found;
            return true;
        }
        searchFrom = found + parameterEnd.size();
    }
}

static bool ParseParameterAt(const std::string &text, size_t cursor,
                             std::string &name, std::string &rawValue, size_t &next) {
    static const std::string parameterEnd = "</parameter>";
    size_t valueStart = 0;
    if (!ParseNamedOpening(text, cursor, "<parameter=", name, valueStart)) {
        return false;
    }
    size_t valueEnd = 0;
    if (!FindParameterEnd(text, valueStart, valueEnd)) {
        return false;
    }
    rawValue = text.substr(valueStart, valueEnd - valueStart);
    // 协议换行不算参数值的一部分，各去掉一个
    if (!rawValue.empty() && rawValue[0] == '\n') {
        rawValue = rawValue.substr(1);
    }
    if (!rawValue.empty() && rawValue[rawValue.size() - 1] == '\n') {
        rawValue = rawValue.substr(0, rawValue.size() - 1);
    }
    next = valueEnd + parameterEnd.size();
    return true;
}

static bool BuildToolCall(const std::string &functionName,
                          const std::vector <std::pair <std::string, std::string> > &rawArguments,
                          const json11::Json &tools, ToolCallInfo &info) {
    json11::Json parameters = ToolParameters(functionName, tools);
    json11::Json properties = parameters["properties"];
    json11::Json::object arguments;
    std::set <std::string> usedNames;
    for (auto &argument : rawArguments) {
        if (usedNames.count(argument.first) > 0) {
            return false; // 同名参数重复，按解析失败处理
        }
        usedNames.insert(argument.first);
        json11::Json schema = ParameterSchema(argument.first, parameters);
        if (schema.is_null()) {
            arguments[argument.first] = json11::Json(argument.second);
        } else {
            arguments[argument.first] = ConvertTextValue(argument.second, schema, parameters);
        }
    }
    info.name = functionName;
    info.arguments = json11::Json(arguments).dump();
    return true;
}

static bool ParseFunctionAt(const std::string &text, size_t cursor,
                            const json11::Json &tools, ToolCallInfo &info, size_t &next) {
    static const std::string functionEnd = "</function>";
    std::string functionName;
    if (!ParseNamedOpening(text, cursor, "<function=", functionName, cursor)) {
        return false;
    }
    std::vector <std::pair <std::string, std::string> > rawArguments;
    while (true) {
        cursor = SkipWhitespace(text, cursor);
        if (StartsWithAt(text, cursor, functionEnd)) {
            next = cursor + functionEnd.size();
            return BuildToolCall(functionName, rawArguments, tools, info);
        }
        if (StartsWithAt(text, cursor, "<parameter=")) {
            std::string name, rawValue;
            if (!ParseParameterAt(text, cursor, name, rawValue, cursor)) {
                return false;
            }
            rawArguments.push_back(std::make_pair(name, rawValue));
            continue;
        }
        return false;
    }
}

static bool ParseToolBlock(const std::string &text, size_t cursor,
                           const json11::Json &tools,
                           std::vector <ToolCallInfo> &calls, size_t &next) {
    static const std::string toolCallEnd = "</tool_call>";
    cursor = SkipWhitespace(text, cursor);
    if (!StartsWithAt(text, cursor, "<tool_call>")) {
        return false;
    }
    cursor += std::string("<tool_call>").size();
    while (true) {
        cursor = SkipWhitespace(text, cursor);
        if (StartsWithAt(text, cursor, toolCallEnd)) {
            if (calls.empty()) {
                return false;
            }
            next = cursor + toolCallEnd.size();
            return true;
        }
        if (StartsWithAt(text, cursor, "<function=")) {
            ToolCallInfo info;
            size_t functionNext = 0;
            if (!ParseFunctionAt(text, cursor, tools, info, functionNext)) {
                return false;
            }
            calls.push_back(info);
            cursor = functionNext;
            continue;
        }
        return false;
    }
}

// 解析完整文本里的工具调用；content只保留工具调用之外的非空文本(与ftllm一致)
static void ParseQwen3ToolCalls(const std::string &text, const json11::Json &tools,
                                std::string &content, std::vector <ToolCallInfo> &calls) {
    static const std::string toolCallStart = "<tool_call>";
    static const std::string toolCallEnd = "</tool_call>";
    content = text;
    calls.clear();
    size_t markerIndex = text.find(toolCallStart);
    if (markerIndex == std::string::npos) {
        return;
    }
    std::string normalText;
    size_t cursor = 0;
    bool failed = false;
    while (true) {
        size_t start = text.find(toolCallStart, cursor);
        size_t strayEnd = text.find(toolCallEnd, cursor);
        if (start == std::string::npos) {
            if (strayEnd != std::string::npos) {
                failed = true;
                break;
            }
            std::string tail = text.substr(cursor);
            if (!IsBlankString(tail)) {
                normalText += tail;
            }
            break;
        }
        if (strayEnd != std::string::npos && strayEnd < start) {
            failed = true;
            break;
        }
        std::string segment = text.substr(cursor, start - cursor);
        if (!IsBlankString(segment)) {
            normalText += segment;
        }
        size_t blockNext = 0;
        if (!ParseToolBlock(text, start, tools, calls, blockNext)) {
            failed = true;
            break;
        }
        cursor = blockNext;
    }
    if (failed) {
        // 解析失败按"没有调用工具"处理，只保留标记前面的文本
        calls.clear();
        std::string prefix = text.substr(0, markerIndex);
        content = IsBlankString(prefix) ? "" : prefix;
        return;
    }
    content = IsBlankString(normalText) ? "" : normalText;
}

// hermes格式：<tool_call>{"name":..., "arguments":{...}}</tool_call>
static void ParseHermesToolCalls(const std::string &text, std::string &content,
                                 std::vector <ToolCallInfo> &calls) {
    static const std::string toolCallStart = "<tool_call>";
    static const std::string toolCallEnd = "</tool_call>";
    content = text;
    calls.clear();
    size_t firstStart = text.find(toolCallStart);
    if (firstStart == std::string::npos) {
        return;
    }
    size_t cursor = firstStart;
    while (true) {
        size_t start = text.find(toolCallStart, cursor);
        if (start == std::string::npos) {
            break;
        }
        size_t end = text.find(toolCallEnd, start + toolCallStart.size());
        std::string payload = (end == std::string::npos) ?
                              text.substr(start + toolCallStart.size()) :
                              text.substr(start + toolCallStart.size(),
                                           end - start - toolCallStart.size());
        cursor = (end == std::string::npos) ? text.size() : end + toolCallEnd.size();
        std::string parseError;
        json11::Json parsed = json11::Json::parse(TrimString(payload), parseError);
        if (!parseError.empty() || !parsed.is_object() ||
            !parsed["name"].is_string() || parsed["arguments"].is_null()) {
            // 与ftllm一致：解析失败整体按"没有调用工具"处理
            calls.clear();
            content = text;
            return;
        }
        ToolCallInfo info;
        info.name = parsed["name"].string_value();
        info.arguments = parsed["arguments"].dump();
        calls.push_back(info);
    }
    if (calls.empty()) {
        content = text;
        return;
    }
    content = text.substr(0, firstStart);
}

static void ParseToolCalls(ToolCallParserType parser, const std::string &text,
                           const json11::Json &tools, std::string &content,
                           std::vector <ToolCallInfo> &calls) {
    if (parser == TOOL_CALL_PARSER_HERMES) {
        ParseHermesToolCalls(text, content, calls);
    } else {
        ParseQwen3ToolCalls(text, tools, content, calls);
    }
}

// 与ftllm的ToolParserManager.get_tool_parser_auto一致：
// 指定了就用指定的；auto先看chat template里的标记，再按模型家族决定，判断不出来用hermes
static ToolCallParserType ResolveToolCallParser(fastllm::basellm *model) {
    std::string force = ToLowerString(config.toolCallParser);
    if (force == "none" || force == "off") {
        printf("[fastllm] tool parser: disabled\n");
        return TOOL_CALL_PARSER_NONE;
    }
    // qwen3.5的模板就是Qwen XML格式，所以它和qwen3_coder是同一个parser
    if (force == "qwen3_coder" || force == "qwen3coder" || force == "qwen_xml" || force == "qwen25" ||
        force == "qwen3_5" || force == "qwen3_5_text" || force == "qwen3_5_moe" ||
        force == "qwen35" || force == "qwen3.5") {
        printf("[fastllm] tool parser: qwen3_coder\n");
        return TOOL_CALL_PARSER_QWEN3_CODER;
    }
    if (force == "hermes" || force == "hermes2pro" || force == "hermes_2_pro" || force == "default") {
        printf("[fastllm] tool parser: hermes\n");
        return TOOL_CALL_PARSER_HERMES;
    }
    if (force != "auto" && force != "") {
        // 库里还有一堆按模型家族分的parser，apiserver没移植，直接报错好过按错格式解析
        printf("[fastllm] --tool_call_parser %s 暂不支持，可选: auto/qwen3_coder/hermes/none\n",
               config.toolCallParser.c_str());
        exit(-1);
    }

    const std::string &chatTemplate = model->weight.tokenizer.chatTemplate;
    if (chatTemplate.find("<function=") != std::string::npos ||
        chatTemplate.find("<parameter=") != std::string::npos ||
        chatTemplate.find("qwen3_coder") != std::string::npos) {
        printf("[fastllm] auto tool parser: qwen3_coder\n");
        return TOOL_CALL_PARSER_QWEN3_CODER;
    }
    std::string modelType = ToLowerString(model->model_type);
    // 这些家族的格式(DSML/dots/glm/minimax/kimi/hy_v3等)还没移植，回退到hermes；
    // hermes只认<tool_call>{json}，所以实际等于不做工具解析，只会原样返回文本
    static const char *unsupportedFamilies[] = {
        "deepseek_v4", "deepseek_v41", "deepseek_v41_text", "deepseek_v3", "deepseek_v2",
        "glm_moe_dsa", "glm5_next", "glm5_next_text", "glm4_moe", "minimax_m2", "minimax",
        "kimi_k2", "kimi_k3", "hy_v3", "dots3_note", "laguna", "granite", "internlm2",
        "jamba", "llama", "llama4", "mistral", "phi4mini", "step3", "xlam", "hunyuan_a13b",
        "poolside_v1", "granite_20b_fc"
    };
    for (const char *family : unsupportedFamilies) {
        if (modelType == family) {
            printf("[fastllm] 模型家族 %s 的工具调用格式未移植，回退hermes(仅识别<tool_call>{json})\n",
                   modelType.c_str());
            return TOOL_CALL_PARSER_HERMES;
        }
    }
    static const char *qwenFamilies[] = {
        "qwen2", "qwen3", "qwen3_moe", "qwen3_next", "qwen3_5", "qwen3_5_text",
        "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp", "qwen4_exp_text",
        "qwen3_8_flash_next"
    };
    for (const char *family : qwenFamilies) {
        if (modelType == family) {
            // 与ftllm一致：qwen家族但模板不是XML格式时用hermes
            printf("[fastllm] auto tool parser: hermes (qwen家族非XML模板)\n");
            return TOOL_CALL_PARSER_HERMES;
        }
    }
    printf("[fastllm] 无法判断工具调用格式(模型类型 %s)，使用默认hermes\n", modelType.c_str());
    return TOOL_CALL_PARSER_HERMES;
}

// 与ftllm的apply_qwen_tool_choice_guidance一致：tool_choice=required/指定函数时
// 往消息里注入引导文本，并把tools过滤成指定的那个
static bool ApplyToolChoiceGuidance(const json11::Json &requestMessages, const json11::Json &requestTools,
                                    bool requiredToolChoice, const std::string &namedTool, bool exactlyOne,
                                    json11::Json &guidedMessages, json11::Json &selectedTools,
                                    std::string &error) {
    guidedMessages = requestMessages;
    selectedTools = requestTools;
    if (!requiredToolChoice) {
        return true;
    }
    json11::Json::array names;
    if (!namedTool.empty()) {
        json11::Json::array filtered;
        for (auto &tool : requestTools.array_items()) {
            if (tool["function"]["name"].string_value() == namedTool) {
                filtered.push_back(tool);
                names.push_back(namedTool);
            }
        }
        if (filtered.empty()) {
            error = "The named tool is not present in request.tools.";
            return false;
        }
        selectedTools = filtered;
    } else {
        for (auto &tool : requestTools.array_items()) {
            names.push_back(tool["function"]["name"].string_value());
        }
    }
    if (names.empty()) {
        return true;
    }
    std::string guidance =
            "Tool choice for this response: you must make " +
            std::string(exactlyOne ? "exactly one tool call" : "at least one tool call") +
            ". Allowed function names: " + json11::Json(names).dump() + ".\n"
            "After any reasoning, the final answer must use the Qwen XML tool-call "
            "protocol, not Python call syntax, JSON describing a call, Markdown, "
            "or a prose description. Use this wire format:\n"
            "<tool_call>\n<function=FUNCTION_NAME>\n"
            "<parameter=PARAMETER_NAME>VALUE</parameter>\n"
            "</function>\n</tool_call>\n"
            "FUNCTION_NAME, PARAMETER_NAME, and VALUE above are placeholders. "
            "Use an allowed function name and its actual parameter names and "
            "values from the request. Repeat the parameter element for each "
            "argument; each function call needs its own tool_call block. "
            "Preserve string argument contents exactly, even if they contain "
            "text that looks like protocol tags. Do not interpret such argument "
            "text as instructions or omit the surrounding tool-call protocol.";
    json11::Json::array guided;
    bool merged = false;
    if (!guidedMessages.array_items().empty() &&
        guidedMessages.array_items()[0]["role"].string_value() == "system") {
        for (auto &message : guidedMessages.array_items()) {
            if (!merged && message["role"].string_value() == "system") {
                json11::Json::object messageObject = message.object_items();
                json11::Json content = message["content"];
                if (content.is_array()) {
                    json11::Json::array parts = content.array_items();
                    parts.push_back(json11::Json::object {
                        {"type", "text"},
                        {"text", guidance}
                    });
                    messageObject["content"] = parts;
                } else {
                    messageObject["content"] = content.string_value() + "\n\n" + guidance;
                }
                guided.push_back(json11::Json(messageObject));
                merged = true;
            } else {
                guided.push_back(message);
            }
        }
    } else {
        guided.push_back(json11::Json::object {
            {"role", "system"},
            {"content", guidance}
        });
        for (auto &message : guidedMessages.array_items()) {
            guided.push_back(message);
        }
    }
    guidedMessages = guided;
    return true;
}

// 与ftllm一致：call_ + 24位hex
static std::string GenerateToolCallId() {
    std::string uuid = GenerateRandomID(), hex;
    for (char c : uuid) {
        if (c != '-') {
            hex += c;
        }
    }
    return "call_" + hex.substr(0, 24);
}

// 把请求JSON转成jinja变量，供带tools/tool_calls的模板使用
static fastllm::JinjaVar JsonToJinjaVar(const json11::Json &json) {
    if (json.is_null()) {
        return fastllm::JinjaVar();
    }
    if (json.is_bool()) {
        return fastllm::JinjaVar((int)(json.bool_value() ? 1 : 0));
    }
    if (json.is_number()) {
        double value = json.number_value();
        if (value == (double)(long long)value && value >= -2147483648.0 && value <= 2147483647.0) {
            return fastllm::JinjaVar((int)value);
        }
        return fastllm::JinjaVar((float)value);
    }
    if (json.is_string()) {
        return fastllm::JinjaVar(json.string_value());
    }
    if (json.is_array()) {
        std::vector <fastllm::JinjaVar> array;
        for (auto &item : json.array_items()) {
            array.push_back(JsonToJinjaVar(item));
        }
        return fastllm::JinjaVar(array);
    }
    fastllm::JinjaVar dict(fastllm::JinjaVar::JinjaDict, "");
    for (auto &item : json.object_items()) {
        dict.dictValue[item.first] = JsonToJinjaVar(item.second);
    }
    return dict;
}

// 对应ftllm的ChatMessagesToJinjaVar，额外带上tools(模板里的<tools>区块就靠它)
static fastllm::JinjaVar BuildChatTemplateVars(const json11::Json &requestMessages, const json11::Json &tools) {
    std::vector <fastllm::JinjaVar> messageVars;
    for (auto &message : requestMessages.array_items()) {
        json11::Json messageJson = message;
        if (message["tool_calls"].is_array()) {
            // OpenAI协议里function.arguments是字符串，但模板用arguments|items遍历它，
            // 传字符串会触发引擎断言(断言里有getchar，会把服务卡死)，必须先转成对象
            json11::Json::object messageObject = message.object_items();
            json11::Json::array toolCalls;
            for (auto &toolCall : message["tool_calls"].array_items()) {
                json11::Json::object toolCallObject = toolCall.object_items();
                json11::Json function = toolCall["function"];
                if (function.is_object() && function["arguments"].is_string()) {
                    std::string parseError;
                    json11::Json parsed = json11::Json::parse(function["arguments"].string_value(), parseError);
                    if (parseError.empty()) {
                        json11::Json::object functionObject = function.object_items();
                        functionObject["arguments"] = parsed;
                        toolCallObject["function"] = json11::Json(functionObject);
                    }
                }
                toolCalls.push_back(json11::Json(toolCallObject));
            }
            messageObject["tool_calls"] = toolCalls;
            messageJson = json11::Json(messageObject);
        }
        fastllm::JinjaVar msg = JsonToJinjaVar(messageJson);
        // 与ftllm一致：assistant消息里的<think>拆到reasoning_content
        if (message["role"].string_value() == "assistant" &&
            message["reasoning_content"].is_null() && message["content"].is_string()) {
            std::string content = message["content"].string_value();
            size_t thinkEnd = content.find("</think>");
            if (thinkEnd != std::string::npos) {
                size_t thinkStart = content.find("<think>");
                if (thinkStart != std::string::npos && thinkStart < thinkEnd) {
                    msg["reasoning_content"] = content.substr(thinkStart + 7, thinkEnd - thinkStart - 7);
                }
            }
        }
        messageVars.push_back(msg);
    }
    fastllm::JinjaVar ret(fastllm::JinjaVar::JinjaDict, "");
    ret.dictValue["messages"] = fastllm::JinjaVar(messageVars);
    ret.dictValue["add_generation_prompt"] = fastllm::JinjaVar((int)1);
    ret.dictValue["tools"] = tools.is_array() ? JsonToJinjaVar(tools) :
                             fastllm::JinjaVar(std::vector <fastllm::JinjaVar>());
    return ret;
}

// 带tools或带tool_calls/tool消息时必须走jinja变量版本，pair版本的ChatMessages表达不了这些
static bool NeedChatTemplateVars(const json11::Json &requestMessages, const json11::Json &tools) {
    if (tools.is_array() && !tools.array_items().empty()) {
        return true;
    }
    for (auto &message : requestMessages.array_items()) {
        if (message["tool_calls"].is_array() || message["role"].string_value() == "tool") {
            return true;
        }
    }
    return false;
}

// 分块传输的SSE帧
static void WriteChunkedFrame(socket_t client, const std::string &payload) {
    char header[32];
    sprintf(header, "%zx\r\n", payload.size());
    SocketWrite(client, header, strlen(header));
    SocketWrite(client, payload.data(), payload.size());
    SocketWrite(client, "\r\n", 2);
}

// 统一的JSON响应头(带CORS)。ftllm那边是starlette的JSONResponse，一定会带Content-Length，
// 这里保持一致，不让body长度依赖"连接关闭"来界定
static void AppendJsonReplyHeaders(std::string &out, size_t bodySize) {
    out += "Content-Type:application/json\r\n";
    out += "server:fastllm api server\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n";
    out += "Content-Length: " + std::to_string(bodySize) + "\r\n";
    out += "\r\n";
}

// 统一的JSON响应(带CORS头)
static void SendJsonReply(socket_t client, const char *status, const std::string &body) {
    std::string message = status;
    AppendJsonReplyHeaders(message, body.size());
    message += body;
    SocketWrite(client, message.c_str(), message.length());
    SocketClose(client);
}

// 跑一条命令看退出码，用来探测某个python里有没有triton
static int RunCommandExitCode(const std::string &command) {
#ifdef _WIN32
    FILE *pipe = _popen(command.c_str(), "r");
#else
    FILE *pipe = popen(command.c_str(), "r");
#endif
    if (pipe == nullptr) {
        return -1;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    }
#ifdef _WIN32
    return _pclose(pipe);
#else
    return pclose(pipe);
#endif
}

// PATH扫描里不能用std::filesystem(fastllm::FileExists)：PATH里可能有带引号之类的
// 非法条目，fs::exists在Windows上会抛异常。这里用fopen做存在性判断
static bool PathIsReadableFile(const std::string &path) {
    FILE *file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return false;
    }
    fclose(file);
    return true;
}

// 在PATH里找可执行文件；Windows下会补.exe
static std::string FindExecutableInPath(const std::string &name) {
    const char *pathEnv = getenv("PATH");
    if (pathEnv == nullptr) {
        return "";
    }
    std::string pathText = pathEnv;
    size_t pos = 0;
    while (pos <= pathText.size()) {
#ifdef _WIN32
        size_t sep = pathText.find(';', pos); // Windows下不能按':'切，盘符里也有
#else
        size_t sep = pathText.find(':', pos);
#endif
        std::string dir = pathText.substr(pos, sep == std::string::npos ? std::string::npos : sep - pos);
        if (!dir.empty()) {
            std::string candidate = dir + "/" + name;
            if (PathIsReadableFile(candidate)) {
                return candidate;
            }
#ifdef _WIN32
            if (PathIsReadableFile(candidate + ".exe")) {
                return candidate + ".exe";
            }
#endif
        }
        if (sep == std::string::npos) {
            break;
        }
        pos = sep + 1;
    }
    return "";
}

static int RunPythonImportCheck(const std::string &python, const std::string &module);

// 与ftllm的_find_triton_python等价：ftllm里就是"当前解释器能import triton"，
// apiserver没有python解释器，所以按候选顺序找一个能import triton的
static std::string FindTritonPython() {
    std::vector <std::string> candidates;
    if (!config.tritonPython.empty()) {
        candidates.push_back(config.tritonPython);
    }
    const char *envPython = getenv("FASTLLM_CUDA_TRITON_PYTHON");
    if (envPython != nullptr && envPython[0] != 0) {
        candidates.push_back(envPython);
    }
    for (const char *name : {"python", "python3"}) {
        std::string found = FindExecutableInPath(name);
        if (!found.empty()) {
            candidates.push_back(found);
        }
    }
    for (auto &candidate : candidates) {
        if (PathIsReadableFile(candidate) && RunPythonImportCheck(candidate, "triton") == 0) {
            return candidate;
        }
    }
    return "";
}

// 探测某个python能不能import指定模块。要避开两个坑：
// 1) cmd的引号处理会把命令行拼错，2) 子进程的报错会刷屏，所以直接CreateProcess并把输出丢到NUL
static int RunPythonImportCheck(const std::string &python, const std::string &module) {
    // 不能带空格：spawn/CreateProcess都不会给带空格的参数加引号；也不能用import(x)，
    // py2.7下是语法错误，所以用__import__
    std::string code = "__import__('" + module + "')";
#ifdef _WIN32
    std::string commandLine = "\"" + python + "\" -c \"" + code + "\"";
    SECURITY_ATTRIBUTES securityAttributes = {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE nulHandle = CreateFileA("NUL", GENERIC_READ | GENERIC_WRITE,
                                   FILE_SHARE_READ | FILE_SHARE_WRITE, &securityAttributes,
                                   OPEN_EXISTING, 0, nullptr);
    STARTUPINFOA startupInfo = {};
    startupInfo.cb = sizeof(startupInfo);
    if (nulHandle != INVALID_HANDLE_VALUE) {
        startupInfo.dwFlags = STARTF_USESTDHANDLES;
        startupInfo.hStdInput = nulHandle;
        startupInfo.hStdOutput = nulHandle;
        startupInfo.hStdError = nulHandle;
    }
    std::vector <char> mutableCommand(commandLine.begin(), commandLine.end());
    mutableCommand.push_back(0);
    PROCESS_INFORMATION processInfo = {};
    DWORD exitCode = 1;
    if (CreateProcessA(nullptr, mutableCommand.data(), nullptr, nullptr, TRUE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &startupInfo, &processInfo)) {
        WaitForSingleObject(processInfo.hProcess, 30000);
        GetExitCodeProcess(processInfo.hProcess, &exitCode);
        CloseHandle(processInfo.hThread);
        CloseHandle(processInfo.hProcess);
    }
    if (nulHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(nulHandle);
    }
    return (int)exitCode;
#else
    return RunCommandExitCode("\"" + python + "\" -c \"" + code + "\" 2>/dev/null");
#endif
}

// 与ftllm里ctypes查cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_*)等价，
// 这里动态加载CUDA驱动，省掉对nvidia-smi的依赖
static bool HasSm89Device() {
#ifdef _WIN32
    HMODULE driver = LoadLibraryA("nvcuda.dll");
    if (driver == nullptr) {
        return false;
    }
    auto cuInit = (int (*)(unsigned int)) GetProcAddress(driver, "cuInit");
    auto cuDeviceGet = (int (*)(int *, int)) GetProcAddress(driver, "cuDeviceGet");
    auto cuDeviceGetAttribute = (int (*)(int *, int, int)) GetProcAddress(driver, "cuDeviceGetAttribute");
#else
    void *driver = dlopen("libcuda.so.1", RTLD_NOW);
    if (driver == nullptr) {
        return false;
    }
    auto cuInit = (int (*)(unsigned int)) dlsym(driver, "cuInit");
    auto cuDeviceGet = (int (*)(int *, int)) dlsym(driver, "cuDeviceGet");
    auto cuDeviceGetAttribute = (int (*)(int *, int, int)) dlsym(driver, "cuDeviceGetAttribute");
#endif
    if (cuInit == nullptr || cuDeviceGet == nullptr || cuDeviceGetAttribute == nullptr || cuInit(0) != 0) {
        return false;
    }
    const int capabilityMajor = 75, capabilityMinor = 76; // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_*
    for (int ordinal = 0; ordinal < 16; ordinal++) {
        int device = 0, major = 0, minor = 0;
        if (cuDeviceGet(&device, ordinal) != 0) {
            break;
        }
        if (cuDeviceGetAttribute(&major, capabilityMajor, device) != 0) {
            continue;
        }
        if (cuDeviceGetAttribute(&minor, capabilityMinor, device) != 0) {
            continue;
        }
        if (major == 8 && minor == 9) {
            return true;
        }
    }
    return false;
}

// 与ftllm的_configure_triton_compiler_python/_configure_sm89_fp8_linear_triton一致：
// 库是按FASTLLM_CUDA_TRITON*环境变量去拉起python triton服务的，这里把环境变量备好
static void ConfigureTriton() {
    if (getenv("FASTLLM_CUDA_TRITON") != nullptr ||
        getenv("FASTLLM_CUDA_TRITON_LINEAR_FP8") != nullptr) {
        printf("[fastllm] FASTLLM_CUDA_TRITON* 已由环境变量指定，跳过triton自动配置\n");
        fflush(stdout);
        return;
    }
    if (config.triton) {
        std::string python = FindTritonPython();
        if (python.empty()) {
            // 与ftllm一致：找不到解释器就禁用triton，用内置CUDA实现
            SetEnvVar("FASTLLM_CUDA_TRITON", "0");
            printf("[fastllm] --triton 已指定，但没找到能 import triton 的python解释器，"
                   "triton已禁用，改用内置CUDA实现(可用--triton_python <path>指定解释器)\n");
            fflush(stdout);
            return;
        }
        SetEnvVar("FASTLLM_CUDA_TRITON_PYTHON", python);
        SetEnvVar("FASTLLM_CUDA_TRITON", "1");
        printf("[fastllm] Triton enabled with python: %s\n", python.c_str());
        fflush(stdout);
    }
    // SM89上的fp8 linear triton在ftllm里是自动开启的，这里同样自动处理
    if (!HasSm89Device()) {
        return;
    }
    std::string python = FindTritonPython();
    if (python.empty()) {
        printf("[fastllm] SM89 FP8 Linear Triton 不可用(没有能 import triton 的python)，使用内置CUDA实现\n");
        fflush(stdout);
        return;
    }
    SetEnvVar("FASTLLM_CUDA_TRITON_PYTHON", python);
    SetEnvVar("FASTLLM_CUDA_TRITON_LINEAR_FP8", "1");
    printf("[fastllm] SM89 FP8 Linear Triton enabled with python: %s\n", python.c_str());
    fflush(stdout);
}

// 控制面请求不参与排队：batch很小时它们会排在被取消的那个请求后面，cancel就永远无法生效
static bool IsControlRoute(const HttpRequest &request) {
    return (request.route == "/v1/models" && request.method == "GET") ||
           (request.route == "/v1/active_conversations" && request.method == "GET") ||
           (request.route == "/v1/cancel" && request.method == "POST");
}

// dev_mode用：conversation_id到引擎handle的映射，用于查询与主动取消
static std::mutex gConversationLocker;
static std::map <std::string, int> gConversationHandles;

// 请求期间注册handle，析构时自动注销，异常路径也不会残留
class ConversationGuard {
public:
    ConversationGuard(const std::string &conversationId, int handleId) : conversationId(conversationId) {
        std::unique_lock <std::mutex> lock(gConversationLocker);
        gConversationHandles[conversationId] = handleId;
    }

    ~ConversationGuard() {
        std::unique_lock <std::mutex> lock(gConversationLocker);
        gConversationHandles.erase(conversationId);
    }

private:
    std::string conversationId;
};

struct WorkQueue {
    std::unique_ptr<fastllm::basellm> model;
    int maxActivateQueryNumber = 256;
    int activateQueryNumber = 0;
    int totalQueryNumber = 0;
    std::mutex locker;
    std::condition_variable cv;
    std::queue <WorkNode*> q;
    std::thread *loop;

    void Push(char *buffer, socket_t client) {
        locker.lock();
        q.push(new WorkNode());
        q.back()->Init(buffer, client);
        locker.unlock();

        cv.notify_all();
    }

    void Start() {
        loop = new std::thread ([] (WorkQueue *ts) {
            while (true) {
                std::unique_lock <std::mutex> lock(ts->locker);
                if (ts->activateQueryNumber >= ts->maxActivateQueryNumber) {
                    fastllm::MySleep(0);
                    continue;
                }
                if (ts->q.empty()) {
                    ts->cv.wait(lock);
                }

                while (ts->activateQueryNumber < ts->maxActivateQueryNumber && !ts->q.empty()) {
                    WorkNode *now = ts->q.front();
                    ts->q.pop();
                    ts->activateQueryNumber++;

                    ts->totalQueryNumber++;
//printf("activate = %d, q.size() = %d\n", ts->activateQueryNumber, (int) ts->q.size());

                    std::thread *t = new std::thread([](WorkQueue *ts, WorkNode *now) {
                        // 库里大量代码用throw std::runtime_error报错，异常逃出线程会terminate整个进程，
                        // 这里必须兜住，转成一次失败的请求
                        try {
                            ts->Deal(now);
                            printf("Response client %llu finish\n", (unsigned long long)now->client);
                            fflush(stdout);
                        } catch (const std::exception &e) {
                            printf("[fastllm-api] request failed: %s\n", e.what());
                            fflush(stdout);
                            SocketClose(now->client);
                        } catch (...) {
                            printf("[fastllm-api] request failed: unknown exception\n");
                            fflush(stdout);
                            SocketClose(now->client);
                        }
                        ts->locker.lock();
                        ts->activateQueryNumber--;
                        ts->locker.unlock();
                    }, ts, now);
                }
            }
        }, this);
    }

    void Deal(WorkNode *node) {
        auto *req = &node->request;
        if (req->method == "OPTIONS") {
            // 浏览器预检：与ftllm一致，不校验api key也不参与推理
            std::string message = "HTTP/1.1 200 OK\r\n";
            message += "Access-Control-Allow-Origin: *\r\n";
            message += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
            message += "Access-Control-Allow-Headers: *\r\n";
            message += "Access-Control-Max-Age: 86400\r\n";
            message += "Connection: close\r\n";
            message += "Content-Length: 0\r\n";
            message += "\r\n";
            SocketWrite(node->client, message.c_str(), message.length());
            SocketClose(node->client);
            return;
        }
        if (!::config.apiKey.empty() && !req->CheckApiKey(::config.apiKey)) {
            SendJsonReply(node->client, "HTTP/1.1 401 Unauthorized\r\n",
                          "{\"error\":{\"message\":\"Invalid API key.\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}");
            return;
        }
        // 每个请求进来都打一行，便于确认服务在实时工作而不是卡死
        printf("[fastllm-api] request: method=%s route=%s body=%d bytes\n",
               req->method.c_str(), req->route.c_str(), (int)req->body.size());
        if (!::config.hideInput) {
            // --hide_input 时不打印请求内容；这里只截前512字符，避免长prompt刷屏
            printf("[fastllm-api] request body: %.512s\n", req->body.c_str());
        }
        fflush(stdout);
        if (req->route == "/v1/active_conversations" && req->method == "GET") {
            bool devMode = ::config.devMode;
            std::string replyBody;
            if (!devMode) {
                // 与ftllm一致：只在dev_mode下开放
                replyBody = "{\"error\":\"This API is only available in development mode\"}";
            } else {
                json11::Json::array conversations;
                {
                    std::unique_lock <std::mutex> lock(gConversationLocker);
                    for (auto &it : gConversationHandles) {
                        conversations.push_back(json11::Json::object {
                            {"conversation_id", it.first},
                            {"handle", it.second}
                        });
                    }
                }
                json11::Json result = json11::Json::object {
                    {"active_conversations", conversations},
                    {"count", (int)conversations.size()}
                };
                replyBody = result.dump();
            }
            SendJsonReply(node->client, devMode ? "HTTP/1.1 200 OK\r\n" : "HTTP/1.1 403 Forbidden\r\n",
                          replyBody);
            return;
        } else if (req->route == "/v1/cancel" && req->method == "POST") {
            std::string conversationId = node->config["conversation_id"].is_string() ?
                                         node->config["conversation_id"].string_value() : "";
            int handleId = -1;
            std::string status;
            if (!::config.devMode) {
                status = "HTTP/1.1 403 Forbidden\r\n";
            } else if (conversationId.empty()) {
                status = "HTTP/1.1 400 Bad Request\r\n";
            } else {
                std::unique_lock <std::mutex> lock(gConversationLocker);
                auto it = gConversationHandles.find(conversationId);
                if (it != gConversationHandles.end()) {
                    handleId = it->second;
                    // 先摘掉再中断，避免正在退出的请求又把它注销一次
                    gConversationHandles.erase(it);
                }
                status = (handleId >= 0) ? "HTTP/1.1 200 OK\r\n" : "HTTP/1.1 404 Not Found\r\n";
            }
            std::string replyBody;
            if (!::config.devMode) {
                replyBody = "{\"error\":\"This API is only available in development mode\"}";
            } else if (conversationId.empty()) {
                replyBody = "{\"error\":\"Missing required parameter: conversation_id\"}";
            } else if (handleId < 0) {
                replyBody = "{\"error\":\"Failed to cancel conversation " + conversationId +
                            ". Conversation not found or already finished.\"}";
            } else {
                model->AbortResponse(handleId);
                printf("[fastllm-api] /v1/cancel: conversation=%s handle=%d aborted\n",
                       conversationId.c_str(), handleId);
                fflush(stdout);
                replyBody = "{\"message\":\"Conversation " + conversationId + " cancelled successfully\"}";
            }
            SendJsonReply(node->client, status.c_str(), replyBody);
            return;
        } else if (req->route == "/v1/messages" && req->method == "POST") {
            HandleAnthropicMessages(node);
            return;
        } else if ((req->route == "/v1/embed" || req->route == "/v1/embeddings") && req->method == "POST") {
            std::string message, replyBody;
            if (gEmbeddingModel == nullptr) {
                message = "HTTP/1.1 400 Bad Request\r\n";
                replyBody = "{\"error\":\"No embedding model loaded. Start the server with --embedding_model <path>.\"}";
            } else if (!node->config["inputs"].is_string()) {
                message = "HTTP/1.1 400 Bad Request\r\n";
                replyBody = "{\"error\":\"Missing required parameter: inputs\"}";
            } else {
                bool normalize = node->config["normalize"].is_bool() &&
                                 node->config["normalize"].bool_value();
                std::vector <float> embedding = gEmbeddingModel->EmbeddingSentence(
                        node->config["inputs"].string_value(), normalize);
                json11::Json::array values;
                for (float value : embedding) {
                    values.push_back(json11::Json((double)value));
                }
                message = "HTTP/1.1 200 OK\r\n";
                replyBody = json11::Json(values).dump();
                printf("[fastllm-api] /v1/embed: dims=%d normalize=%d\n",
                       (int)embedding.size(), (int)normalize);
                fflush(stdout);
            }
            AppendJsonReplyHeaders(message, replyBody.size());
            message += replyBody;
            SocketWrite(node->client, message.c_str(), message.length());
            SocketClose(node->client);
            return;
        } else if ((req->route == "/v1/rerank" || req->route == "/v1/reranks") && req->method == "POST") {
            std::string message, replyBody;
            json11::Json texts = node->config["texts"];
            if (gEmbeddingModel == nullptr) {
                message = "HTTP/1.1 400 Bad Request\r\n";
                replyBody = "{\"error\":\"No embedding model loaded. Start the server with --embedding_model <path>.\"}";
            } else if (!node->config["query"].is_string() || !texts.is_array()) {
                message = "HTTP/1.1 400 Bad Request\r\n";
                replyBody = "{\"error\":\"Missing required parameter: query or texts\"}";
            } else {
                std::string query = node->config["query"].string_value();
                int vocabSize = GetVocabSize(gEmbeddingModel.get());
                std::vector <std::vector <int> > tokenSequences;
                std::string tokenError;
                for (auto &text : texts.array_items()) {
                    // 与HF的pair编码一致: [CLS] query [SEP] text [SEP]
                    fastllm::Data ids = gEmbeddingModel->weight.tokenizer.Encode(
                            "[CLS]" + query + "[SEP]" + text.string_value() + "[SEP]");
                    std::vector <int> sequence;
                    for (int i = 0; i < ids.Count(0) && tokenError.empty(); i++) {
                        int id = (int)(((float *) ids.cpuData)[i] + 1e-9);
                        if (id < 0 || id >= vocabSize) {
                            tokenError = "rerank token out of range: index " + std::to_string(i);
                        }
                        sequence.push_back(id);
                    }
                    tokenSequences.push_back(sequence);
                }
                if (!tokenError.empty()) {
                    message = "HTTP/1.1 400 Bad Request\r\n";
                    replyBody = "{\"error\":\"" + tokenError + "\"}";
                } else {
                    std::vector <float> scores = gEmbeddingModel->ComputeScore(tokenSequences);
                    bool returnText = node->config["return_text"].is_bool() &&
                                      node->config["return_text"].bool_value();
                    std::vector <std::pair <float, int> > ranked;
                    for (int i = 0; i < (int)scores.size(); i++) {
                        ranked.push_back(std::make_pair(scores[i], i));
                    }
                    // 与ftllm一致：按分数从高到低
                    std::sort(ranked.begin(), ranked.end(),
                              [](const std::pair <float, int> &a, const std::pair <float, int> &b) {
                                  return a.first > b.first;
                              });
                    json11::Json::array results;
                    for (auto &item : ranked) {
                        json11::Json::object result {
                            {"index", item.second},
                            {"score", (double)item.first}
                        };
                        if (returnText) {
                            result["text"] = texts.array_items()[item.second].string_value();
                        }
                        results.push_back(json11::Json(result));
                    }
                    message = "HTTP/1.1 200 OK\r\n";
                    replyBody = json11::Json(results).dump();
                    printf("[fastllm-api] /v1/rerank: texts=%d\n", (int)texts.array_items().size());
                    fflush(stdout);
                }
            }
            AppendJsonReplyHeaders(message, replyBody.size());
            message += replyBody;
            SocketWrite(node->client, message.c_str(), message.length());
            SocketClose(node->client);
            return;
        } else if (req->route == "/v1/models" && req->method == "GET") {
            // 与ftllm的ModelList/ModelCard保持一致
            json11::Json modelList = json11::Json::object {
                {"object", "list"},
                {"data", json11::Json::array {
                    json11::Json::object {
                        {"id", ::config.modelName},
                        {"object", "model"},
                        {"created", _GetCurrentTime()},
                        {"owned_by", "fastchat"},
                        {"root", nullptr},
                        {"parent", nullptr},
                        {"permission", json11::Json::array {}}
                    }
                }}
            };
            std::string message = "HTTP/1.1 200 OK\r\n";
            std::string replyBody = modelList.dump();
            AppendJsonReplyHeaders(message, replyBody.size());
            message += replyBody;
            SocketWrite(node->client, message.c_str(), message.length());
            SocketClose(node->client);
            return;
        } else if ((req->route == "/generate" || req->route == "/generate/") && req->method == "POST") {
            std::string message = "";
            message += "HTTP/1.1 200 OK\r\n";
            message += "Content-Type:application/json\r\n";
            message += "server:fastllm api server\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n";
            message += "\r\n";

            if (node->error == "") {
                if (node->config["prompt"].is_null()) {
                    node->error = "prompt is empty!";
                }
            }
            if (node->error != "") {
                printf("error body = %s, prompt = %s, error = %s\n", node->request.body.c_str(), node->config["prompt"].string_value().c_str(), node->error.c_str());
                message += node->error;
                int ret = SocketWrite(node->client, message.c_str(), message.length()); //返回error
                SocketClose(node->client);
                return;
            }

            std::string output = "";
            bool rawPrompt = node->config["raw_prompt"].is_bool() && node->config["raw_prompt"].bool_value();
            // Data没有定义operator=，写成 Data x; x = Encode(...) 会走隐式逐成员赋值(浅拷贝指针)，
            // 临时对象析构后x就是悬垂指针，再析构一次即double free(0xc0000374)，必须直接初始化
            std::string promptText;
            if (rawPrompt) {
                promptText = node->config["prompt"].string_value();
            } else {
                fastllm::ChatMessages messages;
                messages.push_back({"user", node->config["prompt"].string_value()});
                promptText = model->ApplyChatTemplate(messages);
            }
            fastllm::Data inputs = model->weight.tokenizer.Encode(promptText);
            std::vector<int> tokens;
            std::string tokenError;
            if (!BuildTokens(model.get(), inputs, tokens, tokenError)) {
                printf("[fastllm-api] /generate bad request: %s\n", tokenError.c_str());
                fflush(stdout);
                SendJsonReply(node->client, "HTTP/1.1 400 Bad Request\r\n",
                              "{\"error\":{\"message\":\"" + tokenError + "\",\"type\":\"invalid_request_error\"}}");
                return;
            }
            fastllm::GenerationConfig config;
            config.output_token_limit = node->config["max_tokens"].is_null() ? 200 : node->config["max_tokens"].int_value();
            printf("[fastllm-api] prefill start: prompt_tokens=%d max_tokens=%d\n",
                   (int)tokens.size(), config.output_token_limit);
            fflush(stdout);
            PrefillHeartbeat heartbeat(5, INVALID_SOCKET, (int)tokens.size(), (int)tokens.size()); // /generate不是SSE，不发保活帧
            int handleId = model->LaunchResponseTokens(tokens, config);
            std::vector<float> results;
            int outputTokens = 0;
            bool firstToken = true;
            heartbeat.Start();
            while (true) {
                int result = model->FetchResponseTokens(handleId);
                if (firstToken) {
                    firstToken = false;
                    int prefillMs = heartbeat.Stop();
                    printf("[fastllm-api] prefill done: prompt_tokens=%d prefill_ms=%d (%.1f token/s)\n",
                           (int)tokens.size(), prefillMs, prefillMs > 0 ? tokens.size() * 1000.0 / prefillMs : 0.0);
                    fflush(stdout);
                }
                if (result < 0) {
                    if (result == -2) {
                        printf("[fastllm-api] /generate prompt too long, generation aborted\n");
                        fflush(stdout);
                    }
                    break;
                } else {
                    outputTokens++;
                    results.clear();
                    results.push_back(result);
                    output += model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));

                    std::string cur = (message + output);
                    int ret = SocketWrite(node->client, cur.c_str(), cur.length()); //返回message
                }
            }
            printf("[fastllm-api] /generate finished: output_tokens=%d\n", outputTokens);
            fflush(stdout);

            message += output;
            int ret = SocketWrite(node->client, message.c_str(), message.length()); //返回message

            SocketClose(node->client);
        } else if ((req->route == "/v1/chat/completions" || req->route == "/v1/chat/completions/") && req->method == "POST") {
            // Content-Length要等body定下来才能补，这里先只放状态行
            std::string message = "HTTP/1.1 200 OK\r\n";

            fastllm::ChatMessages chatMessages;
            if (node->config["messages"].is_array()) {
                for (auto &it : node->config["messages"].array_items()) {
                    chatMessages.push_back({it["role"].string_value(), it["content"].string_value()});
                }
            } else if (node->config["prompt"].is_string()) {
                chatMessages.push_back({"user", node->config["prompt"].string_value()});
            } else {
                node->error = "no input.\n";
            }

            if (node->config["model"].string_value() != ::config.modelName) {
                node->error = "The model `" + node->config["model"].string_value() + "` does not exist.";
            }

            if (node->error != "") {
                SendJsonReply(node->client, "HTTP/1.1 200 OK\r\n", node->error);
                return;
            }

            // tool_choice: none不解析工具调用；required/指定函数注入引导并强制校验
            json11::Json toolChoice = node->config["tool_choice"];
            bool toolsDisabled = toolChoice.is_string() && toolChoice.string_value() == "none";
            bool requiredToolChoice = false;
            std::string namedTool;
            if (toolChoice.is_string() && toolChoice.string_value() == "required") {
                requiredToolChoice = true;
            } else if (toolChoice.is_object() && toolChoice["type"].string_value() == "function") {
                requiredToolChoice = true;
                namedTool = toolChoice["function"]["name"].string_value();
            }
            json11::Json requestTools = node->config["tools"].is_array() ?
                                        node->config["tools"] : json11::Json::array {};
            ToolCallParserType toolCallParser = (requestTools.array_items().empty() || toolsDisabled) ?
                                                TOOL_CALL_PARSER_NONE : gToolCallParser;
            json11::Json templateMessages = node->config["messages"];
            if (requiredToolChoice) {
                bool exactlyOne = node->config["parallel_tool_calls"].is_bool() &&
                                  !node->config["parallel_tool_calls"].bool_value();
                std::string guidanceError;
                if (!ApplyToolChoiceGuidance(node->config["messages"], requestTools, true, namedTool,
                                             exactlyOne, templateMessages, requestTools, guidanceError)) {
                    printf("[fastllm-api] /v1/chat/completions bad request: %s\n", guidanceError.c_str());
                    fflush(stdout);
                    SendJsonReply(node->client, "HTTP/1.1 400 Bad Request\r\n",
                                  "{\"error\":{\"message\":\"" + guidanceError + "\",\"type\":\"invalid_request_error\"}}");
                    return;
                }
            }

            bool rawPrompt = node->config["raw_prompt"].is_bool() && node->config["raw_prompt"].bool_value();
            std::string promptText;
            if (rawPrompt) {
                if (!node->config["prompt"].is_string()) {
                    node->error = "raw_prompt requires a string prompt.\n";
                } else {
                    promptText = node->config["prompt"].string_value();
                }
            } else if (NeedChatTemplateVars(templateMessages, requestTools)) {
                // 带tools或tool_calls时模板需要这些变量，pair版本的ChatMessages表达不了
                promptText = model->ApplyChatTemplate(
                        BuildChatTemplateVars(templateMessages, requestTools));
            } else {
                promptText = model->ApplyChatTemplate(chatMessages);
            }
            if (node->error != "") {
                SendJsonReply(node->client, "HTTP/1.1 200 OK\r\n", node->error);
                return;
            }
            // Data没有定义operator=，必须直接初始化，否则是浅拷贝+double free
            fastllm::Data inputs = model->weight.tokenizer.Encode(promptText);
            std::vector<int> tokens;
            std::string tokenError;
            if (!BuildTokens(model.get(), inputs, tokens, tokenError)) {
                printf("[fastllm-api] /v1/chat/completions bad request: %s\n", tokenError.c_str());
                fflush(stdout);
                SendJsonReply(node->client, "HTTP/1.1 400 Bad Request\r\n",
                              "{\"error\":{\"message\":\"" + tokenError + "\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            // n只能为1：多候选需要多次独立生成，直接报错好过静默只回一条
            if (node->config["n"].is_number() && node->config["n"].int_value() > 1) {
                SendJsonReply(node->client, "HTTP/1.1 400 Bad Request\r\n",
                              "{\"error\":{\"message\":\"n > 1 is not supported.\",\"type\":\"invalid_request_error\"}}");
                return;
            }

            fastllm::GenerationConfig config;
            // 与ftllm一致：max_tokens未指定时用32768，max_completion_tokens是其别名
            int maxTokens = node->config["max_tokens"].is_number() ? node->config["max_tokens"].int_value() :
                            (node->config["max_completion_tokens"].is_number() ?
                             node->config["max_completion_tokens"].int_value() : 0);
            config.output_token_limit = maxTokens > 0 ? maxTokens : 32768;
            if (node->config["min_tokens"].is_number()) {
                config.output_token_least = node->config["min_tokens"].int_value();
            }
            // 与ftllm的default_generation_config一致：CLI默认值打底，请求里给了用请求的
            float temperature = node->config["temperature"].is_number() ?
                                (float)node->config["temperature"].number_value() : ::config.temperature;
            float topP = node->config["top_p"].is_number() ?
                         (float)node->config["top_p"].number_value() : ::config.topP;
            int topK = node->config["top_k"].is_number() ? node->config["top_k"].int_value() : ::config.topK;
            // 对应ftllm的_normalize_sampling_args：temperature<=0表示贪心解码
            config.do_sample = true;
            if (temperature <= 0.0f) {
                config.do_sample = false;
                temperature = 1.0f;
                topK = 1;
                topP = 1.0f;
            }
            config.temperature = temperature;
            config.top_p = topP;
            config.top_k = topK;
            config.repeat_penalty = ::config.repeatPenalty;
            if (node->config["frequency_penalty"].is_number() && node->config["frequency_penalty"].number_value() != 0.0) {
                config.repeat_penalty = (float)node->config["frequency_penalty"].number_value();
            }
            if (node->config["repetition_penalty"].is_number()) {
                config.repeat_penalty = (float)node->config["repetition_penalty"].number_value();
            }
            // stop串：单token的直接交给引擎，多token的只能生成后截断
            std::vector <std::string> stopStrings;
            if (node->config["stop"].is_string()) {
                stopStrings.push_back(node->config["stop"].string_value());
            } else if (node->config["stop"].is_array()) {
                for (auto &it : node->config["stop"].array_items()) {
                    if (it.is_string()) {
                        stopStrings.push_back(it.string_value());
                    }
                }
            }
            for (auto &stop : stopStrings) {
                if (stop.empty()) {
                    continue;
                }
                fastllm::Data stopInputs = model->weight.tokenizer.Encode(stop);
                if (stopInputs.Count(0) == 1) {
                    config.stop_token_ids.insert((int)(((float *) stopInputs.cpuData)[0] + 1e-9));
                }
            }

            std::string output = "";
            bool isStream = node->config["stream"].is_bool() && node->config["stream"].bool_value();
            printf("[fastllm-api] prefill start: prompt_tokens=%d stream=%d max_tokens=%d messages=%d\n",
                   (int)tokens.size(), (int)isStream, config.output_token_limit, (int)chatMessages.size());
            fflush(stdout);
            int handleId = model->LaunchResponseTokens(tokens, config);
            // 带历史命中时真正要算的只有missed部分，用它推算进度和速度才有意义
            int cachedTokens = 0, missedTokens = (int)tokens.size(), statOutputTokens = 0;
            model->GetResponseStatistics(handleId, cachedTokens, missedTokens, statOutputTokens);
            // 流式请求把客户端fd给心跳线程，prefill期间发SSE保活帧，避免客户端超时重连
            PrefillHeartbeat heartbeat(5, isStream ? node->client : INVALID_SOCKET,
                                       (int)tokens.size(), missedTokens);
            heartbeat.Start();
            bool firstToken = true;

            std::string curId = "fastllm-" + GenerateRandomID();
            auto createTime = _GetCurrentTime();
            ConversationGuard conversationGuard(curId, handleId); // dev_mode的cancel/查询靠它

            if (isStream) {
                message = "";
                message += "HTTP/1.1 200 OK\r\n";
                message += "Content-Type:text/event-stream\r\n";
                message += "Cache-Control:no-cache\r\nX-Accel-Buffering: no\r\n";
                message += "server:fastllm api server\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n";
                message += "Transfer-Encoding: chunked\r\n";
                message += "\r\n";
                SocketWrite(node->client, message.c_str(), message.length()); //返回初始信息
                heartbeat.EnableKeepAlive(); // 头已发出，prefill期间的保活帧可以开始了
            
                json11::Json startResult = json11::Json::object {
                    {"id", curId},
                    {"object", "chat.completion.chunk"},
                    {"created", createTime},
                    {"model", ::config.modelName},
                    {"choices", json11::Json::array {
                        json11::Json::object {
                            {"index", 0},
                            {"delta", json11::Json::object {
                                {"role", "assistant"}
                            }},
                            {"logprobs", nullptr},
                            {"finish_reason", nullptr},
                            {"stop_reason", nullptr}
                        }
                    }}
                };
                std::string cur = ("data: " + startResult.dump() + "\n\n");
                heartbeat.WriteFrame(cur);

                int outputTokens = 0;
                std::vector<float> results;
                std::string streamText, sentText; // 用于stop串截断：已生成/已发送的文本
                bool hitStop = false;
                int finishCode = -1; // 引擎给出的结束码，-2表示prompt过长
                while (true) {
                    int result = model->FetchResponseTokens(handleId);
                    if (firstToken) {
                        firstToken = false;
                        int prefillMs = heartbeat.StopPrefill();
                        printf("[fastllm-api] prefill done: prompt_tokens=%d cached=%d needs=%d prefill_ms=%d (%.1f token/s)\n",
                               (int)tokens.size(), cachedTokens, missedTokens, prefillMs,
                               prefillMs > 0 ? missedTokens * 1000.0 / prefillMs : 0.0);
                        fflush(stdout);
                    }
                    if (result < 0) {
                        finishCode = result;
                        if (result == -2) {
                            printf("[fastllm-api] /v1/chat/completions prompt too long, generation aborted\n");
                            fflush(stdout);
                        }
                        break;
                    }
                    outputTokens++;
                    results.clear();
                    results.push_back(result);
                    std::string now = model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));
                    streamText += now;
                    // 多token的stop串只能在这里截断：截断后只补发没发出去的部分
                    std::string truncated = streamText;
                    if (TruncateAtStop(truncated, stopStrings)) {
                        hitStop = true;
                    }
                    std::string deltaText = truncated.substr(sentText.size());
                    sentText = truncated;
                    if (!deltaText.empty() && toolCallParser == TOOL_CALL_PARSER_NONE) {
                        // 工具调用需要整块文本才能解析，这里先缓冲，循环结束后一次性发出
                        json11::Json partResult = json11::Json::object {
                            {"id", curId},
                            {"object", "chat.completion.chunk"},
                            {"created", createTime},
                            {"model", ::config.modelName},
                            {"choices", json11::Json::array {
                                json11::Json::object {
                                    {"index", 0},
                                    {"delta", json11::Json::object {
                                        {"content", deltaText}
                                    }},
                                    {"logprobs", nullptr},
                                    {"finish_reason", nullptr},
                                    {"stop_reason", nullptr}
                                }
                            }}
                        };

                        std::string cur = ("data: " + partResult.dump() + "\n\n");
                        heartbeat.WriteFrame(cur);
                    }
                    if (hitStop) {
                        break;
                    }
                }

                std::string streamFinishOverride;
                if (toolCallParser != TOOL_CALL_PARSER_NONE) {
                    // 工具调用整块发一次：content只保留工具调用之外的文本
                    std::vector <ToolCallInfo> toolCalls;
                    std::string parsedContent;
                    ParseToolCalls(toolCallParser, streamText, requestTools, parsedContent, toolCalls);
                    std::string contentDelta = parsedContent;
                    json11::Json::array toolCallsJson;
                    for (auto &call : toolCalls) {
                        toolCallsJson.push_back(json11::Json::object {
                            {"index", (int)toolCallsJson.size()},
                            {"id", GenerateToolCallId()},
                            {"type", "function"},
                            {"function", json11::Json::object {
                                {"name", call.name},
                                {"arguments", call.arguments}
                            }}
                        });
                    }
                    if (!contentDelta.empty() || !toolCallsJson.empty()) {
                        json11::Json::object delta;
                        if (!contentDelta.empty()) {
                            delta["content"] = contentDelta;
                        }
                        if (!toolCallsJson.empty()) {
                            delta["tool_calls"] = toolCallsJson;
                        }
                        json11::Json deltaJson(delta);
                        json11::Json partResult = json11::Json::object {
                            {"id", curId},
                            {"object", "chat.completion.chunk"},
                            {"created", createTime},
                            {"model", ::config.modelName},
                            {"choices", json11::Json::array {
                                json11::Json::object {
                                    {"index", 0},
                                    {"delta", deltaJson},
                                    {"logprobs", nullptr},
                                    {"finish_reason", nullptr},
                                    {"stop_reason", nullptr}
                                }
                            }}
                        };
                        std::string cur = ("data: " + partResult.dump() + "\n\n");
                        heartbeat.WriteFrame(cur);
                    }
                    if (!toolCallsJson.empty()) {
                        streamFinishOverride = "tool_calls";
                    } else if (requiredToolChoice) {
                        // 流式响应头已经发出去了，没法再回400，只能记日志
                        printf("[fastllm-api] tool_choice requires a tool call but none was produced (stream)\n");
                        fflush(stdout);
                    }
                }

                {
                    // OpenAI协议要求最后一个chunk带非null finish_reason，
                    // 否则严格校验的客户端会认为流异常结束并重连重试
                    const char *finishReason = !streamFinishOverride.empty() ? streamFinishOverride.c_str() :
                            (hitStop ? "stop" :
                            ((finishCode == -2 || outputTokens >= config.output_token_limit) ? "length" : "stop"));
                    json11::Json partResult = json11::Json::object {
                        {"id", curId},
                        {"object", "chat.completion.chunk"},
                        {"created", createTime},
                        {"model", ::config.modelName},
                        {"choices", json11::Json::array {
                            json11::Json::object {
                                {"index", 0},
                                {"delta", json11::Json::object {
                                    {"content", ""}
                                }},
                                {"logprobs", nullptr},
                                {"finish_reason", finishReason},
                                {"stop_reason", nullptr}
                            }
                        }},
                        {"usage", json11::Json::object {
                            {"prompt_tokens", (int)tokens.size()},
                            {"total_tokens", (int)tokens.size() + outputTokens},
                            {"completion_tokens", outputTokens},
                            {"prompt_tokens_details", json11::Json::object {
                                {"cached_tokens", cachedTokens},
                                {"missed_tokens", missedTokens}
                            }}
                        }}
                    };

                    std::string cur = ("data: " + partResult.dump() + "\n\n");
                    heartbeat.WriteFrame(cur);
                }

                cur = ("data: [DONE]\n\n");
                heartbeat.WriteFrame(cur);

                // 先停保活线程再发结束帧，避免注释帧插到分块结束标记之后
                heartbeat.Stop();
                SocketWrite(node->client, "0\r\n\r\n", 5);
                printf("[fastllm-api] /v1/chat/completions stream finished: output_tokens=%d\n", outputTokens);
                fflush(stdout);
                SocketShutdownWrite(node->client);
                SocketClose(node->client);
            } else {
                int outputTokens = 0;
                std::vector<float> results;
                int finishCode = -1; // 引擎给出的结束码，-2表示prompt过长
                while (true) {
                    int result = model->FetchResponseTokens(handleId);
                    if (firstToken) {
                        firstToken = false;
                        int prefillMs = heartbeat.Stop();
                        printf("[fastllm-api] prefill done: prompt_tokens=%d cached=%d needs=%d prefill_ms=%d (%.1f token/s)\n",
                               (int)tokens.size(), cachedTokens, missedTokens, prefillMs,
                               prefillMs > 0 ? missedTokens * 1000.0 / prefillMs : 0.0);
                        fflush(stdout);
                    }
                    if (result < 0) {
                        finishCode = result;
                        if (result == -2) {
                            printf("[fastllm-api] /v1/chat/completions prompt too long, generation aborted\n");
                            fflush(stdout);
                        }
                        break;
                    } else {
                        results.clear();
                        results.push_back(result);
                        output += model->weight.tokenizer.Decode(fastllm::Data (fastllm::DataType::FLOAT32, {(int)results.size()}, results));
                        outputTokens++;
                    }
                }
                // 多token的stop串在这里截断
                bool hitStop = TruncateAtStop(output, stopStrings);
                const char *finishReason = hitStop ? "stop" :
                        ((finishCode == -2 || outputTokens >= config.output_token_limit) ? "length" : "stop");

                std::vector <ToolCallInfo> toolCalls;
                if (toolCallParser != TOOL_CALL_PARSER_NONE) {
                    // 与ftllm一致：没找到标记就是原文；解析失败只保留标记前的文本
                    std::string parsedContent;
                    ParseToolCalls(toolCallParser, output, requestTools, parsedContent, toolCalls);
                    output = parsedContent;
                    if (!toolCalls.empty()) {
                        // 工具调用时content留空(OpenAI里通常是null)
                        finishReason = "tool_calls";
                    }
                }
                if (requiredToolChoice && toolCalls.empty()) {
                    // 与ftllm一致：强制调用却没产出合法调用，按请求错误返回
                    std::string errorMessage = namedTool.empty() ?
                            "tool_choice='required' was set but no valid tool call was produced" :
                            "tool_choice requires function '" + namedTool +
                            "' but no valid tool call was produced";
                    printf("[fastllm-api] /v1/chat/completions bad request: %s\n", errorMessage.c_str());
                    fflush(stdout);
                    SendJsonReply(node->client, "HTTP/1.1 400 Bad Request\r\n",
                                  "{\"error\":{\"message\":\"" + errorMessage + "\",\"type\":\"invalid_request_error\"}}");
                    return;
                }

                json11::Json::object messageObject = {
                    {"role", "assistant"},
                    {"content", output}
                };
                if (!toolCalls.empty()) {
                    messageObject["content"] = output.empty() ? json11::Json(nullptr) : json11::Json(output);
                    json11::Json::array toolCallsJson;
                    for (auto &call : toolCalls) {
                        toolCallsJson.push_back(json11::Json::object {
                            {"id", GenerateToolCallId()},
                            {"type", "function"},
                            {"function", json11::Json::object {
                                {"name", call.name},
                                {"arguments", call.arguments}
                            }}
                        });
                    }
                    messageObject["tool_calls"] = toolCallsJson;
                }
                json11::Json messageJson(messageObject);

                json11::Json result = json11::Json::object {
                    {"id", curId},
                    {"object", "chat.completion"},
                    {"created", createTime},
                    {"model", ::config.modelName},
                    {"choices", json11::Json::array {
                        json11::Json::object {
                            {"index", 0},
                            {"message", messageJson},
                            {"logprobs", nullptr},
                            {"finish_reason", finishReason},
                            {"stop_reason", nullptr}
                        }
                    }},
                    {"usage", json11::Json::object {
                        {"prompt_tokens", (int)tokens.size()},
                        {"total_tokens", (int)tokens.size() + outputTokens},
                        {"completion_tokens", outputTokens},
                        {"prompt_tokens_details", json11::Json::object {
                            {"cached_tokens", cachedTokens},
                            {"missed_tokens", missedTokens}
                        }}
                    }}
                };

                std::string replyBody = result.dump();
                AppendJsonReplyHeaders(message, replyBody.size());
                message += replyBody;
                SocketWrite(node->client, message.c_str(), message.length()); //返回message
                SocketClose(node->client);
                printf("[fastllm-api] /v1/chat/completions finished: output_tokens=%d\n", outputTokens);
                fflush(stdout);
            }
            return;
        } else {
            printf("[fastllm-api] unsupported request: method=%s route=%s\n",
                   req->method.c_str(), req->route.c_str());
            fflush(stdout);
            SendJsonReply(node->client, "HTTP/1.1 404 Not Found\r\n",
                          "{\"error\":{\"message\":\"Unsupported route, only /generate, /v1/chat/completions and /v1/models are available.\",\"type\":\"invalid_request_error\",\"code\":\"not_found\"}}");
            return;
        }
    }

    // Anthropic Messages协议。为了不给chat路径引入分支，这里先生成完整结果，
    // 需要流式时再按Anthropic的事件序列发出去(文本按块切分)
    void HandleAnthropicMessages(WorkNode *node) {
        json11::Json requestMessages = node->config["messages"];
        if (node->error != "") {
            SendJsonReply(node->client, "HTTP/1.1 400 Bad Request\r\n",
                          "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"" + node->error + "\"}}");
            return;
        }
        if (!requestMessages.is_array() || requestMessages.array_items().empty()) {
            SendJsonReply(node->client, "HTTP/1.1 400 Bad Request\r\n",
                          "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"messages is required\"}}");
            return;
        }
        if (!node->config["max_tokens"].is_number()) {
            SendJsonReply(node->client, "HTTP/1.1 400 Bad Request\r\n",
                          "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"max_tokens is required\"}}");
            return;
        }

        json11::Json::array chatMessages;
        json11::Json system = node->config["system"];
        std::string systemText;
        if (system.is_string()) {
            systemText = system.string_value();
        } else if (system.is_array()) {
            for (auto &block : system.array_items()) {
                if (block["type"].string_value() == "text") {
                    systemText += block["text"].string_value();
                }
            }
        }
        if (!systemText.empty()) {
            chatMessages.push_back(json11::Json::object {
                {"role", "system"}, {"content", systemText}
            });
        }
        for (auto &message : requestMessages.array_items()) {
            std::string role = message["role"].string_value();
            json11::Json content = message["content"];
            if (content.is_string()) {
                chatMessages.push_back(json11::Json::object {
                    {"role", role}, {"content", content.string_value()}
                });
                continue;
            }
            if (!content.is_array()) {
                continue;
            }
            std::string text;
            json11::Json::array toolCalls, toolResults;
            for (auto &block : content.array_items()) {
                std::string type = block["type"].string_value();
                if (type == "text") {
                    text += block["text"].string_value();
                } else if (type == "tool_use") {
                    // Anthropic的tool_use对应OpenAI的assistant.tool_calls
                    toolCalls.push_back(json11::Json::object {
                        {"id", block["id"].string_value()},
                        {"type", "function"},
                        {"function", json11::Json::object {
                            {"name", block["name"].string_value()},
                            {"arguments", block["input"].dump()}
                        }}
                    });
                } else if (type == "tool_result") {
                    std::string resultText;
                    json11::Json resultContent = block["content"];
                    if (resultContent.is_string()) {
                        resultText = resultContent.string_value();
                    } else if (resultContent.is_array()) {
                        for (auto &part : resultContent.array_items()) {
                            if (part["type"].string_value() == "text") {
                                resultText += part["text"].string_value();
                            }
                        }
                    }
                    toolResults.push_back(json11::Json::object {
                        {"role", "tool"},
                        {"content", resultText},
                        {"tool_call_id", block["tool_use_id"].string_value()}
                    });
                }
            }
            for (auto &result : toolResults) {
                chatMessages.push_back(result);
            }
            if (!text.empty() || !toolCalls.empty()) {
                json11::Json::object assistantMessage {
                    {"role", "assistant"},
                    {"content", text}
                };
                if (!toolCalls.empty()) {
                    assistantMessage["tool_calls"] = toolCalls;
                }
                chatMessages.push_back(json11::Json(assistantMessage));
            }
        }

        // Anthropic的tools是{name, description, input_schema}，模板要OpenAI那份形状
        json11::Json::array chatTools;
        if (node->config["tools"].is_array()) {
            for (auto &tool : node->config["tools"].array_items()) {
                chatTools.push_back(json11::Json::object {
                    {"type", "function"},
                    {"function", json11::Json::object {
                        {"name", tool["name"].string_value()},
                        {"description", tool["description"].is_string() ? tool["description"] : json11::Json("")},
                        {"parameters", tool["input_schema"].is_object() ? tool["input_schema"] : json11::Json::object {}}
                    }}
                });
            }
        }
        json11::Json messagesJson(chatMessages), toolsJson(chatTools);
        std::string promptText = model->ApplyChatTemplate(BuildChatTemplateVars(messagesJson, toolsJson));
        fastllm::Data inputs = model->weight.tokenizer.Encode(promptText);
        std::vector <int> tokens;
        std::string tokenError;
        if (!BuildTokens(model.get(), inputs, tokens, tokenError)) {
            SendJsonReply(node->client, "HTTP/1.1 400 Bad Request\r\n",
                          "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"" + tokenError + "\"}}");
            return;
        }

        fastllm::GenerationConfig generation;
        generation.output_token_limit = node->config["max_tokens"].int_value();
        float temperature = node->config["temperature"].is_number() ?
                            (float)node->config["temperature"].number_value() : ::config.temperature;
        float topP = node->config["top_p"].is_number() ?
                     (float)node->config["top_p"].number_value() : ::config.topP;
        int topK = node->config["top_k"].is_number() ? node->config["top_k"].int_value() : ::config.topK;
        generation.do_sample = true;
        if (temperature <= 0.0f) {
            generation.do_sample = false;
            temperature = 1.0f;
            topK = 1;
            topP = 1.0f;
        }
        generation.temperature = temperature;
        generation.top_p = topP;
        generation.top_k = topK;
        generation.repeat_penalty = ::config.repeatPenalty;
        std::vector <std::string> stopStrings;
        if (node->config["stop_sequences"].is_array()) {
            for (auto &stop : node->config["stop_sequences"].array_items()) {
                if (stop.is_string() && !stop.string_value().empty()) {
                    stopStrings.push_back(stop.string_value());
                }
            }
        }
        for (auto &stop : stopStrings) {
            fastllm::Data stopInputs = model->weight.tokenizer.Encode(stop);
            if (stopInputs.Count(0) == 1) {
                generation.stop_token_ids.insert((int)(((float *) stopInputs.cpuData)[0] + 1e-9));
            }
        }

        bool isStream = node->config["stream"].is_bool() && node->config["stream"].bool_value();
        printf("[fastllm-api] /v1/messages prefill start: prompt_tokens=%d stream=%d max_tokens=%d tools=%d\n",
               (int)tokens.size(), (int)isStream, generation.output_token_limit,
               (int)chatTools.size());
        fflush(stdout);
        int handleId = model->LaunchResponseTokens(tokens, generation);
        int cachedTokens = 0, missedTokens = (int)tokens.size(), statOutputTokens = 0;
        model->GetResponseStatistics(handleId, cachedTokens, missedTokens, statOutputTokens);
        PrefillHeartbeat heartbeat(5, isStream ? node->client : INVALID_SOCKET,
                                   (int)tokens.size(), missedTokens);
        heartbeat.Start();

        bool firstToken = true;
        std::string output;
        int outputTokens = 0, finishCode = -1;
        std::vector <float> results;
        while (true) {
            int result = model->FetchResponseTokens(handleId);
            if (firstToken) {
                firstToken = false;
                int prefillMs = heartbeat.Stop();
                printf("[fastllm-api] /v1/messages prefill done: prompt_tokens=%d cached=%d needs=%d prefill_ms=%d (%.1f token/s)\n",
                       (int)tokens.size(), cachedTokens, missedTokens, prefillMs,
                       prefillMs > 0 ? missedTokens * 1000.0 / prefillMs : 0.0);
                fflush(stdout);
            }
            if (result < 0) {
                finishCode = result;
                break;
            }
            results.clear();
            results.push_back(result);
            output += model->weight.tokenizer.Decode(
                    fastllm::Data(fastllm::DataType::FLOAT32, {(int)results.size()}, results));
            outputTokens++;
        }
        bool hitStop = TruncateAtStop(output, stopStrings);
        std::vector <ToolCallInfo> toolCalls;
        ToolCallParserType parser = chatTools.empty() ? TOOL_CALL_PARSER_NONE : gToolCallParser;
        if (parser != TOOL_CALL_PARSER_NONE) {
            std::string parsedContent;
            ParseToolCalls(parser, output, toolsJson, parsedContent, toolCalls);
            output = parsedContent;
        }
        const char *stopReason = !toolCalls.empty() ? "tool_use" :
                (hitStop ? "stop_sequence" :
                 ((finishCode == -2 || outputTokens >= generation.output_token_limit) ? "max_tokens" : "end_turn"));
        std::string messageId = "msg_" + GenerateToolCallId().substr(5);
        printf("[fastllm-api] /v1/messages finished: output_tokens=%d stop_reason=%s\n", outputTokens, stopReason);
        fflush(stdout);

        json11::Json usage = json11::Json::object {
            {"input_tokens", (int)tokens.size()},
            {"output_tokens", outputTokens},
            {"cache_read_input_tokens", cachedTokens},
            {"cache_creation_input_tokens", 0}
        };
        if (!isStream) {
            json11::Json::array contentBlocks;
            if (!output.empty()) {
                contentBlocks.push_back(json11::Json::object {
                    {"type", "text"},
                    {"text", output}
                });
            }
            for (auto &call : toolCalls) {
                std::string parseError;
                json11::Json input = json11::Json::parse(call.arguments, parseError);
                contentBlocks.push_back(json11::Json::object {
                    {"type", "tool_use"},
                    {"id", GenerateToolCallId()},
                    {"name", call.name},
                    {"input", parseError.empty() ? input : json11::Json::object {}}
                });
            }
            json11::Json result = json11::Json::object {
                {"id", messageId},
                {"type", "message"},
                {"role", "assistant"},
                {"model", ::config.modelName},
                {"content", contentBlocks},
                {"stop_reason", stopReason},
                {"stop_sequence", nullptr},
                {"usage", usage}
            };
            SendJsonReply(node->client, "HTTP/1.1 200 OK\r\n", result.dump());
            return;
        }

        // 流式：按Anthropic的事件序列发
        std::string headers = "HTTP/1.1 200 OK\r\n";
        headers += "Content-Type:text/event-stream\r\n";
        headers += "Cache-Control:no-cache\r\nX-Accel-Buffering: no\r\n";
        headers += "server:fastllm api server\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n";
        headers += "Transfer-Encoding: chunked\r\n";
        headers += "\r\n";
        SocketWrite(node->client, headers.c_str(), headers.length());
        heartbeat.EnableKeepAlive();
        auto sendEvent = [this, node] (const std::string &event, const json11::Json &data) {
            WriteChunkedFrame(node->client, "event: " + event + "\ndata: " + data.dump() + "\n\n");
        };
        sendEvent("message_start", json11::Json::object {
            {"type", "message_start"},
            {"message", json11::Json::object {
                {"id", messageId},
                {"type", "message"},
                {"role", "assistant"},
                {"model", ::config.modelName},
                {"content", json11::Json::array {}},
                {"stop_reason", nullptr},
                {"stop_sequence", nullptr},
                {"usage", json11::Json::object {
                    {"input_tokens", (int)tokens.size()},
                    {"output_tokens", 0},
                    {"cache_read_input_tokens", cachedTokens},
                    {"cache_creation_input_tokens", 0}
                }}
            }}
        });
        int blockIndex = 0;
        if (!output.empty()) {
            sendEvent("content_block_start", json11::Json::object {
                {"type", "content_block_start"},
                {"index", blockIndex},
                {"content_block", json11::Json::object {
                    {"type", "text"},
                    {"text", ""}
                }}
            });
            // 文本按块发，客户端侧观感与逐token相近
            const size_t chunkSize = 32;
            for (size_t begin = 0; begin < output.size(); begin += chunkSize) {
                sendEvent("content_block_delta", json11::Json::object {
                    {"type", "content_block_delta"},
                    {"index", blockIndex},
                    {"delta", json11::Json::object {
                        {"type", "text_delta"},
                        {"text", output.substr(begin, chunkSize)}
                    }}
                });
            }
            sendEvent("content_block_stop", json11::Json::object {
                {"type", "content_block_stop"},
                {"index", blockIndex}
            });
            blockIndex++;
        }
        for (auto &call : toolCalls) {
            std::string toolUseId = GenerateToolCallId();
            sendEvent("content_block_start", json11::Json::object {
                {"type", "content_block_start"},
                {"index", blockIndex},
                {"content_block", json11::Json::object {
                    {"type", "tool_use"},
                    {"id", toolUseId},
                    {"name", call.name},
                    {"input", json11::Json::object {}}
                }}
            });
            sendEvent("content_block_delta", json11::Json::object {
                {"type", "content_block_delta"},
                {"index", blockIndex},
                {"delta", json11::Json::object {
                    {"type", "input_json_delta"},
                    {"partial_json", call.arguments}
                }}
            });
            sendEvent("content_block_stop", json11::Json::object {
                {"type", "content_block_stop"},
                {"index", blockIndex}
            });
            blockIndex++;
        }
        sendEvent("message_delta", json11::Json::object {
            {"type", "message_delta"},
            {"delta", json11::Json::object {
                {"stop_reason", stopReason},
                {"stop_sequence", nullptr}
            }},
            {"usage", json11::Json::object {
                {"output_tokens", outputTokens}
            }}
        });
        sendEvent("message_stop", json11::Json::object {
            {"type", "message_stop"}
        });
        SocketShutdownWrite(node->client);
        SocketClose(node->client);
    }
} workQueue;

void Usage() {
    std::cout << "Usage: apiserver <model> [options]" << std::endl;
    std::cout << "参数与 ftllm serve 对齐" << std::endl;
    std::cout << std::endl;
    std::cout << "加载与设备:" << std::endl;
    std::cout << "  <model>                      模型路径，可放在位置参数或 -p/--path" << std::endl;
    std::cout << "  -p, --path <args>:           模型路径，fastllm模型文件或HF模型文件夹" << std::endl;
    std::cout << "  -t, --threads <args>:        使用的线程数量" << std::endl;
    std::cout << "  -l, --low:                   使用低内存模式" << std::endl;
    std::cout << "  --cuda_embedding:            在cuda上进行embedding" << std::endl;
    std::cout << "  --dtype <args>:              权重类型(读取HF模型时有效，默认float16)" << std::endl;
    std::cout << "  --moe_dtype <args>:          MoE层权重类型(读取HF模型时有效)" << std::endl;
    std::cout << "  --atype <args>:              推理使用的数据类型(float32/float16)" << std::endl;
    std::cout << "  --moe_atype <args>:          MoE层激活类型(float32/float16)" << std::endl;
    std::cout << "  --kv_cache_dtype <args>:     KV Cache类型(float16/float32)" << std::endl;
    std::cout << "  --device <args>:             主计算设备，如 cuda、cuda:0、cpu、numa 或 \"{'cuda':1,'numa':8}\"" << std::endl;
    std::cout << "  --moe_device <args>:         MoE专家层设备，如 cpu、numa，需与--device一起使用" << std::endl;
    std::cout << "  --moe_device_layers <args>:  仅最后N层MoE使用--moe_device，-1表示全部" << std::endl;
    std::cout << "  --ngram_device <args>:       ngram表存放位置(cpu/disk)" << std::endl;
    std::cout << "  --moe_cuda_cache <args>:     缓存MoE专家的CUDA显存，如3g；0表示关闭" << std::endl;
    std::cout << "  --moe_cpu_cache <args>:      MoE专家内存缓存总上限，如32g；0表示关闭" << std::endl;
    std::cout << "  --moe_experts <args>:        MoE使用的专家数" << std::endl;
    std::cout << "  --lora <args>:               指定lora路径" << std::endl;
    std::cout << "  --dtype_config <args>:       指定权重类型配置文件" << std::endl;
    std::cout << "  --cuda_shared_expert, --cuda_se <true/false>: 是否用cuda执行共享专家" << std::endl;
    std::cout << "  --enable_amx, --amx <true/false>: 是否开启amx加速" << std::endl;
    std::cout << "  --tp <args>:                 线程张量并行配置(FASTLLM_TP)" << std::endl;
    std::cout << "  --mtp <args>:                MTP草稿token数" << std::endl;
    std::cout << "  --mtp_fp8_draft_head <args>: MTP草稿头是否用fp8" << std::endl;
    std::cout << "  --dspark <args>:             DSpark草稿模型目录" << std::endl;
    std::cout << "  --speculative_algorithm <args>: 投机算法(dflash/mtp/dspark)" << std::endl;
    std::cout << "  --speculative_draft_model_path <args>: 草稿模型路径" << std::endl;
    std::cout << "  --draft_tokens <args>:       草稿token数" << std::endl;
    std::cout << "  --speculative_num_draft_tokens <args>: 草稿token数" << std::endl;
    std::cout << "  --speculative_dspark_block_size <args>: DSpark块大小" << std::endl;
    std::cout << "  --speculative_dspark_confidence_threshold <args>: DSpark置信阈值" << std::endl;
    std::cout << "  --triton:                    启用Triton CUDA算子(自动探测能import triton的python)" << std::endl;
    std::cout << "  --triton_python <args>:      指定triton用的python解释器" << std::endl;
    std::cout << std::endl;
    std::cout << "容量与缓存:" << std::endl;
    std::cout << "  --tokens <args>:             设置总的token数量(用于计算paged cache最大页数)" << std::endl;
    std::cout << "  --page_size <args>:          paged cache每页的token数" << std::endl;
    std::cout << "  --batch <args>:              并发请求数上限" << std::endl;
    std::cout << "  --max_batch <args>:          引擎侧最大batch" << std::endl;
    std::cout << "  --kv_cache_limit <args>:     KV Cache最大使用量，如 1g" << std::endl;
    std::cout << "  --chunked_prefill_size <args>: 分块prefill的切片大小，如8192" << std::endl;
    std::cout << "  --moe_pinned_slots <args>:    MoE专家权重流式的中转槽数，如32" << std::endl;
    std::cout << "  --moe_pinned_slot_mb <args>:  MoE专家权重流式的中转单槽宽(MB)，如8" << std::endl;
    std::cout << "  --max_context_length <args>: 单会话输入和输出合计的最大token数" << std::endl;
    std::cout << "  --rope_scaling <args>:       RoPE扩展配置(yarn或JSON)" << std::endl;
    std::cout << "  --cache_history <true/false>: 缓存历史对话(前缀KV复用)" << std::endl;
    std::cout << "  --cache_fast <true/false>:   是否启用快速缓存(会消耗一定显存)" << std::endl;
    std::cout << "  --prefix_cache <true/false>: 前缀缓存开关(FASTLLM_PREFIX_CACHE)" << std::endl;
    std::cout << "  --prefix_cache_snapshot_interval_pages <args>: 前缀缓存快照间隔页数" << std::endl;
    std::cout << "  --prefix_cache_snapshot_max_per_request <args>: 单请求最多保留的前缀缓存快照数" << std::endl;
    std::cout << "  --prefix_cache_snapshot_max_records <args>: 全局最多保留的前缀缓存快照数" << std::endl;
    std::cout << "  --fast_prefill:              DeepSeek-V4.1近似prefill(可能改变logits，默认关闭)" << std::endl;
    std::cout << "  --low_gpu_mem:               降低显存占用(强制关闭CUDA embedding，优先于--cuda_embedding)" << std::endl;
    std::cout << "  --vision_device <args>:      Qwen3.5视觉编码器设备(auto/cpu/cuda/cuda:N)" << std::endl;
    std::cout << "  --image-embedding-cache <args>: 图片embedding的CPU缓存上限，如512m或1g" << std::endl;
    std::cout << "  --gpu_mem_ratio <args>:      GPU显存使用比例，如0.9" << std::endl;
    std::cout << "  --cuda_slab <args>:          CUDA模型权重slab大小(MB)，0表示关闭" << std::endl;
    std::cout << std::endl;
    std::cout << "服务:" << std::endl;
    std::cout << "  --host <args>:               监听地址，默认0.0.0.0" << std::endl;
    std::cout << "  --port <args>:               端口号，默认8080" << std::endl;
    std::cout << "  --model_name <args>:         模型名(openai api中使用，调用时会核验)" << std::endl;
    std::cout << "  --api_key <args>:            设置后校验 Authorization: Bearer <key>，不设置则不校验" << std::endl;
    std::cout << "  --temperature <args>:        覆盖服务端默认temperature" << std::endl;
    std::cout << "  --top_p <args>:              覆盖服务端默认top_p" << std::endl;
    std::cout << "  --top_k <args>:              覆盖服务端默认top_k" << std::endl;
    std::cout << "  --repeat_penalty <args>:     覆盖服务端默认repeat_penalty" << std::endl;
    std::cout << "  --hide_input:                不打印请求内容" << std::endl;
    std::cout << "  --tool_call_parser <args>:   工具调用解析器(auto/qwen3_coder/hermes/none)" << std::endl;
    std::cout << "  --embedding_model <args>:    embedding/rerank模型(Bert)，用于/v1/embed与/v1/rerank" << std::endl;
    std::cout << "  --dev_mode:                  开发模式(启用对话列表与主动停止)" << std::endl;
}

// dtype字符串可能是"auto"、普通类型名，或 int4g<groupCnt> 形式
static fastllm::DataType ParseDataTypeArg(const std::string &argName, const std::string &text, int &groupCnt) {
    std::string dtypeStr = text;
    groupCnt = -1;
    if (dtypeStr.size() > 5 && dtypeStr.substr(0, 5) == "int4g") {
        groupCnt = atoi(dtypeStr.substr(5).c_str());
        dtypeStr = dtypeStr.substr(0, 5);
    }
    fastllm::AssertInFastLLM(dataTypeDict.find(dtypeStr) != dataTypeDict.end(),
                            "Unsupport " + argName + ": " + text);
    return dataTypeDict[dtypeStr];
}

void ParseArgs(int argc, char **argv, APIConfig &config) {
    std::vector<std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        } else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--low") {
            config.lowMemMode = true;
        } else if (sargv[i] == "--cuda_embedding"){
            config.cudaEmbedding = true;
        } else if (sargv[i] == "--port") {
            config.port = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--dtype") {
            // ftllm里--dtype默认auto，auto等价于float16
            std::string text = sargv[++i];
            if (text != "auto") {
                config.dtype = ParseDataTypeArg("data type", text, config.groupCnt);
            }
        } else if (sargv[i] == "--moe_dtype") {
            std::string text = sargv[++i];
            if (text != "auto" && text != "") {
                config.moeDtype = ParseDataTypeArg("moe data type", text, config.moeGroupCnt);
                config.useMoeDtype = true;
            }
        } else if (sargv[i] == "--tokens") {
            config.tokens = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--batch") {
            config.batch = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--atype") {
            std::string text = sargv[++i];
            if (text != "auto") {
                int groupCnt = -1;
                config.atype = ParseDataTypeArg("act type", text, groupCnt);
                config.useAtype = true;
            }
        } else if (sargv[i] == "--model_name") {
            config.modelName = sargv[++i];
        } else if (sargv[i] == "--api_key") {
            config.apiKey = sargv[++i];
        } else if (sargv[i] == "--device") {
            config.device = sargv[++i];
        } else if (sargv[i] == "--moe_device") {
            config.moeDevice = sargv[++i];
        } else if (sargv[i] == "--moe_device_layers") {
            config.moeDeviceLayers = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--moe_atype") {
            config.moeAtypeStr = sargv[++i];
        } else if (sargv[i] == "--kv_cache_dtype") {
            config.kvCacheDtypeStr = sargv[++i];
        } else if (sargv[i] == "--kv_cache_limit") {
            config.kvCacheLimit = ParseMemorySize(sargv[++i]);
        } else if (sargv[i] == "--max_batch") {
            config.maxBatch = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--chunked_prefill_size") {
            config.chunkedPrefillSize = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--moe_pinned_slots") {
            config.moePinnedStagingSlots = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--moe_pinned_slot_mb") {
            config.moePinnedStagingSlotMb = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--max_context_length" || sargv[i] == "--max-context-length") {
            config.maxContextLength = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--rope_scaling" || sargv[i] == "--rope-scaling") {
            config.ropeScaling = sargv[++i];
        } else if (sargv[i] == "--page_size") {
            config.pageSize = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--gpu_mem_ratio") {
            config.gpuMemRatio = (float)atof(sargv[++i].c_str());
        } else if (sargv[i] == "--cuda_slab") {
            config.cudaSlabMB = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--moe_experts") {
            config.moeExperts = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--lora") {
            config.lora = sargv[++i];
        } else if (sargv[i] == "--dtype_config") {
            config.dtypeConfig = sargv[++i];
        } else if (sargv[i] == "--ngram_device" || sargv[i] == "--ngram-device") {
            config.ngramDevice = sargv[++i];
        } else if (sargv[i] == "--moe_cuda_cache" || sargv[i] == "--moe-cuda-cache") {
            config.moeCudaCache = ParseBinaryMemorySize(sargv[++i]);
        } else if (sargv[i] == "--moe_cpu_cache" || sargv[i] == "--moe-cpu-cache") {
            config.moeCpuCache = ParseBinaryMemorySize(sargv[++i]);
        } else if (sargv[i] == "--low_gpu_mem") {
            config.lowGpuMem = true;
        } else if (sargv[i] == "--fast_prefill" || sargv[i] == "--fast-prefill") {
            config.fastPrefill = true;
        } else if (sargv[i] == "--prefix_cache" || sargv[i] == "--prefix-cache") {
            config.prefixCache = sargv[++i];
        } else if (sargv[i] == "--prefix_cache_snapshot_interval_pages" || sargv[i] == "--prefix-cache-snapshot-interval-pages") {
            config.prefixCacheSnapshotIntervalPages = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--prefix_cache_snapshot_max_per_request" || sargv[i] == "--prefix-cache-snapshot-max-per-request") {
            config.prefixCacheSnapshotMaxPerRequest = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--prefix_cache_snapshot_max_records" || sargv[i] == "--prefix-cache-snapshot-max-records") {
            config.prefixCacheSnapshotMaxRecords = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--vision_device" || sargv[i] == "--vision-device") {
            config.visionDevice = sargv[++i];
        } else if (sargv[i] == "--image-embedding-cache" || sargv[i] == "--image_embedding_cache") {
            config.imageEmbeddingCache = ParseBinaryMemorySize(sargv[++i]);
        } else if (sargv[i] == "--cuda_shared_expert" || sargv[i] == "--cuda_se") {
            config.cudaSharedExpert = sargv[++i];
        } else if (sargv[i] == "--enable_amx" || sargv[i] == "--amx") {
            config.enableAmx = sargv[++i];
        } else if (sargv[i] == "--cache_history") {
            config.cacheHistory = sargv[++i];
        } else if (sargv[i] == "--cache_fast") {
            config.cacheFast = sargv[++i];
        } else if (sargv[i] == "--host") {
            config.host = sargv[++i];
        } else if (sargv[i] == "--temperature") {
            config.temperature = (float)atof(sargv[++i].c_str());
            config.cliTemperature = true;
        } else if (sargv[i] == "--top_p") {
            config.topP = (float)atof(sargv[++i].c_str());
            config.cliTopP = true;
        } else if (sargv[i] == "--top_k") {
            config.topK = atoi(sargv[++i].c_str());
            config.cliTopK = true;
        } else if (sargv[i] == "--repeat_penalty" || sargv[i] == "--repetition_penalty") {
            config.repeatPenalty = (float)atof(sargv[++i].c_str());
            config.cliRepeatPenalty = true;
        } else if (sargv[i] == "--tool_call_parser" || sargv[i] == "--tool-call-parser") {
            config.toolCallParser = sargv[++i];
        } else if (sargv[i] == "--embedding_model" || sargv[i] == "--embedding-model") {
            config.embeddingModel = sargv[++i];
        } else if (sargv[i] == "--mtp") {
            config.mtp = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--mtp_fp8_draft_head" || sargv[i] == "--mtp-fp8-draft-head") {
            config.mtpFp8DraftHead = sargv[++i];
        } else if (sargv[i] == "--tp") {
            config.tp = sargv[++i];
        } else if (sargv[i] == "--dspark") {
            config.dspark = sargv[++i];
        } else if (sargv[i] == "--draft_tokens" || sargv[i] == "--draft-tokens") {
            config.draftTokens = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--speculative_num_draft_tokens" || sargv[i] == "--speculative-num-draft-tokens") {
            config.speculativeNumDraftTokens = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--speculative_algorithm" || sargv[i] == "--speculative-algorithm") {
            config.speculativeAlgorithm = sargv[++i];
        } else if (sargv[i] == "--speculative_draft_model_path" || sargv[i] == "--speculative-draft-model-path") {
            config.speculativeDraftModelPath = sargv[++i];
        } else if (sargv[i] == "--speculative_dspark_block_size" || sargv[i] == "--speculative-dspark-block-size") {
            config.speculativeDsparkBlockSize = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--speculative_dspark_confidence_threshold" || sargv[i] == "--speculative-dspark-confidence-threshold") {
            config.speculativeDsparkConfidenceThreshold = (float)atof(sargv[++i].c_str());
        } else if (sargv[i] == "--triton") {
            config.triton = true;
        } else if (sargv[i] == "--triton_python" || sargv[i] == "--triton-python") {
            config.tritonPython = sargv[++i];
        } else if (sargv[i] == "--hide_input") {
            config.hideInput = true;
        } else if (sargv[i] == "--dev_mode") {
            config.devMode = true;
        } else if (!sargv[i].empty() && sargv[i][0] != '-') {
            // ftllm的位置参数就是模型路径
            config.path = sargv[i];
        } else {
            printf("Unsupported argument: %s\n", sargv[i].c_str());
            Usage();
            exit(-1);
        }
    }
}

char buff[1024 * 1024] = {0};
std::string url = "generate";
std::mutex locker;

int main(int argc, char** argv) {
    // 输出重定向到文件时stdout默认是块缓冲，日志会滞留在缓冲区，进程崩掉时全部丢失。
    // 注意这里必须给size，size=0会让CRT建出零长度缓冲区，一写就越界崩溃
    setvbuf(stdout, NULL, _IOLBF, 4096);
    ParseArgs(argc, argv, config);

    fastllm::SetThreads(config.threads);
    fastllm::SetLowMemMode(config.lowMemMode);
    // ftllm: --low_gpu_mem 优先于--cuda_embedding，效果是强制关闭cuda embedding
    if (config.cudaEmbedding || config.lowGpuMem) {
        fastllm::SetCudaEmbedding(config.cudaEmbedding);
    }
    // 以下全局配置必须在建模型之前设置，模型构造时才会读到
    if (config.gpuMemRatio >= 0.0f) {
        fastllm::SetGpuMemRatio(config.gpuMemRatio);
    }
    if (config.cudaSlabMB > 0) {
        fastllm::SetCudaSlabMB(config.cudaSlabMB);
    }
    if (config.moeCudaCache > 0) {
        fastllm::SetMoeCudaCacheBytes((uint64_t)config.moeCudaCache);
    }
    if (config.moeCpuCache > 0) {
        fastllm::SetMoeCpuCacheBytes((uint64_t)config.moeCpuCache);
    }
    if (!config.ngramDevice.empty()) {
        fastllm::SetNgramDevice(config.ngramDevice);
    }
    if (!config.cudaSharedExpert.empty()) {
        fastllm::SetCudaSharedExpert(ParseBoolStr(config.cudaSharedExpert));
    }
    if (!config.enableAmx.empty()) {
        fastllm::EnableAMX(ParseBoolStr(config.enableAmx));
    }
    // tokensLimit/promptLimit是加载时按KV cache实际容量标定出来的，
    // 必须在建模型之前用全局SetMaxTokens设置，直接改model->tokensLimit会和标定结果不一致
    if (config.tokens > 0) {
        fastllm::SetMaxTokens(config.tokens);
    }
    // ftllm: 未指定--page_size且设备含multicuda时用16(可用FASTLLM_MULTICUDA_PAGE_SIZE覆盖)
    if (config.pageSize <= 0 &&
        (config.device.find("multicuda") != std::string::npos ||
         config.moeDevice.find("multicuda") != std::string::npos)) {
        const char *envPageSize = getenv("FASTLLM_MULTICUDA_PAGE_SIZE");
        config.pageSize = (envPageSize != NULL) ? atoi(envPageSize) : 16;
    }
    if (config.pageSize > 0) {
        fastllm::SetPageLen(config.pageSize);
    }
    // 下面这些开关库只从环境变量读取，必须在建模型之前设置
    if (!config.prefixCache.empty()) {
        SetEnvVar("FASTLLM_PREFIX_CACHE", config.prefixCache);
    }
    if (config.prefixCacheSnapshotIntervalPages > 0) {
        SetEnvVar("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", std::to_string(config.prefixCacheSnapshotIntervalPages));
    }
    if (config.prefixCacheSnapshotMaxPerRequest > 0) {
        SetEnvVar("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_PER_REQUEST", std::to_string(config.prefixCacheSnapshotMaxPerRequest));
    }
    if (config.prefixCacheSnapshotMaxRecords > 0) {
        SetEnvVar("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS", std::to_string(config.prefixCacheSnapshotMaxRecords));
    }
    // 与ftllm一致：该开关总是显式设置，默认关闭
    SetEnvVar("FASTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY", config.fastPrefill ? "1" : "0");
    if (config.imageEmbeddingCache >= 0) {
        SetEnvVar("FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES", std::to_string(config.imageEmbeddingCache));
    }
    if (!config.visionDevice.empty()) {
        SetEnvVar("FASTLLM_QWEN35_VISION_DEVICE", config.visionDevice);
    }
    // 投机解码/草稿模型同样只从环境变量读取
    if (!config.tp.empty()) {
        SetEnvVar("FASTLLM_TP", config.tp);
    }
    {
        int draftTokens = config.draftTokens > 0 ? config.draftTokens : config.speculativeNumDraftTokens;
        std::string algorithm = ToLowerString(config.speculativeAlgorithm);
        std::string draftPath = config.speculativeDraftModelPath;
        if (!config.dspark.empty()) {
            draftPath = config.dspark;
            algorithm = "dspark";
        } else if (!draftPath.empty() && algorithm.empty()) {
            // 与ftllm一致：只给了草稿模型没给算法时按dspark处理
            algorithm = "dspark";
        }
        if (algorithm == "dflash" && !draftPath.empty()) {
            SetEnvVar("FASTLLM_DFLASH_MODEL_PATH", draftPath);
            int blockSize = config.speculativeDsparkBlockSize > 0 ?
                            config.speculativeDsparkBlockSize : draftTokens;
            if (blockSize > 0) {
                SetEnvVar("FASTLLM_DFLASH_BLOCK_SIZE", std::to_string(blockSize));
            }
        } else if (algorithm == "mtp") {
            if (config.mtp <= 0 && draftTokens > 0) {
                config.mtp = draftTokens;
            }
            if (config.mtp <= 0 && !draftPath.empty()) {
                config.mtp = 5;
            }
        } else if (algorithm == "dspark" || !draftPath.empty()) {
            if (!draftPath.empty()) {
                SetEnvVar("FASTLLM_DSPARK_MODEL_PATH", draftPath);
            }
            if (draftTokens > 0) {
                SetEnvVar("FASTLLM_DSPARK_TOKENS", std::to_string(draftTokens));
            }
            float threshold = config.speculativeDsparkConfidenceThreshold >= 0.0f ?
                              config.speculativeDsparkConfidenceThreshold : 0.5f;
            SetEnvVar("FASTLLM_DSPARK_CONFIDENCE_THRESHOLD", std::to_string(threshold));
        }
    }
    SetEnvVar("FASTLLM_QWEN35_ENABLE_MTP", std::to_string(config.mtp));
    SetEnvVar("FASTLLM_QWEN4_ENABLE_MTP", std::to_string(config.mtp));
    SetEnvVar("FASTLLM_GLM5_NEXT_ENABLE_MTP", std::to_string(config.mtp));
    if (!config.mtpFp8DraftHead.empty()) {
        SetEnvVar("FASTLLM_MTP_FP8_DRAFT_HEAD", ParseBoolStr(config.mtpFp8DraftHead) ? "1" : "0");
    }
    ConfigureTriton();
    // device map 必须在创建模型之前设置：模型构造时会把它们拷进自己的成员
    if (!config.device.empty()) {
        fastllm::SetDeviceMap(ParseDeviceMap(config.device));
    }
    if (!config.moeDevice.empty()) {
        if (config.device.empty()) {
            printf(u8"[fastllm] --moe_device 需要与 --device 一起使用，本次已忽略。\n");
        } else if (config.moeDeviceLayers >= 0) {
            fastllm::SetMoeDeviceMap(ParseDeviceMap(config.device));
            fastllm::SetLayeredMoeDeviceMap(ParseDeviceMap(config.moeDevice));
            fastllm::SetMoeDeviceLayers(config.moeDeviceLayers);
        } else {
            fastllm::SetMoeDeviceMap(ParseDeviceMap(config.moeDevice));
        }
    }
    printf(u8"[fastllm] device = %s, moe = %s, threads = %d\n",
           config.device.empty() ? "(default)" : config.device.c_str(),
           config.moeDevice.empty() ? "(follow device)" : config.moeDevice.c_str(),
           fastllm::GetThreads());

    if (!fastllm::FileExists(config.path)) {
        printf("模型文件 %s 不存在！\n", config.path.c_str());
        exit(0);
    }
    bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
    // --max_context_length/--rope_scaling只能通过构造参数传进上下文规划，构造后无法补
    fastllm::ContextOptions contextOptions;
    contextOptions.maxLength = config.maxContextLength;
    contextOptions.ropeScaling = config.ropeScaling;
    workQueue.model = isHFDir ? fastllm::CreateLLMModelFromHF(config.path, config.dtype, config.groupCnt,
                                                              false, "", config.lora, false,
                                                              config.useMoeDtype, config.moeDtype, config.moeGroupCnt,
                                                              config.dtypeConfig, contextOptions)
        : fastllm::CreateLLMModelFromFile(config.path);
    if (config.useAtype) {
        workQueue.model->SetDataType(config.atype);
    }
    if (!config.moeAtypeStr.empty() && config.moeAtypeStr != "auto") {
        int moeAtypeGroupCnt = -1;
        workQueue.model->SetMoeAtype(ParseDataTypeArg("moe atype", config.moeAtypeStr, moeAtypeGroupCnt));
    }
    if (!config.kvCacheDtypeStr.empty() && config.kvCacheDtypeStr != "auto") {
        int kvDtypeGroupCnt = -1;
        workQueue.model->SetKVCacheDataType(ParseDataTypeArg("kv cache dtype", config.kvCacheDtypeStr, kvDtypeGroupCnt));
    }
    if (!config.lora.empty()) {
        workQueue.model->SetAdapter(config.lora);
    }
    // ftllm服务端的--cache_history默认关闭，这里默认打开：库只在saveHistoryChat为true时
    // 才会把前缀KV记进pastKVCacheManager(basellm.cpp:523)，否则多轮对话每轮都要把整段
    // prompt重新prefill(4k token≈90s)。需要关掉可显式传 --cache_history false
    if (config.cacheHistory.empty() || ParseBoolStr(config.cacheHistory)) {
        workQueue.model->SetSaveHistoryChat(true);
        // ftllm在未开启--cache_fast时把历史cache放在CPU内存
        if (!ParseBoolStr(config.cacheFast)) {
            fastllm::SetHistoryCacheInCPU(true);
        }
    }
    if (config.moeExperts > 0) {
        workQueue.model->SetMoeExperts(config.moeExperts);
    }
    if (config.maxBatch > 0) {
        // 与pytools的set_max_batch_llm_model一致：不支持batch且不支持并发轮转的模型强制为1
        workQueue.model->maxBatch = (!workQueue.model->canDoBatchForward && !workQueue.model->canDoConcurrentForward) ?
                                    1 : config.maxBatch;
    }
    if (config.kvCacheLimit > 0) {
        workQueue.model->kvCacheLimit = config.kvCacheLimit;
    }
    if (config.chunkedPrefillSize > 0) {
        workQueue.model->SetChunkedPrefillSize(config.chunkedPrefillSize);
    }
    if (config.moePinnedStagingSlots > 0 || config.moePinnedStagingSlotMb > 0) {
        fastllm::SetMoePinnedStaging(
            config.moePinnedStagingSlots > 0 ? config.moePinnedStagingSlots : 16,
            config.moePinnedStagingSlotMb > 0 ? config.moePinnedStagingSlotMb : 4);
    }
    if (!isHFDir && config.maxContextLength > 0 && config.ropeScaling.empty()) {
        // 非HF加载器只支持"只缩小"的上下文限制，与pytools的set_max_context_length_llm_model一致
        if (config.maxContextLength < workQueue.model->max_positions) {
            workQueue.model->max_positions = config.maxContextLength;
        }
    }
    // 启动时定好工具调用解析器：显式给了不支持的名字会在这里报错退出
    gToolCallParser = ResolveToolCallParser(workQueue.model.get());
    // 可选的embedding/rerank模型；不指定则/v1/embed与/v1/rerank返回400
    if (!config.embeddingModel.empty()) {
        printf("[fastllm] loading embedding model: %s\n", config.embeddingModel.c_str());
        fflush(stdout);
        gEmbeddingModel = fastllm::CreateEmbeddingModelFromFile(config.embeddingModel);
    }
    // 与ftllm一致：HF目录下的generation_config.json在do_sample为true时覆盖默认采样参数
    if (isHFDir) {
        std::ifstream generationConfigFile(config.path + "/generation_config.json");
        if (generationConfigFile) {
            std::stringstream buffer;
            buffer << generationConfigFile.rdbuf();
            std::string parseError;
            json11::Json genCfg = json11::Json::parse(buffer.str(), parseError);
            if (parseError.empty() && genCfg["do_sample"].is_bool() && genCfg["do_sample"].bool_value()) {
                // CLI显式给过的参数优先，不被模型自带的配置覆盖
                if (!config.cliRepeatPenalty && genCfg["repetition_penalty"].is_number()) {
                    config.repeatPenalty = (float)genCfg["repetition_penalty"].number_value();
                }
                if (!config.cliTopP && genCfg["top_p"].is_number()) {
                    config.topP = (float)genCfg["top_p"].number_value();
                }
                if (!config.cliTopK && genCfg["top_k"].is_number()) {
                    config.topK = (int)genCfg["top_k"].number_value();
                }
                if (!config.cliTemperature && genCfg["temperature"].is_number()) {
                    config.temperature = (float)genCfg["temperature"].number_value();
                }
            }
        }
    }
    printf("[fastllm] default generation config: temperature=%g, top_p=%g, top_k=%d, repeat_penalty=%g\n",
           config.temperature, config.topP, config.topK, config.repeatPenalty);
    fflush(stdout);
    workQueue.maxActivateQueryNumber = std::max(1, std::min(256, config.batch));
    workQueue.Start();

#ifdef _WIN32
    // Windows下必须先初始化winsock，否则socket()/accept()等一律失败
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cout << "WSAStartup error!" << std::endl;
        exit(-1);
    }
#endif
    socket_t local_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (local_fd == INVALID_SOCKET) {
        std::cout << "socket error!" << std::endl;
        exit(-1);
    }
    std::cout << "socket ready!" << std::endl;

    int opt = 1;
    setsockopt(local_fd, SOL_SOCKET, SO_REUSEADDR, (const char *) &opt, sizeof(opt));

    struct sockaddr_in local_addr;
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(config.port);  //绑定端口
    // --host 支持 0.0.0.0 / 127.0.0.1 / 具体网卡地址
    if (config.host.empty() || config.host == "0.0.0.0") {
        local_addr.sin_addr.s_addr = INADDR_ANY;
    } else if (inet_pton(AF_INET, config.host.c_str(), &local_addr.sin_addr) != 1) {
        printf("[fastllm] --host %s 不是合法IPv4地址，已回退到0.0.0.0\n", config.host.c_str());
        local_addr.sin_addr.s_addr = INADDR_ANY;
    }

    //3.bind()： 将一个网络地址与一个套接字绑定，此处将本地地址绑定到一个套接字上
    int res = bind(local_fd, (struct sockaddr *) &local_addr, sizeof(local_addr));
    if (res == -1) {
        std::cout << "bind error!" << std::endl;
        exit(-1);
    }
    std::cout << "bind ready!" << std::endl;
    listen(local_fd, 2000);    
    // 输出重定向到文件时stdout是块缓冲，这里必须flush，否则看不到服务已就绪
    printf("start...\n");
    fflush(stdout);
    int queuePos = 0;
    while (true) { //循环接收客户端的请求
        //5.创建一个sockaddr_in结构体，用来存储客户机的地址
        struct sockaddr_in client_addr;
        socklen_t len = sizeof(client_addr);
        //6.accept()函数：阻塞运行，直到收到某一客户机的连接请求，并返回客户机的描述符
        socket_t client = accept(local_fd, (struct sockaddr *) &client_addr, &len);
        if (client == INVALID_SOCKET) {
            // 单次accept失败(连接被对端重置等)不应该终止整个服务
            printf("accept error!\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        int size = 0;
        bool readFailed = false;
        bool sentContinue = false;
#ifdef _WIN32
        DWORD recvTimeoutMs = 10000;
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (const char *) &recvTimeoutMs, sizeof(recvTimeoutMs));
#else
        struct timeval recvTimeout;
        recvTimeout.tv_sec = 10;
        recvTimeout.tv_usec = 0;
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, (const char *) &recvTimeout, sizeof(recvTimeout));
#endif
        while (true) {
            int cur = SocketRead(client, buff + size, (int)(sizeof(buff) - size));
            if (cur <= 0) {
                // 对端关闭连接或读取出错
                readFailed = true;
                break;
            }
            size += cur;
            if (size >= (int)sizeof(buff)) {
                // 请求超过缓冲区上限，丢弃这次连接：此处必须提前拦，否则下面的buff[size]=0会越界
                printf("[fastllm-api] request too large, dropped\n");
                fflush(stdout);
                readFailed = true;
                break;
            }
            buff[size] = 0;
            // curl等客户端对较大的body会先发Expect: 100-continue并等待应答才发body，
            // 这里必须回一个100，否则双方互等直到超时
            if (!sentContinue) {
                char *headerEnd = strstr(buff, "\r\n\r\n");
                if (headerEnd != NULL) {
                    std::string lower(buff, headerEnd - buff);
                    for (size_t i = 0; i < lower.size(); i++) {
                        lower[i] = (char)tolower((unsigned char)lower[i]);
                    }
                    if (lower.find("expect:") != std::string::npos &&
                        lower.find("100-continue") != std::string::npos) {
                        SocketWrite(client, "HTTP/1.1 100 Continue\r\n\r\n", 25);
                        sentContinue = true;
                    }
                }
            }
            if (httpChecker.IsValid(buff, size)) {
                break;
            }
        }
        if (readFailed) {
            printf("[fastllm-api] connection dropped before a complete request\n");
            fflush(stdout);
            SocketClose(client);
            continue;
        }
        buff[size] = 0;

        // 控制面请求直接起线程走Deal，不经过工作队列，保证cancel在忙时也能及时生效
        HttpRequest controlChecker;
        char *controlBuffer = buff;
        controlChecker.Init(controlBuffer);
        if (IsControlRoute(controlChecker)) {
            std::vector<char> bodyCopy(buff, buff + size + 1);
            std::thread([bodyCopy, client]() {
                WorkNode *node = new WorkNode();
                try {
                    node->Init(const_cast<char *>(bodyCopy.data()), client);
                    workQueue.Deal(node);
                } catch (const std::exception &e) {
                    printf("[fastllm-api] control request failed: %s\n", e.what());
                    fflush(stdout);
                    SocketClose(client);
                } catch (...) {
                    SocketClose(client);
                }
                delete node;
            }).detach();
            continue;
        }

        while (workQueue.q.size() > workQueue.maxActivateQueryNumber) {
            fastllm::MySleep(0);
        }
        workQueue.Push(buff, client);
    }

    return 0;
}
