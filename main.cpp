#include "model.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <stdlib.h>
#include <windows.h>
#endif

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
    {"fp8", fastllm::DataType::FP8_E4M3},
    {"float8", fastllm::DataType::FP8_E4M3},
    {"fp8_e4m3", fastllm::DataType::FP8_E4M3}
};

struct RunConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    std::string systemPrompt = "";
    std::set <std::string> eosToken;
    int threads = 0; // 使用的线程数，<= 0 表示按CPU核数自动选择
    bool lowMemMode = false; // 是否使用低内存模式

    std::string device = ""; // 主计算设备，例如 cuda、cuda:0、cpu、numa
    std::string moeDevice = ""; // MoE 专家层设备，例如 cpu、numa
    int moeDeviceLayers = -1; // 仅最后 N 层 MoE 使用 moeDevice，-1 表示全部

    fastllm::DataType dtype = fastllm::DataType::FLOAT16;
    fastllm::DataType moeDtype = fastllm::DataType::FLOAT32;
    fastllm::DataType atype = fastllm::DataType::FLOAT32;
    fastllm::DataType kvDtype = fastllm::DataType::FLOAT32; // 仅 --kv_dtype 显式指定时生效
    bool useKvDtype = false;
    bool useAtype = false; // 仅在显式传入 --atype 时调用 SetDataType
    int groupCnt = -1;
    int moeGroupCnt = -1;
    bool useMoeDtype = false;
};

// 与 ftllm 的默认值保持一致：留两个核给系统调度，最多 32 个线程
static int DefaultThreadNum() {
    unsigned int cores = std::thread::hardware_concurrency();
    if (cores == 0) {
        return 4;
    }
    int available = (int)cores - 2;
    return available > 0 ? std::min(32, available) : 1;
}

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

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量，不填时按CPU核数自动选择" << std::endl;
    std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--device> <args>:            主计算设备，如 cuda、cuda:0、cpu、numa 或 \"{'cuda':1,'numa':8}\"" << std::endl;
    std::cout << "<--moe_device> <args>:        MoE专家层设备，如 cpu、numa，需与--device一起使用" << std::endl;
    std::cout << "<--moe_device_layers> <args>: 仅最后N层MoE使用--moe_device，-1表示全部" << std::endl;
    std::cout << "<--system> <args>:            设置系统提示词(system prompt)" << std::endl;
    std::cout << "<--eos_token> <args>:         设置eos token" << std::endl;
    std::cout << "<--dtype> <args>:             设置权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<--moe_dtype> <args>:         设置MoE expert权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<--atype> <args>:             设置推理使用的数据类型(float32/float16)" << std::endl;
    std::cout << "<--kv_dtype> <args>:          设置KV cache数据类型(float32/float16/bfloat16/fp8/fp4)，不填则跟随atype" << std::endl;
    std::cout << "<--top_p> <args>:             采样参数top_p" << std::endl;
    std::cout << "<--top_k> <args>:             采样参数top_k" << std::endl;
    std::cout << "<--temperature> <args>:       采样参数温度，越高结果越不固定" << std::endl;
    std::cout << "<--repeat_penalty> <args>:    采样参数重复惩罚" << std::endl;
}

void ParseArgs(int argc, char **argv, RunConfig &config, fastllm::GenerationConfig &generationConfig) {
    std::vector <std::string> sargv;
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
        } else if (sargv[i] == "--device") {
            config.device = sargv[++i];
        } else if (sargv[i] == "--moe_device") {
            config.moeDevice = sargv[++i];
        } else if (sargv[i] == "--moe_device_layers") {
            config.moeDeviceLayers = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-m" || sargv[i] == "--model") {
            i++;
        } else if (sargv[i] == "--top_p") {
            generationConfig.top_p = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--top_k") {
            generationConfig.top_k = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--temperature") {
            generationConfig.temperature = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--repeat_penalty") {
            generationConfig.repeat_penalty = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--system") {
            config.systemPrompt = sargv[++i];
        } else if (sargv[i] == "--eos_token") {
            config.eosToken.insert(sargv[++i]);
        } else if (sargv[i] == "--dtype") {
            std::string dtypeStr = sargv[++i];
            if (dtypeStr.size() > 5 && dtypeStr.substr(0, 5) == "int4g") {
                config.groupCnt = atoi(dtypeStr.substr(5).c_str());
                dtypeStr = dtypeStr.substr(0, 5);
            }
            fastllm::AssertInFastLLM(dataTypeDict.find(dtypeStr) != dataTypeDict.end(),
                                    "Unsupport data type: " + dtypeStr);
            config.dtype = dataTypeDict[dtypeStr];
        } else if (sargv[i] == "--moe_dtype") {
            std::string dtypeStr = sargv[++i];
            if (dtypeStr.size() > 5 && dtypeStr.substr(0, 5) == "int4g") {
                config.moeGroupCnt = atoi(dtypeStr.substr(5).c_str());
                dtypeStr = dtypeStr.substr(0, 5);
            }
            fastllm::AssertInFastLLM(dataTypeDict.find(dtypeStr) != dataTypeDict.end(),
                                    "Unsupport moe data type: " + dtypeStr);
            config.moeDtype = dataTypeDict[dtypeStr];
            config.useMoeDtype = true;
        } else if (sargv[i] == "--atype") {
            std::string atypeStr = sargv[++i];
            fastllm::AssertInFastLLM(dataTypeDict.find(atypeStr) != dataTypeDict.end(),
                                    "Unsupport act type: " + atypeStr);
            config.atype = dataTypeDict[atypeStr];
            config.useAtype = true;
        } else if (sargv[i] == "--kv_dtype") {
            std::string kvStr = sargv[++i];
            fastllm::AssertInFastLLM(dataTypeDict.find(kvStr) != dataTypeDict.end(),
                                    "Unsupport kv cache type: " + kvStr);
            config.kvDtype = dataTypeDict[kvStr];
            config.useKvDtype = true;
        } else {
            Usage();
            exit(-1);
        }
    }
}

#ifdef _WIN32
// 控制台读到的字节按当前输入代码页(中文系统为GBK)编码，需要转成UTF-8再交给模型
static std::string ConsoleInputToUtf8(const std::string &text) {
    if (text.empty()) {
        return text;
    }
    UINT codePage = GetConsoleCP();
    int wideLen = MultiByteToWideChar(codePage, 0, text.c_str(), (int)text.size(), NULL, 0);
    if (wideLen <= 0) {
        return text;
    }
    std::wstring wide(wideLen, L'\0');
    MultiByteToWideChar(codePage, 0, text.c_str(), (int)text.size(), &wide[0], wideLen);
    int utf8Len = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), wideLen, NULL, 0, NULL, NULL);
    if (utf8Len <= 0) {
        return text;
    }
    std::string utf8(utf8Len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), wideLen, &utf8[0], utf8Len, NULL, NULL);
    return utf8;
}
#endif

int main(int argc, char **argv) {
#ifdef _WIN32
    // 只把输出代码页切到UTF-8，不能再用 chcp 连输入代码页一起切，
    // 否则控制台输入会卡住或丢失多字节字符
    SetConsoleOutputCP(CP_UTF8);
#endif
    RunConfig config;
    fastllm::GenerationConfig generationConfig;
    ParseArgs(argc, argv, config, generationConfig);

    fastllm::PrintInstructionInfo();
    fastllm::SetThreads(config.threads > 0 ? config.threads : DefaultThreadNum());
    fastllm::SetLowMemMode(config.lowMemMode);
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
        printf(u8"模型文件 %s 不存在！\n", config.path.c_str());
        exit(0);
    }
    bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
    auto model = !isHFDir ? fastllm::CreateLLMModelFromFile(config.path) :
                 fastllm::CreateLLMModelFromHF(config.path, config.dtype, config.groupCnt,
                                               false, "", "", false,
                                               config.useMoeDtype, config.moeDtype, config.moeGroupCnt);
    if (config.useKvDtype) {
        model->SetKVCacheDataType(config.kvDtype);
    }
    if (config.useAtype) {
        model->SetDataType(config.atype);
    }
    model->SetSaveHistoryChat(true);
    
    for (auto &it : config.eosToken) {
        generationConfig.stop_token_ids.insert(model->weight.tokenizer.GetTokenId(it));
    }
    std::string systemConfig = config.systemPrompt;
    fastllm::ChatMessages messages = config.systemPrompt.empty() ? fastllm::ChatMessages() : fastllm::ChatMessages({{"system", systemConfig}});

    static std::string modelType = model->model_type;
    printf(u8"欢迎使用 %s 模型. 输入内容对话，reset清空历史记录，stop退出程序.\n", model->model_type.c_str());

    while (true) {
        printf(u8"用户: ");
        fflush(stdout);
        std::string input;
        if (!std::getline(std::cin, input)) {
            // 输入流结束或出错，直接退出，避免用空输入反复触发模型
            printf("\n");
            break;
        }
#ifdef _WIN32
        input = ConsoleInputToUtf8(input);
#endif
        if (input == "reset") {
            messages = config.systemPrompt.empty() ? fastllm::ChatMessages() : fastllm::ChatMessages({{"system", config.systemPrompt}});
            continue;
        }
        if (input == "stop") {
            break;
        }
        if (input.empty()) {
            continue;
        }
        messages.push_back(std::make_pair("user", input));
        fastllm::ClearProfileSummary();
        auto profileStartTime = std::chrono::steady_clock::now();
        std::string ret = model->Response(model->ApplyChatTemplate(messages), [](int index, const char* content) {
            if (index == 0) {
                printf("%s:%s", modelType.c_str(), content);
                fflush(stdout);
            }
            if (index > 0) {
                printf("%s", content);
                fflush(stdout);
            }
            if (index == -1) {
                printf("\n");
            }
        }, generationConfig);
        double profileSpend = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profileStartTime).count();
        int outputTokens = model->weight.tokenizer.Encode(ret).Count(0);
        printf("[fastllm-profile] output tokens = %d, spend = %f s, tokens / s = %f\n",
               outputTokens, profileSpend,
               profileSpend > 0.0 ? (double)outputTokens / profileSpend : 0.0);
        fflush(stdout);
        fastllm::PrintProfileSummary();
        messages.push_back(std::make_pair("assistant", ret));
    }

    return 0;
}
