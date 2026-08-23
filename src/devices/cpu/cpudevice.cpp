//
// Created by huangyuyang on 6/13/23.
//

#define _USE_MATH_DEFINES
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/computeutils.h"
#include "devices/cpu/kimi_k3_ops.h"

#include <cstring>
#include <thread>
#include <chrono>
#include <sstream>

#include <cfloat>
#include <cmath>
#include <atomic>
#include <set>
#include <mutex>
#include <unordered_map>

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#ifdef __AVX2__
#include "avxMath.h"
#endif

#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

#ifdef USE_NUMAS
#include "numas.h"
#elif defined(USE_CPU_NUMA)
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

#include "utils.h"
#include "gguf.h"

namespace fastllm {
    void Float32ToBFloat16(float *float32, uint16_t *bfloat16, int len);
    extern bool Float32ToBFloat16_AVX512BF16_RNE(float *float32, uint16_t *bfloat16, int len);
    extern bool FastllmGemmBFloat16NVFP4Block16_AVX512BF16(
        const void *A, long lda, const void *B, long ldb, void *C, long ldc,
        int n, int m, int k, int st, int end);
    extern bool FastllmGemmBFloat16NVFP4Block16E8M0_AVX512BF16(
        const void *A, long lda, const void *B, long ldb, void *C, long ldc,
        int n, int m, int k, int st, int end);
    extern bool FastllmGemmBFloat16NVFP4Block32E8M0_AVX512BF16(
        const void *A, long lda, const void *B, long ldb, void *C, long ldc,
        int n, int m, int k, int st, int end);
    extern bool FastllmGemmFloat32NVFP4Block16_AVX512BF16(
        const void *A, long lda, const void *B, long ldb, void *C, long ldc,
        int n, int m, int k, int st, int end);
    extern bool FastllmGemmFloat32NVFP4Block16E8M0_AVX512BF16(
        const void *A, long lda, const void *B, long ldb, void *C, long ldc,
        int n, int m, int k, int st, int end);

    static double CpuProfileNowMs() {
        using Clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
    }

    struct MultiThreadDeepSeekV4MoeDownPrepareOp : MultiThreadBaseOp {
        float *gateUpData;
        float *swigluData;
        uint16_t *downInputData;
        int mid, st, end;
        bool routed, computeActivation, quantize, convert;
        float routeWeight, swigluLimit;

        MultiThreadDeepSeekV4MoeDownPrepareOp(
            float *gateUpData, float *swigluData, uint16_t *downInputData,
            int mid, int st, int end, bool routed, float routeWeight,
            float swigluLimit, bool computeActivation, bool quantize,
            bool convert
        ) : gateUpData(gateUpData), swigluData(swigluData),
            downInputData(downInputData), mid(mid), st(st), end(end),
            routed(routed), computeActivation(computeActivation),
            quantize(quantize), convert(convert), routeWeight(routeWeight),
            swigluLimit(swigluLimit) {}

        void Run() override {
            if (computeActivation) {
                for (int i = st; i < end; i++) {
                    float gate = RoundFloat32ToBFloat16RNE(gateUpData[i]);
                    float up = RoundFloat32ToBFloat16RNE(gateUpData[mid + i]);
                    if (routed && swigluLimit > 0.0f) {
                        gate = std::min(gate, swigluLimit);
                        up = std::max(-swigluLimit, std::min(up, swigluLimit));
                    }
                    float h = (gate / (1.0f + std::exp(-gate))) * up;
                    swigluData[i] = RoundFloat32ToBFloat16RNE(routeWeight * h);
                }
            }
            if (quantize) {
                // Tasks are block-128 aligned, so quantizing a task at a time
                // preserves the official per-block activation scales exactly.
                QuantizeDequantizeFP8E4M3Block128(swigluData + st, end - st);
            }
            if (convert) {
                Float32ToBFloat16(swigluData + st, downInputData + st, end - st);
            }
        }
    };

    struct MultiThreadDeepSeekV4MoeReduceOp : MultiThreadBaseOp {
        std::vector<std::vector<float>> *results;
        float *output;
        int expertCount, st, end;

        MultiThreadDeepSeekV4MoeReduceOp(
            std::vector<std::vector<float>> *results, float *output,
            int expertCount, int st, int end
        ) : results(results), output(output), expertCount(expertCount),
            st(st), end(end) {}

        void Run() override {
            for (int expert = 0; expert < expertCount; expert++) {
                float *curOutput = (*results)[expert].data();
                for (int d = st; d < end; d++) {
                    curOutput[d] =
                        RoundFloat32ToBFloat16RNE(curOutput[d]);
                }
                int d = st;
#ifdef __AVX2__
                const __m256 one = _mm256_set1_ps(1.0f);
                for (; d + 7 < end; d += 8) {
                    __m256 cur = _mm256_loadu_ps(curOutput + d);
                    __m256 sum = _mm256_loadu_ps(output + d);
                    sum = _mm256_add_ps(
                        sum, _mm256_mul_ps(cur, one));
                    _mm256_storeu_ps(output + d, sum);
                }
#endif
                for (; d < end; d++) {
                    output[d] += curOutput[d] * 1.0f;
                }
            }
        }
    };

    struct MultiThreadDeepSeekV4MoeTaskQueueOp : MultiThreadBaseOp {
        std::vector<MultiThreadBaseOp*> *tasks;
        std::atomic<int> *next;
        int firstTask;

        MultiThreadDeepSeekV4MoeTaskQueueOp(
            std::vector<MultiThreadBaseOp*> *tasks,
            std::atomic<int> *next, int firstTask
        ) : tasks(tasks), next(next), firstTask(firstTask) {}

        void Run() override {
            const int taskCount = (int)tasks->size();
            if (firstTask < taskCount) {
                (*tasks)[firstTask]->Run();
            }
            while (true) {
                int taskId =
                    next->fetch_add(1, std::memory_order_relaxed);
                if (taskId >= taskCount) {
                    break;
                }
                (*tasks)[taskId]->Run();
            }
        }
    };

    static void ScheduleDeepSeekV4MoeTasks(
        std::vector<MultiThreadBaseOp*> &tasks,
        bool deleteTasks = true
    ) {
        if (tasks.empty()) {
            return;
        }

        auto *pool = GetAlivePool();
        int workerCount = std::min(
            (int)pool->threads.size(), (int)tasks.size());
        std::atomic<int> next(workerCount);
        std::vector<MultiThreadDeepSeekV4MoeTaskQueueOp> workers;
        workers.reserve(workerCount);
        for (int i = 0; i < workerCount; i++) {
            workers.emplace_back(&tasks, &next, i);
        }
        for (int i = 0; i < workerCount; i++) {
            pool->PushOp(i, &workers[i]);
        }
        for (int i = 0; i < workerCount; i++) {
            pool->Wait(i);
        }
        if (deleteTasks) {
            for (auto *task : tasks) {
                delete task;
            }
        }
    }

#ifdef USE_CPU_NUMA
    // 纯 CPU 构建的 NUMA 支持：检测拓扑、把线程池线程绑到各 node 的 cpu，
    // 并记录 threadId -> node 映射，供 decode 转置 GEMV 做 per-node 调度。
    struct CpuNumaState {
        int numaCnt = 1;
        bool enabled = false;
        std::vector<int> nodeThreadStart;            // {nodeId -> 起始 pool threadId}
        std::vector<int> nodeThreadCount;            // {nodeId -> 线程数}
    };

    static CpuNumaState &GetCpuNumaState() {
        static CpuNumaState state;
        return state;
    }

    struct CpuBindCPUOp : MultiThreadBaseOp {
        int cpuId;
        CpuBindCPUOp(int cpuId) : cpuId(cpuId) {}
        void Run() override {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpuId, &set);
            sched_setaffinity(0, sizeof(cpu_set_t), &set);
        }
    };

    static void InitCpuNumaBinding() {
        static std::once_flag once;
        std::call_once(once, []() {
            CpuNumaState &s = GetCpuNumaState();
            if (numa_available() < 0) {
                return;
            }
            int nodes = numa_num_configured_nodes();
            if (nodes < 2) {
                return;
            }
            AliveThreadPool *pool = GetAlivePool();
            int threads = pool->curActivateThreadInterval.second;
            if (threads <= 0) {
                threads = (int)pool->threads.size();
            }
            if (threads < 2 * nodes) {
                return;
            }

            // 收集每个 node 的 cpu 列表
            std::vector<std::vector<int>> cpusPerNode(nodes);
            for (int n = 0; n < nodes; n++) {
                struct bitmask *mask = numa_allocate_cpumask();
                if (numa_node_to_cpus(n, mask) == 0) {
                    int cpuCnt = numa_num_configured_cpus();
                    for (int c = 0; c < cpuCnt; c++) {
                        if (numa_bitmask_isbitset(mask, c)) {
                            cpusPerNode[n].push_back(c);
                        }
                    }
                }
                numa_free_cpumask(mask);
            }

            int perNode = threads / nodes;
            int boundThreads = perNode * nodes;   // 余数线程不参与 node 分片

            s.numaCnt = nodes;
            s.nodeThreadStart.resize(nodes);
            s.nodeThreadCount.resize(nodes);

            for (int n = 0; n < nodes; n++) {
                int start = n * perNode;
                s.nodeThreadStart[n] = start;
                s.nodeThreadCount[n] = perNode;
                for (int j = 0; j < perNode; j++) {
                    int tid = start + j;
                    if (!cpusPerNode[n].empty()) {
                        int cpu = cpusPerNode[n][j % cpusPerNode[n].size()];
                        pool->PushOp(tid, new CpuBindCPUOp(cpu));
                    }
                }
            }
            for (int tid = 0; tid < boundThreads; tid++) {
                pool->Wait(tid);
            }
            s.enabled = true;
        });
    }

    // 在 [firstThread, firstThread + threadCount) 线程范围内做 work-stealing，
    // 复用 DynamicScheduleTasks 的 TaskState/WorkStealingOp 逻辑。
    static void DynamicScheduleTasksRange(std::vector<MultiThreadBaseOp*> &ops,
                                          int firstThread, int threadCount,
                                          AliveThreadPool *pool) {
        using TaskState = typename WorkStealingOp::TaskState;
        std::vector<TaskState*> taskStates(threadCount, nullptr);
        for (int i = 0; i < threadCount; i++) {
            taskStates[i] = new TaskState();
            taskStates[i]->curr.store(0, std::memory_order_relaxed);
            taskStates[i]->end = 0;
            taskStates[i]->completed.store(false, std::memory_order_relaxed);
        }

        int totalOps = (int)ops.size();
        if (totalOps > 0) {
            int tasksPerThread = totalOps / threadCount;
            int remainingTasks = totalOps % threadCount;
            int taskIndex = 0;
            for (int i = 0; i < threadCount; i++) {
                int numTasks = tasksPerThread + (i < remainingTasks ? 1 : 0);
                if (numTasks > 0) {
                    taskStates[i]->tasks.clear();
                    taskStates[i]->tasks.reserve(numTasks);
                    for (int j = 0; j < numTasks && taskIndex < totalOps; j++) {
                        taskStates[i]->tasks.push_back(ops[taskIndex++]);
                    }
                    taskStates[i]->curr.store(0, std::memory_order_relaxed);
                    taskStates[i]->end = (int)taskStates[i]->tasks.size();
                } else {
                    taskStates[i]->end = 0;
                }
            }
        }

        std::vector<WorkStealingOp*> wsOps(threadCount);
        for (int i = 0; i < threadCount; i++) {
            wsOps[i] = new WorkStealingOp(i, &taskStates, taskStates[i], threadCount);
            pool->PushOp(firstThread + i, wsOps[i]);
        }
        for (int i = 0; i < threadCount; i++) {
            pool->Wait(firstThread + i);
        }
        for (int i = 0; i < threadCount; i++) {
            delete wsOps[i];
            delete taskStates[i];
        }
        for (auto *op : ops) {
            delete op;
        }
    }
#endif

    struct DeepSeekV4MoeLinearTaskStorage {
        std::vector<MultiThreadLinearBFloat16FP8E4M3Op> fp8;
        std::vector<MultiThreadLinearBFloat16NVFP4Op> nvfp4;
        std::vector<MultiThreadLinearBFloat16BFloat16Op> bf16;

        void BuildPointers(std::vector<MultiThreadBaseOp*> &tasks) {
            tasks.clear();
            tasks.reserve(fp8.size() + nvfp4.size() + bf16.size());
            // FP8 rows carry roughly twice as many weight bytes as compact
            // NVFP4 rows. Queue the heavier shared-expert work first so it
            // cannot become a half-populated final wave.
            for (auto &task : fp8) {
                tasks.push_back(&task);
            }
            for (auto &task : nvfp4) {
                tasks.push_back(&task);
            }
            for (auto &task : bf16) {
                tasks.push_back(&task);
            }
        }
    };

    static void AppendDeepSeekV4MoeLinearTasks(
        DeepSeekV4MoeLinearTaskStorage &tasks, uint16_t *linearInput,
        Data &weight, float *linearOutput, int inputColumns, int outputRows,
        int rowsPerTask
    ) {
        for (int st = 0; st < outputRows; st += rowsPerTask) {
            int end = std::min(st + rowsPerTask, outputRows);
            if (weight.dataType == DataType::FP8_E4M3) {
                tasks.fp8.emplace_back(
                    linearInput, weight.cpuData, nullptr, linearOutput,
                    1, inputColumns, outputRows, st, end,
                    weight.scales.data(), weight.blockK, weight.blockM);
            } else if (weight.dataType == DataType::NVFP4) {
                float *scaleFloats =
                    weight.scales.empty() ? nullptr : weight.scales.data();
                tasks.nvfp4.emplace_back(
                    linearInput, weight.cpuData, nullptr, linearOutput,
                    1, inputColumns, outputRows, st, end,
                    scaleFloats, GetNVFP4ScaleData(weight),
                    weight.blockK, weight.blockM);
            } else if (weight.dataType == DataType::BFLOAT16) {
                tasks.bf16.emplace_back(
                    linearInput, (uint16_t*)weight.cpuData, nullptr,
                    linearOutput, 1, inputColumns, outputRows, st, end);
            }
        }
    }

    static void ReserveDeepSeekV4MoeLinearTasks(
        DeepSeekV4MoeLinearTaskStorage &tasks,
        const std::vector<std::pair<int, float>> &experts,
        Data **weights, int weightOffset, int rowsPerTask
    ) {
        size_t fp8Count = 0, nvfp4Count = 0, bf16Count = 0;
        for (const auto &expert : experts) {
            Data &weight = *weights[expert.first * 2 + weightOffset];
            size_t taskCount =
                (weight.dims[0] + rowsPerTask - 1) / rowsPerTask;
            if (weight.dataType == DataType::FP8_E4M3) {
                fp8Count += taskCount;
            } else if (weight.dataType == DataType::NVFP4) {
                nvfp4Count += taskCount;
            } else if (weight.dataType == DataType::BFLOAT16) {
                bf16Count += taskCount;
            }
        }
        tasks.fp8.reserve(fp8Count);
        tasks.nvfp4.reserve(nvfp4Count);
        tasks.bf16.reserve(bf16Count);
    }

    static uint64_t GetConvertedBufferBytes(const Data &data) {
        uint64_t elementCount = data.expansionSize > 0 ? data.expansionSize : data.Count(0);
        return (elementCount * data.unitSize - 1) / data.unitSizeDiv + 1;
    }

    static void InvalidateCudaMirror(Data &data) {
#ifdef USE_CUDA
        if (data.cudaData != nullptr) {
            FastllmCudaFree(data.cudaData);
            data.cudaData = nullptr;
        }
#endif
    }

    CpuDevice::CpuDevice() {
        this->deviceType = "cpu";
        this->ops["ToFloat16"] = (BaseOperator*)(new CpuToFloat16());
        this->ops["ToFloat32"] = (BaseOperator*)(new CpuToFloat32());
        this->ops["ConvertToFloat16"] = (BaseOperator*)(new CpuConvertToFloat16());
        this->ops["ConvertToFloat32"] = (BaseOperator*)(new CpuConvertToFloat32());
        this->ops["ToBFloat16"] = (BaseOperator*)(new CpuToBFloat16());
        this->ops["ConvertToBFloat16"] = (BaseOperator*)(new CpuConvertToBFloat16());

        this->ops["Attention"] = (BaseOperator*)(new CpuAttention());
        this->ops["MergeMOE"] = (BaseOperator*)(new CpuMergeMOE());
        this->ops["MergeMLA"] = (BaseOperator*)(new CpuMergeMLA());
        this->ops["MergeMLAPaged"] = (BaseOperator*)(new CpuMergeMLAPaged());
        this->ops["CopyKVCache"] = (BaseOperator*)(new CpuCopyKVCacheOp());
        this->ops["Embedding"] = (BaseOperator*)(new CpuEmbedding());
        this->ops["EmbeddingDirect"] = (BaseOperator*)(new CpuEmbeddingDirect());
        this->ops["LayerNorm"] = (BaseOperator*)(new CpuLayerNormOp());
        this->ops["RMSNorm"] = (BaseOperator*)(new CpuRMSNormOp());
        this->ops["RMSNormPart"] = (BaseOperator*)(new CpuRMSNormPartOp());
        this->ops["KimiK3RMSNorm"] = (BaseOperator*)(new CpuKimiK3RMSNormOp());
        this->ops["KimiK3CausalConv1D"] = (BaseOperator*)(new CpuKimiK3CausalConv1DOp());
        this->ops["KimiK3UpdatePackedConvCache"] =
            (BaseOperator*)(new CpuKimiK3UpdatePackedConvCacheOp());
        this->ops["KimiK3L2Norm"] = (BaseOperator*)(new CpuKimiK3L2NormOp());
        this->ops["KimiK3RecurrentKDA"] = (BaseOperator*)(new CpuKimiK3RecurrentKDAOp());
        this->ops["KimiK3RMSNormSigmoidGate"] = (BaseOperator*)(new CpuKimiK3RMSNormSigmoidGateOp());
        this->ops["KimiK3AttnRes"] = (BaseOperator*)(new CpuKimiK3AttnResOp());
        this->ops["KimiK3SiTUAndMul"] = (BaseOperator*)(new CpuKimiK3SiTUAndMulOp());
        this->ops["KimiK3RoutedExperts"] = (BaseOperator*)(new CpuKimiK3RoutedExpertsOp());
        this->ops["KimiK3CausalAttention"] = (BaseOperator*)(new CpuKimiK3CausalAttentionOp());
        this->ops["Linear"] = (BaseOperator*)(new CpuLinearOp());
        this->ops["Conv1DPerChannel"] = (BaseOperator*)(new CpuConv1DPerChannel());
        this->ops["Conv2D"] = (BaseOperator*)(new CpuConv2DOp());
        this->ops["Split"] = (BaseOperator*)(new CpuSplitOp());
        this->ops["Repeat"] = (BaseOperator*)(new CpuRepeatOp());
        this->ops["Copy"] = (BaseOperator*)(new CpuCopyOp());
        this->ops["DeepSeekV4HcPre"] = (BaseOperator*)(new CpuDeepSeekV4HcPreOp());
        this->ops["DeepSeekV4HcPost"] = (BaseOperator*)(new CpuDeepSeekV4HcPostOp());
        this->ops["ScaleQRatory"] = (BaseOperator*)(new CpuScaleQRatoryOp());
        this->ops["DeepSeekV4RotaryQuant"] = (BaseOperator*)(new CpuDeepSeekV4RotaryQuantOp());
        this->ops["DeepSeekV4SparseAttention"] = (BaseOperator*)(new CpuDeepSeekV4SparseAttentionOp());
        this->ops["DeepSeekV4SparseAttentionDecodeCached"] =
            (BaseOperator*)(new CpuDeepSeekV4SparseAttentionDecodeCachedOp());
        this->ops["DeepSeekV4IndexerTopK"] = (BaseOperator*)(new CpuDeepSeekV4IndexerTopKOp());
        this->ops["DeepSeekV4WoA"] = (BaseOperator*)(new CpuDeepSeekV4WoAOp());
        this->ops["DeepSeekV4BuildCompressedKVFromRaw"] = (BaseOperator*)(new CpuDeepSeekV4BuildCompressedKVFromRawOp());
        this->ops["DeepSeekV4StoreWindowKVCache"] = (BaseOperator*)(new CpuDeepSeekV4StoreWindowKVCacheOp());
        this->ops["DeepSeekV4UpdateWindowKVCache"] = (BaseOperator*)(new CpuDeepSeekV4UpdateWindowKVCacheOp());
        this->ops["Cat"] = (BaseOperator*)(new CpuCatOp());
        this->ops["Pad"] = (BaseOperator*)(new CpuPadOp());
        this->ops["CatDirect"] = (BaseOperator*)(new CpuCatDirectOp());
        this->ops["MatMul"] = (BaseOperator*)(new CpuMatMulOp());
        this->ops["MatMulTransB"] = (BaseOperator*)(new CpuMatMulTransBOp());
        this->ops["SoftMax"] = (BaseOperator*)(new CpuSoftMaxOp());
        this->ops["Normalize"] = (BaseOperator*)(new CpuNormalizeOp());
        this->ops["Silu"] = (BaseOperator*)(new CpuSiluOp());
        this->ops["TanH"] = (BaseOperator*)(new CpuTanHOp());
        this->ops["Relu"] = (BaseOperator*)(new CpuReluOp());
        this->ops["Exp"] = (BaseOperator*)(new CpuExpOp());
        this->ops["Sigmoid"] = (BaseOperator*)(new CpuSigmoidOp());
        this->ops["Gelu"] = (BaseOperator*)(new CpuGeluOp());
        this->ops["GeluNew"] = (BaseOperator*)(new CpuGeluNewOp());
        this->ops["Geglu"] = (BaseOperator*)(new CpuGegluOp());
        this->ops["Swiglu"] = (BaseOperator*)(new CpuSwigluOp());
        this->ops["CrossSwiglu"] = (BaseOperator*)(new CpuCrossSwigluOp());
        this->ops["SwigluGptOss"] = (BaseOperator*)(new CpuSwigluGptOssOp());
        this->ops["MambaSoftplus"] = (BaseOperator*)(new CpuMambaSoftplusOp());
        this->ops["Mul"] = (BaseOperator*)(new CpuMulOp());
        this->ops["MulTo"] = (BaseOperator*)(new CpuMulToOp());
        this->ops["Add"] = (BaseOperator*)(new CpuAddOp());
        this->ops["AddTo"] = (BaseOperator*)(new CpuAddToOp());
        this->ops["AttentionMask"] = (BaseOperator*)(new CpuAttentionMaskOp());
        this->ops["AttentionExtendedMask"] = (BaseOperator*)(new CpuAttentionExtendedMaskOp());
        this->ops["AlibiMask"] = (BaseOperator*)(new CpuAlibiMaskOp());
        this->ops["TransferAttn"] = (BaseOperator*)(new CpuTransferAttnOp());
        this->ops["ApplyChunkDecayByLastLogG"] = (BaseOperator*)(new CpuApplyChunkDecayByLastLogGOp());
        this->ops["RecurrentGatedDeltaRule"] = (BaseOperator*)(new CpuRecurrentGatedDeltaRuleOp());
        this->ops["CausalMask"] = (BaseOperator*)(new CpuCausalMaskOp());
        this->ops["TopK"] = (BaseOperator*)(new CpuTopKOp());
        this->ops["SelectExpert"] = (BaseOperator*)(new CpuSelectExpertOp());
        this->ops["Permute"] = (BaseOperator*)(new CpuPermuteOp());
        this->ops["PermuteSelf"] = (BaseOperator*)(new CpuPermuteSelfOp());
        this->ops["RotatePosition2D"] = (BaseOperator*)(new CpuRotatePosition2DOp());
        this->ops["NearlyRotatePosition2D"] = (BaseOperator*)(new CpuNearlyRotatePosition2DOp());
        this->ops["LlamaRotatePosition2D"] = (BaseOperator*)(new CpuLlamaRotatePosition2DOp());
        this->ops["LlamaRotatePosition2DPart"] = (BaseOperator*)(new CpuLlamaRotatePosition2DPartOp());
        this->ops["RopeEncoding"] = (BaseOperator*)(new CpuRopeEncodingOp());
        this->ops["Llama3RopeEncoding"] = (BaseOperator*)(new CpuLlama3RopeEncodingOp());
        this->ops["YarnRopeEncoding"] = (BaseOperator*)(new CpuYarnRopeEncodingOp());
        this->ops["Qwen35InterleavedRope"] = (BaseOperator*)(new CpuQwen35InterleavedRopeOp());
        this->ops["QKVRMSNormRope"] = (BaseOperator*)(new CpuQKVRMSNormRopeOp());
        this->ops["QKVRMSNormRopeSplitAppendPagedCache"] = (BaseOperator*)(new CpuQKVRMSNormRopeSplitAppendPagedCacheOp());
        this->ops["RepeatPenalty"] = (BaseOperator*)(new CpuRepeatPenaltyOp());
        this->ops["ApplyLognAttn"] = (BaseOperator*)(new CpuApplyLognAttnOp());
        this->ops["CumSumLastDim"] = (BaseOperator*)(new CpuCumSumLastDimOp());
        this->ops["MakeDecayMask"] = (BaseOperator*)(new CpuMakeDecayMaskOp());

        this->ops["SplitBatch"] = (BaseOperator*)(new CpuSplitBatchOp());
        this->ops["CatBatch"] = (BaseOperator*)(new CpuCatBatchOp());
        this->ops["MulBatch"] = (BaseOperator*)(new CpuMulBatchOp());
        this->ops["MatMulBatch"] = (BaseOperator*)(new CpuMatMulBatchOp());
        this->ops["MatMulTransBBatch"] = (BaseOperator*)(new CpuMatMulTransBBatchOp());
        this->ops["SoftMaxBatch"] = (BaseOperator*)(new CpuSoftmaxBatchOp());
        this->ops["CatDirectBatch"] = (BaseOperator*)(new CpuCatDirectBatchOp());
        this->ops["AppendKVCachebatch"] = (BaseOperator*)(new CpuAppendKVCacheBatchOp());
        this->ops["AttentionBatch"] = (BaseOperator*)(new CpuAttentionBatchOp());

        this->ops["AttentionPaged"] = (BaseOperator*)(new CpuAttentionPagedOp());
        this->ops["AttentionPagedBatch"] = (BaseOperator*)(new CpuAttentionPagedBatchOp());
        this->ops["GeneratePagedBatchParams"] = (BaseOperator*)(new CpuGeneratePagedBatchParamsOp());
        this->ops["GenerateAppendPagedCacheBatchParams"] = (BaseOperator*)(new CpuGenerateAppendPagedCacheBatchParamsOp());
        this->ops["AppendPagedCache"] = (BaseOperator*)(new CpuAppendPagedCacheOp());
        this->ops["AppendPagedCacheBatch"] = (BaseOperator*)(new CpuAppendPagedCacheBatchOp());
    }

    bool CpuDevice::Malloc(void **ret, size_t size) {
        *ret = (void*)new uint8_t [size];
        return true;
    }

    bool CpuDevice::Free(void *ret) {
        delete[] (uint8_t*)ret;
        return true;
    }

    bool CpuDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        return true;
    }

    bool CpuDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        return true;
    }

    CPUInstructInfo cpuInstructInfo;

#ifdef __AVX2__
    extern int DotU4U8_AVX512VNNI(uint8_t *a, uint8_t *b, int n);

    int DotU4U8(uint8_t *a, uint8_t *b, int n) {
         if (cpuInstructInfo.hasAVX512VNNI && n % 64 == 0) {
            return DotU4U8_AVX512VNNI(a, b, n);
         }
        if (n % 32 != 0) {
            int ans = 0;
            int i = 0;
            for (; i + 1 < n; i += 2) {
                uint8_t packed = a[i / 2];
                ans += (packed >> 4) * b[i];
                ans += (packed & 0xF) * b[i + 1];
            }
            if (i < n) {
                ans += (a[i / 2] >> 4) * b[i];
            }
            return ans;
        }
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        const __m256i lowMask = _mm256_set1_epi8(0xf);
        const __m256i ones = _mm256_set1_epi16(1);
        for (; i + 31 < n; i += 32) {
            __m128i orix = _mm_loadu_si128((const __m128i *) (a + i / 2));
            __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
            __m256i bx = _mm256_and_si256(lowMask, bytex);
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(by, bx), ones));
        }
        return I32sum(acc);
    };
#endif

    FP16ToFP32Manager fp16tofp32;
    BF16ToFP32Manager bf16tofp32;
    FP8E4M3ToFP32Manager fp8e4m3tofp32;
    extern BF16ToFP16Manager bf16tofp16;
    FP16ToBF16Manager fp16tobf16;

    static inline float NVFP4E2M1ToFloat(uint8_t v) {
        static const float table[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
        };
        return table[v & 0xF];
    }

    static inline uint16_t FloatToBFloat16Trunc(float v) {
        uint32_t bits;
        memcpy(&bits, &v, sizeof(bits));
        return (uint16_t)(bits >> 16);
    }

#ifdef __AVX2__
    static inline __m256 _mm256_nvfp4_to_fp32_ps(const uint8_t *packed) {
        uint32_t raw;
        memcpy(&raw, packed, sizeof(raw));
        __m128i bytes = _mm_cvtsi32_si128(static_cast<int>(raw));
        const __m128i lowMask = _mm_set1_epi8(0x0F);
        __m128i low = _mm_and_si128(bytes, lowMask);
        __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), lowMask);
        __m128i interleaved = _mm_unpacklo_epi8(low, high);

        __m256i fp4 = _mm256_cvtepu8_epi32(interleaved);
        __m256i sign = _mm256_slli_epi32(_mm256_and_si256(fp4, _mm256_set1_epi32(0x8)), 28);
        __m256i body = _mm256_and_si256(fp4, _mm256_set1_epi32(0x7));

        __m256i exp = _mm256_slli_epi32(_mm256_add_epi32(_mm256_srli_epi32(body, 1), _mm256_set1_epi32(126)), 23);
        __m256i mant = _mm256_slli_epi32(_mm256_and_si256(body, _mm256_set1_epi32(1)), 22);
        mant = _mm256_andnot_si256(_mm256_cmpeq_epi32(body, _mm256_set1_epi32(1)), mant);

        __m256i bits = _mm256_or_si256(sign, _mm256_or_si256(exp, mant));
        bits = _mm256_andnot_si256(_mm256_cmpeq_epi32(body, _mm256_setzero_si256()), bits);
        return _mm256_castsi256_ps(bits);
    }

    static inline __m128i _mm256_float_to_bf16_trunc(__m256 value) {
        __m256i bits = _mm256_srli_epi32(_mm256_castps_si256(value), 16);
        __m128i lo = _mm256_castsi256_si128(bits);
        __m128i hi = _mm256_extracti128_si256(bits, 1);
        return _mm_packus_epi32(lo, hi);
    }

    static inline void NVFP4Block16ToBFloat16_AVX2(const uint8_t *blockStart, uint16_t *dst, float scale, int blockElems) {
        __m256 scaleVec = _mm256_set1_ps(scale);
        int l = 0;
        if (blockElems >= 8) {
            __m256 values = _mm256_mul_ps(_mm256_nvfp4_to_fp32_ps(blockStart), scaleVec);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_float_to_bf16_trunc(values));
            l = 8;
        }
        if (blockElems >= 16) {
            __m256 values = _mm256_mul_ps(_mm256_nvfp4_to_fp32_ps(blockStart + 4), scaleVec);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8), _mm256_float_to_bf16_trunc(values));
            l = 16;
        }
        for (; l < blockElems; l++) {
            uint8_t packed = blockStart[l >> 1];
            uint8_t fp4 = (l & 1) ? (packed >> 4) : (packed & 0xF);
            dst[l] = FloatToBFloat16Trunc(scale * NVFP4E2M1ToFloat(fp4));
        }
    }
#endif

    static void NVFP4Block16RowsToBFloat16(
        const void *B, long ldb, uint16_t *bf16B, int m, int st, int end, bool scaleE8M0 = false
    ) {
        const int blockSize = 16;
        const int packedBlockBytes = 8 + (scaleE8M0 ? (int)sizeof(uint8_t) : (int)sizeof(float));
        const int blocks = (m - 1) / blockSize + 1;
        for (int row = st; row < end; row++) {
            const uint8_t *rowStart = (const uint8_t*)B + (size_t)row * ldb;
            uint16_t *dstRow = bf16B + (size_t)(row - st) * m;
            for (int block = 0; block < blocks; block++) {
                const uint8_t *blockStart = rowStart + block * packedBlockBytes;
                float scale = scaleE8M0 ? NVFP4E8M0ScaleToFloat(blockStart[8]) : 0.0f;
                if (!scaleE8M0) {
                    memcpy(&scale, blockStart + 8, sizeof(float));
                }
                int l = block * blockSize;
                int blockEnd = std::min(m, l + blockSize);
#ifdef __AVX2__
                if (cpuInstructInfo.hasAVX2) {
                    NVFP4Block16ToBFloat16_AVX2(blockStart, dstRow + l, scale, blockEnd - l);
                    continue;
                }
#endif
                for (; l < blockEnd; l++) {
                    int offset = l - block * blockSize;
                    uint8_t packed = blockStart[offset >> 1];
                    uint8_t fp4 = (offset & 1) ? (packed >> 4) : (packed & 0xF);
                    dstRow[l] = FloatToBFloat16Trunc(scale * NVFP4E2M1ToFloat(fp4));
                }
            }
        }
    }

    static void NVFP4Block32RowsToBFloat16(
        const void *B, long ldb, uint16_t *bf16B,
        int m, int st, int end
    ) {
        const int blockSize = 32;
        const int packedBlockBytes = 16 + (int)sizeof(uint8_t);
        const int blocks = (m - 1) / blockSize + 1;
        for (int row = st; row < end; row++) {
            const uint8_t *rowStart =
                (const uint8_t*)B + (size_t)row * ldb;
            uint16_t *dstRow = bf16B + (size_t)(row - st) * m;
            for (int block = 0; block < blocks; block++) {
                const uint8_t *blockStart =
                    rowStart + block * packedBlockBytes;
                const float scale =
                    NVFP4E8M0ScaleToFloat(blockStart[16]);
                const int begin = block * blockSize;
                const int blockElems = std::min(blockSize, m - begin);
#ifdef __AVX2__
                if (cpuInstructInfo.hasAVX2) {
                    const int first = std::min(16, blockElems);
                    NVFP4Block16ToBFloat16_AVX2(
                        blockStart, dstRow + begin, scale, first);
                    if (blockElems > 16) {
                        NVFP4Block16ToBFloat16_AVX2(
                            blockStart + 8, dstRow + begin + 16,
                            scale, blockElems - 16);
                    }
                    continue;
                }
#endif
                for (int offset = 0; offset < blockElems; offset++) {
                    const uint8_t packed = blockStart[offset >> 1];
                    const uint8_t fp4 = (offset & 1) ?
                        (packed >> 4) : (packed & 0xF);
                    dstRow[begin + offset] = FloatToBFloat16Trunc(
                        scale * NVFP4E2M1ToFloat(fp4));
                }
            }
        }
    }

    template <int COLS, bool INPUT_BF16, bool SCALE_E8M0>
    static inline void GemmNVFP4Block16Cols_CPU(
        const void *inputBase, const uint8_t *weightBase, long ldb, float *output,
        int outputCol, int m
    ) {
        const float *inputF32 = reinterpret_cast<const float*>(inputBase);
        const uint16_t *inputBF16 = reinterpret_cast<const uint16_t*>(inputBase);
        const int blocks = (m - 1) / 16 + 1;
        float now[COLS] = {};

        const int packedBlockBytes = 8 + (SCALE_E8M0 ? (int)sizeof(uint8_t) : (int)sizeof(float));
        for (int block = 0; block < blocks; block++) {
            const uint8_t *blockStart[COLS];
            float scale[COLS];
            for (int c = 0; c < COLS; c++) {
                blockStart[c] = weightBase + (size_t)(outputCol + c) * ldb + block * packedBlockBytes;
                if constexpr (SCALE_E8M0) {
                    scale[c] = NVFP4E8M0ScaleToFloat(blockStart[c][8]);
                } else {
                    memcpy(scale + c, blockStart[c] + 8, sizeof(float));
                }
            }

            int l = block * 16;
            int blockEnd = std::min(m, l + 16);
#ifdef __AVX2__
            __m256 vsum[COLS];
            for (int c = 0; c < COLS; c++) {
                vsum[c] = _mm256_setzero_ps();
            }
            for (; l + 7 < blockEnd; l += 8) {
                __m256 vi;
                if constexpr (INPUT_BF16) {
                    __m128i bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(inputBF16 + l));
                    __m256i bf16_32 = _mm256_cvtepu16_epi32(bf16);
                    vi = _mm256_castsi256_ps(_mm256_slli_epi32(bf16_32, 16));
                } else {
                    vi = _mm256_loadu_ps(inputF32 + l);
                }
                int offset = (l - block * 16) >> 1;
                for (int c = 0; c < COLS; c++) {
                    __m256 vw = _mm256_nvfp4_to_fp32_ps(blockStart[c] + offset);
                    vsum[c] = _mm256_fmadd_ps(vi, vw, vsum[c]);
                }
            }
            for (int c = 0; c < COLS; c++) {
                now[c] += Floatsum(vsum[c]) * scale[c];
            }
#endif
            for (; l < blockEnd; l++) {
                float x = INPUT_BF16 ? bf16tofp32.dict[inputBF16[l]] : inputF32[l];
                int offset = l - block * 16;
                uint8_t shift = (offset & 1) ? 4 : 0;
                for (int c = 0; c < COLS; c++) {
                    uint8_t packed = blockStart[c][offset >> 1];
                    now[c] += scale[c] * x * NVFP4E2M1ToFloat((packed >> shift) & 0xF);
                }
            }
        }

        for (int c = 0; c < COLS; c++) {
            output[c] = now[c];
        }
    }

    template <bool INPUT_BF16, bool SCALE_E8M0 = false>
    static inline void GemmNVFP4Block16_CPU_Run(
        const void *A, long lda, const void *B, long ldb, void *C, long ldc,
        int n, int m, int st, int end
    ) {
        const uint8_t *aBytes = reinterpret_cast<const uint8_t*>(A);
        const uint8_t *weightBase = reinterpret_cast<const uint8_t*>(B);
        for (int i = 0; i < n; i++) {
            const void *input = aBytes + (size_t)i * lda;
            float *output = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(C) + (size_t)i * ldc);
            int j = st;
            for (; j + 3 < end; j += 4) {
                GemmNVFP4Block16Cols_CPU<4, INPUT_BF16, SCALE_E8M0>(input, weightBase, ldb, output + j, j, m);
            }
            switch (end - j) {
                case 0: break;
                case 1: GemmNVFP4Block16Cols_CPU<1, INPUT_BF16, SCALE_E8M0>(input, weightBase, ldb, output + j, j, m); break;
                case 2: GemmNVFP4Block16Cols_CPU<2, INPUT_BF16, SCALE_E8M0>(input, weightBase, ldb, output + j, j, m); break;
                case 3: GemmNVFP4Block16Cols_CPU<3, INPUT_BF16, SCALE_E8M0>(input, weightBase, ldb, output + j, j, m); break;
            }
        }
    }

    void Float16ToFloat32(uint16_t *float16, float *float32, int len) {
        int i = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        for (; i + 7 < len; i += 8) {
            float16x8_t input_vec = vld1q_f16((float16_t*)float16 + i);
            float32x4_t output_vec1 = vcvt_f32_f16(vget_low_f16(input_vec));
            float32x4_t output_vec2 = vcvt_f32_f16(vget_high_f16(input_vec));
            vst1q_f32(float32 + i, output_vec1);
            vst1q_f32(float32 + i + 4, output_vec2);
        }
#endif
#ifdef __AVX2__
        for (; i + 7 < len; i += 8) {
            __m128i f16 = _mm_loadu_si128((const __m128i*)(float16 + i));
            _mm256_storeu_ps(float32 + i, _mm256_cvtph_ps(f16));
        }
#endif
        for (; i < len; i++) {
            float32[i] = fp16tofp32.dict[float16[i]];
        }
    }

    void Float32ToFloat16(float *float32, uint16_t *float16, int len) {
        int i = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        for (; i + 3 < len; i += 4) {
            float32x4_t input_vec = vld1q_f32(float32 + i);
            float16x4_t output_vec = vcvt_f16_f32(input_vec);
            vst1_f16((float16_t*)float16 + i, output_vec);
        }
#endif
#ifdef __AVX__
        for (; i + 7 < len; i += 8) {
            __m256 input_vec = _mm256_loadu_ps(float32 + i);  // 加载 8 个 float32
            __m128i output_vec = _mm256_cvtps_ph(input_vec, _MM_FROUND_TO_NEAREST_INT);  // 转换为 8 个 float16
            _mm_storeu_si128((__m128i*)(float16 + i), output_vec);  // 存储 8 个 float16
        }
#endif
        for (; i < len; i++) {
            float16[i] = float_to_half(float32[i]);
        }
    }

    void Float32ToBFloat16(float *float32, uint16_t *bfloat16, int len) {
        int i = 0;
        
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        for (; i + 3 < len; i += 4) {
            // Load 4 float32 values
            float32x4_t f32x4 = vld1q_f32(&float32[i]);
            
            // Reinterpret as uint32 to access bits
            uint32x4_t u32x4 = vreinterpretq_u32_f32(f32x4);
            
            // IEEE bfloat16 conversion uses round-to-nearest-even.  A plain
            // shift truncates half of the values differently from PyTorch and
            // makes every BF16 boundary in a model observably inaccurate.
            uint32x4_t lsb = vandq_u32(vshrq_n_u32(u32x4, 16), vdupq_n_u32(1));
            uint32x4_t rounded = vaddq_u32(u32x4, vaddq_u32(vdupq_n_u32(0x7FFFu), lsb));
            uint32x4_t shifted = vshrq_n_u32(rounded, 16);
            
            // Narrow to 16-bit (takes bottom 16 bits from each 32-bit element)
            uint16x4_t bf16x4 = vmovn_u32(shifted);
            
            // Store 4 bfloat16 values
            vst1_u16(&bfloat16[i], bf16x4);
        }
#endif
    
#ifdef __AVX__
        for (; i + 7 < len; i += 8) {
            __m256i float_vec = _mm256_loadu_si256((__m256i*)&float32[i]);
            __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(float_vec, 16),
                                           _mm256_set1_epi32(1));
            __m256i rounding = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb);
            __m256i shifted = _mm256_srli_epi32(_mm256_add_epi32(float_vec, rounding), 16);
            __m128i lo = _mm256_castsi256_si128(shifted);
            __m128i hi = _mm256_extracti128_si256(shifted, 1);
            __m128i packed = _mm_packus_epi32(lo, hi);
            _mm_storeu_si128((__m128i*)&bfloat16[i], packed);
        }
#endif
        // 标量处理剩余元素
        for (; i < len; i++) {
            uint32_t val;
            memcpy(&val, &float32[i], sizeof(val));
            val += 0x7FFFu + ((val >> 16) & 1u);
            bfloat16[i] = (uint16_t)(val >> 16);
        }
    }

    void Float16ToBFloat16(uint16_t *float16, uint16_t *bfloat16, int len) {
        int i = 0;
#ifdef __AVX2__
        // f16 -> f32 (exact) -> bf16: truncate low 16 bits, matching fp16tobf16.dict
        for (; i + 7 < len; i += 8) {
            __m128i f16 = _mm_loadu_si128((const __m128i*)(float16 + i));
            __m256 f32 = _mm256_cvtph_ps(f16);
            __m256i bf = _mm256_srli_epi32(_mm256_castps_si256(f32), 16);
            __m128i lo = _mm256_castsi256_si128(bf);
            __m128i hi = _mm256_extracti128_si256(bf, 1);
            _mm_storeu_si128((__m128i*)(bfloat16 + i), _mm_packus_epi32(lo, hi));
        }
#endif
        for (; i < len; i++) {
            bfloat16[i] = fp16tobf16.dict[float16[i]];
        }
    }

    void BFloat16ToFloat16(uint16_t *bfloat16, uint16_t *float16, int len) {
        for (int i = 0; i < len; i++) {
            float16[i] = bf16tofp16.dict[bfloat16[i]];
        }
    }

    void BFloat16ToFloat32(uint16_t *bfloat16, float *float32, int len) {
        int i = 0;
#ifdef __AVX2__
        for (; i + 7 < len; i += 8) {
            __m128i b16 = _mm_loadu_si128((const __m128i*)(bfloat16 + i));
            __m256i x = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b16), 16);
            _mm256_storeu_si256((__m256i*)(float32 + i), x);
        }
#endif
        for (; i < len; i++) {
            uint32_t x = (uint32_t)bfloat16[i] << 16;
            memcpy(&float32[i], &x, sizeof(float));
        }
    }

    void CpuToFloat16::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        if (data.dataType == DataType::FLOAT16) {
            return;
        }
        if (data.dims.size() == 0) {
            data.dataType = DataType::FLOAT16;
            data.UpdateUnitSize();
            return;
        }
        if (data.dataType == DataType::FLOAT32) {
            float *old = (float*)data.cpuData;
            data.dataType = DataType::FLOAT16;
            data.UpdateUnitSize();
            uint64_t newBytes = GetConvertedBufferBytes(data);
            data.cpuData = new uint8_t[newBytes];
            uint16_t *cur = (uint16_t*)data.cpuData;
            int len = data.Count(0);
            for (int i = 0; i < len; i++) {
                cur[i] = float_to_half(old[i]);
            }
            delete[] old;
        } else if (data.dataType == DataType::BFLOAT16) {
            data.dataType = DataType::FLOAT16;
            data.UpdateUnitSize();
            int len = data.Count(0);
            BFloat16ToFloat16((uint16_t*)data.cpuData, (uint16_t*)data.cpuData, len);
        } else {
            ErrorInFastLLM("ToFloat16: unsupport dataType.\n");
        }
        data.expansionBytes = GetConvertedBufferBytes(data);
        InvalidateCudaMirror(data);
    }

    void CpuToFloat32::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        if (data.dataType == DataType::FLOAT32) {
            return;
        }
        if (data.dims.size() == 0) {
            data.dataType = DataType::FLOAT32;
            data.UpdateUnitSize();
            return;
        }
        if (data.dataType == DataType::FLOAT16) {
            uint16_t *old = (uint16_t*)data.cpuData;
            data.dataType = DataType::FLOAT32;
            data.UpdateUnitSize();
            uint64_t newBytes = GetConvertedBufferBytes(data);
            data.cpuData = new uint8_t[newBytes];
            float *cur = (float*)data.cpuData;
            int len = data.Count(0);
            for (int i = 0; i < len; i++) {
                cur[i] = fp16tofp32.dict[old[i]];
            }
            delete[] old;
        } else if (data.dataType == DataType::BFLOAT16) {
            uint16_t *old = (uint16_t*)data.cpuData;
            data.dataType = DataType::FLOAT32;
            data.UpdateUnitSize();
            uint64_t newBytes = GetConvertedBufferBytes(data);
            data.cpuData = new uint8_t[newBytes];
            float *cur = (float*)data.cpuData;
            int len = data.Count(0);
            for (int i = 0; i < len; i++) {
                uint32_t x = (uint32_t)old[i] << 16;
                cur[i] = *(float*)&x;
            }
            delete[] old;
        } else {
            ErrorInFastLLM("ToFloat32: unsupport dataType.\n");
        }
        data.expansionBytes = GetConvertedBufferBytes(data);
        InvalidateCudaMirror(data);
    }

    void CpuConvertToFloat16::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = (datas.find("input")->second);
        Data *output = (datas.find("output")->second);
        output->dataType = DataType::FLOAT16;
        output->Resize(input->dims);
        output->UpdateUnitSize();
        if (input->expansionDims.size() != 0)
            output->Expansion(input->expansionDims);
    }

    void CpuConvertToFloat16::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        if (input.dataType == DataType::FLOAT16) {
            memcpy(output.cpuData, input.cpuData, input.GetBytes());
            return;
        }
        if (input.dataType == DataType::FLOAT32) {
            Float32ToFloat16((float*)input.cpuData, (uint16_t*)output.cpuData, input.Count(0));
        } else if (input.dataType == DataType::BFLOAT16) {
            BFloat16ToFloat16((uint16_t*)input.cpuData, (uint16_t*)output.cpuData, input.Count(0));
        } else {
            ErrorInFastLLM("ToFloat16: unsupport dataType.\n");
        }
    }

    void CpuConvertToFloat32::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = (datas.find("input")->second);
        Data *output = (datas.find("output")->second);
        output->dataType = DataType::FLOAT32;
        output->Resize(input->dims);
        output->UpdateUnitSize();
        if (input->expansionDims.size() != 0)
            output->Expansion(input->expansionDims);
    }

    void CpuConvertToFloat32::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        if (input.dataType == DataType::FLOAT32) {
            memcpy(output.cpuData, input.cpuData, input.GetBytes());
            return;
        }
        if (input.dataType == DataType::FLOAT16) {
            Float16ToFloat32((uint16_t*)input.cpuData, (float*)output.cpuData, input.Count(0));
        } else if (input.dataType == DataType::BFLOAT16) {
            uint16_t *src = (uint16_t*)input.cpuData;
            float *dst = (float*)output.cpuData;
            int len = input.Count(0);
            for (int i = 0; i < len; i++) {
                uint32_t x = (uint32_t)src[i] << 16;
                dst[i] = *(float*)&x;
            }
        } else {
            ErrorInFastLLM("ToFloat32: unsupport dataType.\n");
        }
    }

    void CpuToBFloat16::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        if (data.dataType == DataType::BFLOAT16) {
            return;
        }
        if (data.dims.size() == 0) {
            data.dataType = DataType::BFLOAT16;
            data.UpdateUnitSize();
            return;
        }
        if (data.dataType == DataType::FLOAT32) {
            float *old = (float*)data.cpuData;
            data.dataType = DataType::BFLOAT16;
            data.UpdateUnitSize();
            uint64_t newBytes = GetConvertedBufferBytes(data);
            data.cpuData = new uint8_t[newBytes];
            uint16_t *cur = (uint16_t*)data.cpuData;
            int len = data.Count(0);
            Float32ToBFloat16(old, cur, len);
            delete[] old;
        } else if (data.dataType == DataType::FLOAT16) {
            data.dataType = DataType::BFLOAT16;
            data.UpdateUnitSize();
            int len = data.Count(0);
            Float16ToBFloat16((uint16_t*)data.cpuData, (uint16_t*)data.cpuData, len);
        } else {
            ErrorInFastLLM("ToBFloat16: unsupport dataType.\n");
        }
        data.expansionBytes = GetConvertedBufferBytes(data);
        InvalidateCudaMirror(data);
    }

    void CpuConvertToBFloat16::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = (datas.find("input")->second);
        Data *output = (datas.find("output")->second);
        output->dataType = DataType::BFLOAT16;
        output->Resize(input->dims);
        output->UpdateUnitSize();
        if (input->expansionDims.size() != 0)
            output->Expansion(input->expansionDims);
    }

    void CpuConvertToBFloat16::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        if (input.dataType == DataType::BFLOAT16) {
            memcpy(output.cpuData, input.cpuData, input.GetBytes());
            return;
        }
        if (input.dataType == DataType::FLOAT32) {
            Float32ToBFloat16((float*)input.cpuData, (uint16_t*)output.cpuData, input.Count(0));
        } else if (input.dataType == DataType::FLOAT16) {
            Float16ToBFloat16((uint16_t*)input.cpuData, (uint16_t*)output.cpuData, input.Count(0));
        } else {
            ErrorInFastLLM("ToBFloat16: unsupport dataType.\n");
        }
    }

    void CpuAttention::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];

        AssertInFastLLM(q.dims.size() == 3 && k.dims.size() == 3 && v.dims.size() == 3, "Attention: dims of q, k, v should be 3.\n");
        AssertInFastLLM(q.dims[2] == k.dims[2], "Attention: q.dims[2] should be equal to k.dims[2].\n");
        AssertInFastLLM(k.dims[1] == v.dims[1], "Attention: k.dims[1] should be equal to v.dims[1].\n");
        AssertInFastLLM(k.dims[0] == v.dims[0], "Attention: k.dims[0] should be equal to v.dims[0].\n");
        AssertInFastLLM(q.dims[0] == k.dims[0] * group, "Attention: q.dims[0] should be equal to k.dims[0] * group.\n");

        AssertInFastLLM(q.dataType == k.dataType && q.dataType == v.dataType,
                        "Attention: q, k, v's datatype should be same.\n");
        AssertInFastLLM(q.dataType == DataType::FLOAT32 ||
                        q.dataType == DataType::FLOAT16, 
                        "Attention's input's type should be float32.\n");

        std::vector <int> dims = {q.dims[0], q.dims[1], v.dims[2]};
        output.dataType = q.dataType;
        output.Resize(dims);
    }

    struct MultiThreadSingleAttentionOp : MultiThreadBaseOp {
        float *qd, *kd, *vd, *maskd, *od;
        float scale;
        int q1, q2, k1, v2;

        MultiThreadSingleAttentionOp(float *qd, float *kd, float *vd, float *maskd, float *od,
                         float scale, int q1, int q2, int k1, int v2) :
                         qd(qd), kd(kd), vd(vd), maskd(maskd), od(od), 
                         scale(scale), q1(q1), q2(q2), k1(k1), v2(v2) {}
        
        void Run() {
            float *qk = new float[k1];
            float *temp = new float[k1];
            int base = k1 - q1;
            for (int i = 0; i < q1; i++) {
                float maxValue = -10000, sum = 0.0;
                for (int j = 0; j < k1; j++) {
                    if (maskd && maskd[i * k1 + j] > 0.99) {
                        qk[j] = -10000;
                        continue;
                    }
                    if (!maskd && (base + i) < j) {
                        qk[j] = -10000;
                        continue;
                    }
                    float now = 0.0f;
                    int l = 0;
#ifdef __aarch64__
                    float32x4_t sum = {0, 0, 0, 0};
                    for (; l + 3 < q2; l += 4) {
                        sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(qd + i * q2 + l),
                                                    vld1q_f32(kd + j * q2 + l)));
                    }
                    now += sum[0] + sum[1] + sum[2] + sum[3];
#elif defined(__AVX__)
                    __m256 vsum = _mm256_set1_ps(0.0f);
                    for (; l + 7 < q2; l += 8) {
                        __m256 vx = _mm256_loadu_ps((const float *) (qd + i * q2 + l));
                        __m256 vy = _mm256_loadu_ps((const float *) (kd + j * q2 + l));
                        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
                    }
                    now += Floatsum(vsum);
#endif
                    for (; l < q2; l++) {
                        now += qd[i * q2 + l] * kd[j * q2 + l];
                    }
                    qk[j] = now * scale;
                    maxValue = std::max(maxValue, now * scale);
                }

                int j = 0;
#ifdef __aarch64__
                float32x4_t vmax = vdupq_n_f32(maxValue);
                for (; j + 3 < k1; j += 4) {
                    vst1q_f32(temp + j, exp_ps(vsubq_f32(vld1q_f32(qk + j), vmax)));
                }
#endif
                for (; j < k1; j++) {
                    temp[j] = expf(qk[j] - maxValue);
                }

                sum = 0.0f;
                for (int j = 0; j < k1; j++) {
                    sum += temp[j];
                }
                sum = std::max(sum, 0.1f);
                for (int j = 0; j < k1; j++) {
                    qk[j] = temp[j] / sum;
                }
                for (int j = 0; j < k1; j++) {
                    for (int l = 0; l < v2; l++) {
                        od[i * v2 + l] += qk[j] * vd[j * v2 + l];
                    }
                }
            }
            delete[] qk;
            delete[] temp;
        }
    };

    struct MultiThreadSingleAttentionFloat16Op : MultiThreadBaseOp {
        uint16_t *qd, *kd, *vd, *maskd, *od;
        float scale;
        int q1, q2, k1, v2;

        MultiThreadSingleAttentionFloat16Op(uint16_t *qd, uint16_t *kd, uint16_t *vd, uint16_t *maskd, uint16_t *od,
                         float scale, int q1, int q2, int k1, int v2) :
                         qd(qd), kd(kd), vd(vd), maskd(maskd), od(od), 
                         scale(scale), q1(q1), q2(q2), k1(k1), v2(v2) {}
        
        void Run() {
            std::vector <float> fqd, fkd, fvd, fmaskd, fod;
        
            fqd.resize(q1 * q2);
            fkd.resize(k1 * q2);
            fvd.resize(k1 * v2);
            fmaskd.resize(maskd ? q1 * k1 : 0);
            fod.resize(q1 * v2);

            Float16ToFloat32(qd, fqd.data(), (int)fqd.size());
            Float16ToFloat32(kd, fkd.data(), (int)fkd.size());
            Float16ToFloat32(vd, fvd.data(), (int)fvd.size());
            if (maskd) {
                Float16ToFloat32(maskd, fmaskd.data(), (int)fmaskd.size());
            }

            MultiThreadSingleAttentionOp(fqd.data(), fkd.data(), fvd.data(), maskd ? fmaskd.data() : nullptr, fod.data(), 
                            scale, q1, q2, k1, v2).Run();

            Float32ToFloat16(fod.data(), od, (int)fod.size());
        }
    };

    void CpuAttention::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &mask = *(datas.find("mask")->second);
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];
        float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0;
        output.Allocate();
        int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];

        if (q.dataType == DataType::FLOAT32) {
            float *qd = (float*)q.cpuData;
            float *kd = (float*)k.cpuData;
            float *vd = (float*)v.cpuData;
            float *maskd = (datas.find("mask")->second && mask.dims.size() > 0) ? (float*)mask.cpuData : nullptr;
            float *od = (float*)output.cpuData;
            int batch = (maskd != nullptr && mask.dims.size() == 3) ? mask.dims[0] : 1; 
            batch = intParams.find("mask___batch") != intParams.end() ? intParams.find("mask___batch")->second : batch;
            int maskStride = (maskd != nullptr) ? (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0)) : 0;
            std::fill(od, od + output.Count(0), 0.0f);

            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadSingleAttentionOp*> ops;
            for (int o = 0; o < q0; o++) {
                ops.push_back(new MultiThreadSingleAttentionOp(qd + o * q.strides[0], kd + (o / group) * k.strides[0], vd + (o / group) * v.strides[0],
                                maskd + (o / (q0 / batch)) * maskStride, od + o * output.strides[0], scale,
                                q1, q2, k1, v2));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        } else if (q.dataType == DataType::FLOAT16) {
            uint16_t *qd = (uint16_t*)q.cpuData;
            uint16_t *kd = (uint16_t*)k.cpuData;
            uint16_t *vd = (uint16_t*)v.cpuData;
            uint16_t *maskd = (datas.find("mask")->second && mask.dims.size() > 0) ? (uint16_t*)mask.cpuData : nullptr;
            uint16_t *od = (uint16_t*)output.cpuData;
            int batch = (maskd != nullptr && mask.dims.size() == 3) ? mask.dims[0] : 1; 
            batch = intParams.find("mask___batch") != intParams.end() ? intParams.find("mask___batch")->second : batch;
            int maskStride = (maskd != nullptr) ? (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0)) : 0;
            std::fill(od, od + output.Count(0), float_to_half(0.0f));

            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadSingleAttentionFloat16Op*> ops;
            for (int o = 0; o < q0; o++) {
                ops.push_back(new MultiThreadSingleAttentionFloat16Op(qd + o * q.strides[0], kd + (o / group) * k.strides[0], vd + (o / group) * v.strides[0],
                                maskd + (o / (q0 / batch)) * maskStride, od + o * output.strides[0], scale,
                                q1, q2, k1, v2));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        } else {
            ErrorInFastLLM("Attention error: unsupport dataType.\n");
        }
    }

    void OnlineQuantization(float *inputData, std::vector<uint8_t> &uinput, std::vector<LowBitConfig> &inputConfigs, 
                            int n, int m, int group, int groupCnt,
                            std::vector <float> &inputSums, std::vector <float> &iscales, std::vector <float> &izeros, 
                            int permuteType) {
        if (uinput.size() < n * m) {
            uinput.resize(n * m);
        }
        inputConfigs.resize(n * group);
        inputSums.resize(n * group);
        iscales.resize(n * group);
        izeros.resize(n * group);

        if (n > 1) {
            auto pool = GetAlivePool();
            int threadNum = pool->threads.size();
            int per = n / pool->threads.size();
            int cur = 0;
            std::vector<fastllm::MultiThreadOnlineQuantizationOp*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
                ops.push_back(new MultiThreadOnlineQuantizationOp(
                                inputData + cur * m, uinput.data() + cur * m, inputConfigs.data() + cur * group,
                                end - cur, m, group, groupCnt,
                                inputSums.data() + cur * group, iscales.data() + cur * group, izeros.data() + cur * group, permuteType));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        } else {
            MultiThreadOnlineQuantizationOp(inputData, uinput.data(), inputConfigs.data(), n, m, group, groupCnt,
                                            inputSums.data(), iscales.data(), izeros.data(), permuteType).Run();
        }
    }

    void MultiThreadSwigluGptOssOp::Run() {
        for (int o = 0; o < n; o++) {
                float *cur = (float*)input + o * inputStride;
                float *out = (float*)output + o * outputStride;
                int i = 0;
    #ifdef __aarch64__X
                float32x4_t c1 = vdupq_n_f32(1.0f);
                for (; i + 3 < len; i += 4) {
                    float32x4_t vx = vld1q_f32(cur + i);
                    float32x4_t vy = vld1q_f32(cur + i + mid);
                    vx = vdivq_f32(vx, vaddq_f32(c1, exp_ps(vnegq_f32(vx))));
                    vy = vmulq_f32(vx, vy);
                    vst1q_f32(out + i, vy);
                }
    #endif
    #ifdef __AVX2__X
                for (; i + 7 < len; i += 8) {  // Process 8 elements at a time
                    // Load x values (inputData[i..i+7]) and y values (inputData[i+mid..i+mid+7])
                    __m256 x = _mm256_loadu_ps(&cur[i]);
                    __m256 y = _mm256_loadu_ps(&cur[i + mid]);
                    
                    // Compute sigmoid: 1.0 / (1.0 + expf(-x))
                    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
                    __m256 exp_neg_x = exp256_ps(neg_x);  // See note below about exp_ps
                    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
                    __m256 sigmoid = _mm256_div_ps(x, denom);
                    
                    // Multiply by y and store result
                    __m256 result = _mm256_mul_ps(sigmoid, y);
                    _mm256_storeu_ps(&out[i], result);
                }
    #endif
                for (; i < len; i++) {
                    float x = cur[i * 2], y = cur[i * 2 + 1];
                    float gate = std::min(x, 7.0f);
                    float up = std::max(-7.0f, std::min(y, 7.0f));
                    float glu = gate * (1.0 / (1.0 + exp(-(gate * 1.702))));
                    out[i] = (up + 1) * glu;                    
                }
        }
    }

    void MultiThreadSwigluOp::Run() {
        for (int o = 0; o < n; o++) {
                float *cur = (float*)input + o * inputStride;
                float *out = (float*)output + o * outputStride;
                int i = 0;
    #ifdef __aarch64__
                float32x4_t c1 = vdupq_n_f32(1.0f);
                for (; i + 3 < len; i += 4) {
                    float32x4_t vx = vld1q_f32(cur + i);
                    float32x4_t vy = vld1q_f32(cur + i + mid);
                    vx = vdivq_f32(vx, vaddq_f32(c1, exp_ps(vnegq_f32(vx))));
                    vy = vmulq_f32(vx, vy);
                    vst1q_f32(out + i, vy);
                }
    #endif
    #ifdef __AVX2__
                for (; i + 7 < len; i += 8) {  // Process 8 elements at a time
                    // Load x values (inputData[i..i+7]) and y values (inputData[i+mid..i+mid+7])
                    __m256 x = _mm256_loadu_ps(&cur[i]);
                    __m256 y = _mm256_loadu_ps(&cur[i + mid]);
                    
                    // Compute sigmoid: 1.0 / (1.0 + expf(-x))
                    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
                    __m256 exp_neg_x = exp256_ps(neg_x);  // See note below about exp_ps
                    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
                    __m256 sigmoid = _mm256_div_ps(x, denom);
                    
                    // Multiply by y and store result
                    __m256 result = _mm256_mul_ps(sigmoid, y);
                    _mm256_storeu_ps(&out[i], result);
                }
    #endif
                for (; i < len; i++) {
                    float x = cur[i], y = cur[i + mid];
                    out[i] = (x / (1.0 + expf(-x))) * y;
                }
        }
    }

    void MultiThreadGegluOp::Run() {
        for (int o = 0; o < n; o++) {
            float *cur = (float*)input + o * inputStride;
            float *out = (float*)output + o * outputStride;
            for (int i = 0; i < len; i++) {
                float gate = cur[i], up = cur[i + mid];
                out[i] = gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up;
            }
        }
    }

    void MultiThreadCrossSwigluOp::Run() {
        // CrossSwiglu: y[i] = x[i*2+1] * silu(x[i*2])  (即 x[1::2] * silu(x[0::2]))
        for (int o = 0; o < n; o++) {
                float *cur = (float*)input + o * inputStride;
                float *out = (float*)output + o * outputStride;
                int i = 0;
    #ifdef __aarch64__
                float32x4_t c1 = vdupq_n_f32(1.0f);
                for (; i + 3 < len; i += 4) {
                    // 交错加载：x[0::2] 和 x[1::2]
                    float32x4x2_t xy = vld2q_f32(cur + i * 2);
                    float32x4_t vx = xy.val[0]; // 偶数位 (gate)
                    float32x4_t vy = xy.val[1]; // 奇数位 (up)
                    vx = vdivq_f32(vx, vaddq_f32(c1, exp_ps(vnegq_f32(vx))));
                    vy = vmulq_f32(vx, vy);
                    vst1q_f32(out + i, vy);
                }
    #endif
    #ifdef __AVX2__
                __m256i deinterleave_idx = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
                for (; i + 7 < len; i += 8) {
                    // 加载16个连续交错的float: [g0,u0,g1,u1,...,g7,u7]
                    __m256 lo = _mm256_loadu_ps(&cur[i * 2]);       // [g0,u0,g1,u1, g2,u2,g3,u3]
                    __m256 hi = _mm256_loadu_ps(&cur[i * 2 + 8]);   // [g4,u4,g5,u5, g6,u6,g7,u7]
                    // shuffle_ps 0x88 在每个128位lane内提取偶数位
                    __m256 x = _mm256_shuffle_ps(lo, hi, 0x88); // [g0,g1,g4,g5, g2,g3,g6,g7]
                    __m256 y = _mm256_shuffle_ps(lo, hi, 0xDD); // [u0,u1,u4,u5, u2,u3,u6,u7]
                    // 跨lane重排以恢复正确顺序
                    x = _mm256_permutevar8x32_ps(x, deinterleave_idx); // [g0,g1,g2,g3,g4,g5,g6,g7]
                    y = _mm256_permutevar8x32_ps(y, deinterleave_idx); // [u0,u1,u2,u3,u4,u5,u6,u7]

                    // Compute silu: x / (1.0 + exp(-x))
                    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
                    __m256 exp_neg_x = exp256_ps(neg_x);
                    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
                    __m256 sigmoid = _mm256_div_ps(x, denom);

                    __m256 result = _mm256_mul_ps(sigmoid, y);
                    _mm256_storeu_ps(&out[i], result);
                }
    #endif
                for (; i < len; i++) {
                    float x = cur[i * 2], y = cur[i * 2 + 1];
                    out[i] = (x / (1.0 + expf(-x))) * y;
                }
        }
    }

    void MultiThreadSwigluFloat16Op::Run() {
        for (int o = 0; o < n; o++) {
            uint16_t *cur = (uint16_t*)input + o * inputStride;
            uint16_t *out = (uint16_t*)output + o * outputStride;
            int i = 0;
#ifdef __AVX2__
            for (; i + 7 < len; i += 8) {  // Process 8 elements at a time
                __m128i x_half = _mm_loadu_si128((const __m128i*)&cur[i]);
                __m256 x = _mm256_cvtph_ps(x_half);  // Convert float16 to float32
    
                // Load 8 float16 values from cur[i+mid..i+mid+7] and convert to float32
                __m128i y_half = _mm_loadu_si128((const __m128i*)&cur[i + mid]);
                __m256 y = _mm256_cvtph_ps(y_half);  // Convert float16 to float32
                    
                // Compute sigmoid: 1.0 / (1.0 + expf(-x))
                __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
                __m256 exp_neg_x = exp256_ps(neg_x);  // See note below about exp_ps
                __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
                __m256 sigmoid = _mm256_div_ps(x, denom);
                    
                // Multiply by y and store result
                __m256 result = _mm256_mul_ps(sigmoid, y);
                
                // Convert result back to float16 and store
                __m128i result_half = _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128((__m128i*)&out[i], result_half);
            }
#endif
            for (; i < len; i++) {
                float x = fp16tofp32.dict[cur[i]], y = fp16tofp32.dict[cur[i + mid]];
                out[i] = float_to_half((x / (1.0 + expf(-x))) * y);
            }
        }
    }

    void MultiThreadGegluFloat16Op::Run() {
        for (int o = 0; o < n; o++) {
            uint16_t *cur = (uint16_t*)input + o * inputStride;
            uint16_t *out = (uint16_t*)output + o * outputStride;
            for (int i = 0; i < len; i++) {
                float gate = fp16tofp32.dict[cur[i]];
                float up = fp16tofp32.dict[cur[i + mid]];
                float val = gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up;
                out[i] = float_to_half(val);
            }
        }
    }

    void MultiThreadSwigluBFloat16Op::Run() {
        for (int o = 0; o < n; o++) {
            uint16_t *cur = (uint16_t*)input + o * inputStride;
            uint16_t *out = (uint16_t*)output + o * outputStride;
            int i = 0;
            for (; i < len; i++) {
                float x = bf16tofp32.dict[cur[i]], y = bf16tofp32.dict[cur[i + mid]];
                float val = (x / (1.0f + expf(-x))) * y;
                uint32_t tmp;
                memcpy(&tmp, &val, sizeof(tmp));
                out[i] = (uint16_t)(tmp >> 16);
            }
        }
    }

    void MultiThreadGegluBFloat16Op::Run() {
        for (int o = 0; o < n; o++) {
            uint16_t *cur = (uint16_t*)input + o * inputStride;
            uint16_t *out = (uint16_t*)output + o * outputStride;
            for (int i = 0; i < len; i++) {
                float gate = bf16tofp32.dict[cur[i]];
                float up = bf16tofp32.dict[cur[i + mid]];
                float val = gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up;
                uint32_t tmp;
                memcpy(&tmp, &val, sizeof(tmp));
                out[i] = (uint16_t)(tmp >> 16);
            }
        }
    }

    struct MultiThreadLinearInt4NoZeroOp : MultiThreadBaseOp {
        uint8_t *a, *b;
        int32_t *c;
        int n, m, k, kstride;
        int *weightSums;
        float *weightMins, *scales, *bias;
        LowBitConfig *config;
        float *inputSums;

        MultiThreadLinearInt4NoZeroOp(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride,
                      int *weightSums, float *weightMins, float *scales, float *bias, LowBitConfig *config,
                      float *inputSums) :
                      a(a), b(b), c(c), n(n), m(m), k(k), kstride(kstride), 
                      weightSums(weightSums), weightMins(weightMins), scales(scales), bias(bias), config(config), inputSums(inputSums) {}

#ifdef __ARM_FEATURE_DOTPROD
        inline static void RunSomeBlock(uint8_t *weightWalk, uint8_t *inputStart, int32_t *c, 
                            int curBlock, uint32x2_t *sum, uint8x8x2_t *vi, 
                            int block, int k, int m, int kstride) {
                uint8x8_t maskHigh = vdup_n_u8(0xF0);
                uint8x8_t maskLow = vdup_n_u8(0xF);
                for (int i = 0; i < k; i++) {
                    std::vector <int> values = std::vector <int> (curBlock, 0);
                    uint8_t *inputWalk = inputStart;
                    int j = 0;

                    for (int j = 0; j < curBlock; j++) {
                        sum[j][0] = sum[j][1] = 0;
                    }
                    for (; j + 15 < m; j += 16) {
                        for (int x = 0; x < curBlock; x++) {
                            vi[x] = vld2_u8(inputWalk + j + m * x);
                        }
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        for (int x = 0; x < curBlock; x++) {
                            sum[x] = vdot_u32(sum[x], va, vi[x].val[1]);
                            sum[x] = vdot_u32(sum[x], vb, vi[x].val[0]);
                        }
                    }
                    for (int x = 0; x < curBlock; x++) {
                        values[x] += sum[x][0] + sum[x][1];
                    }

                    for (; j + 1 < m; j += 2) {
                        int id = (i * m + j) / 2;
                        for (int x = 0; x < curBlock; x++) {
                            values[x] += (weightWalk[id] >> 4) * inputWalk[j + x * m];
                            values[x] += (weightWalk[id] & 0xF) * inputWalk[j + 1 + x * m];
                        }
                    }
                    
                    for (int x = 0; x < curBlock; x++) {
                        c[(block + x) * kstride + i] = values[x];
                    }
                }
        }
#endif
        void Run() {
#ifdef __ARM_FEATURE_DOTPROD
#define RUNBLOCK(x) for (; block + (x - 1) < n; block += (x)) RunSomeBlock(b, a + block * m, c, (x), sum, vi, block, k, m, kstride);
            int block = 0;
            uint32x2_t sum[16];
            uint8x8x2_t vi[16];
            RUNBLOCK(16);
            RUNBLOCK(8);RUNBLOCK(7);RUNBLOCK(6);RUNBLOCK(5);
            RUNBLOCK(4);RUNBLOCK(3);RUNBLOCK(2);RUNBLOCK(1);
#undef RUNBLOCK
#else
            int block = 0;

            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    int value = 0;
                    uint8_t *inputWalk = inputStart;
                    int j = 0;
#ifdef __ARM_FEATURE_DOTPROD
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x2_t sum0 = {0, 0};

                    for (; j + 15 < m; j += 16) {
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8x2_t in = vld2_u8(inputWalk + j);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        sum0 = vdot_u32(sum0, va, in.val[1]);
                        sum0 = vdot_u32(sum0, vb, in.val[0]);
                    }
                    value += sum0[0] + sum0[1];
#elif defined(__aarch64__)
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x4_t sum0 = {0, 0, 0, 0};

                    for (; j + 15 < m; j += 16) {
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8x2_t in = vld2_u8(inputWalk + j);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        sum0 = vpadalq_u16(sum0, vmull_u8(va, in.val[1]));
                        sum0 = vpadalq_u16(sum0, vmull_u8(vb, in.val[0]));
                    }
                    value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#elif defined(__AVX2__)
                    value += DotU4U8(weightWalk + i * m / 2, inputWalk, m);
                    j += m;
#endif

                    for (; j + 1 < m; j += 2) {
                        int id = (i * m + j) / 2;
                        value += (weightWalk[id] >> 4) * inputWalk[j];
                        value += (weightWalk[id] & 0xF) * inputWalk[j + 1];
                    }

                    c[block * kstride + i] = value;
                }
            }
#endif
            for (int block = 0; block < n; block++) {
                for (int i = 0; i < k; i++) {
                    int value = c[block * kstride + i];
                    value -= weightSums[i] * config[block].zeroPoint;
                    ((float*)c)[block * kstride + i] = scales[i] * config[block].scale * value +
                            weightMins[i] * ((float)inputSums[block] - (int)config[block].zeroPoint * m) * config[block].scale +
                            (bias == nullptr ? 0.0 : bias[i]);
                }
            }
        }
    };

    struct MOEIntSingleVarManager {
        std::vector<LowBitConfig> inputConfigs;
        std::vector<uint8_t> uinput;
        std::vector <float> inputSums;
        std::vector <float> iscales, izeros;
        std::vector <std::vector <float> > middles, results;
        std::vector <std::vector <LowBitConfig> > inputConfigsDown;
        std::vector <std::vector <uint8_t> > uinputsDown;
        std::vector <std::vector <float> > inputSumsDown;
        std::vector <std::vector <float> > iscalesDown, izerosDown;
        std::vector <std::vector <uint8_t> > uinputsGate;
    } moeIntSingleVarManager;

    struct moeFloatSingleVarManager {
        std::vector <std::vector <float> > middles, swigluResults, results;
        std::vector <int> localKs;
        std::vector <float*> tempResults;
        std::vector <uint16_t> bf16Input;
    } moeFloatSingleVarManager;

    // 转置权重缓冲。USE_CPU_NUMA 下优先按 node 分片（每片 numa_alloc_onnode 到
    // 对应 node，decode GEMV 读权重零远程）；不可分片时退回单块交织分配；再退回
    // 对齐分配。USE_NUMAS 走 numas 模块的交织分配。
    struct TransposedWeightBuf {
        uint16_t *data = nullptr;      // 单块模式
        size_t bytes = 0;
        bool numaAllocated = false;    // data 是否来自 numa_alloc_interleaved

        std::vector<uint16_t*> shards; // per-node 分片（仅 USE_CPU_NUMA）
        std::vector<size_t> shardBytes;
        int numCb = 0;
        int cbPerNode = 0;             // 分片模式下每 node 覆盖的 cb 数

        bool AllocateSingle(size_t b) {
            bytes = b;
#ifdef USE_NUMAS
            data = (uint16_t*)allocate_interleaved(b);
            return data != nullptr;
#elif defined(USE_CPU_NUMA)
            if (numa_available() == 0) {
                data = (uint16_t*)numa_alloc_interleaved(b);
                numaAllocated = data != nullptr;
            }
            if (data == nullptr) {
                data = (uint16_t*)alignedAllocator<uint8_t, 64>().allocate(b);
            }
            return data != nullptr;
#else
            data = (uint16_t*)alignedAllocator<uint8_t, 64>().allocate(b);
            return data != nullptr;
#endif
        }

        bool AllocateSharded(int cbTotal, int m, int nodes) {
#ifdef USE_CPU_NUMA
            if (nodes < 2 || cbTotal % nodes != 0) {
                return false;
            }
            int per = cbTotal / nodes;
            size_t sBytes = (size_t)per * (m / 8) * 512 * 2;
            shards.resize(nodes, nullptr);
            shardBytes.resize(nodes, 0);
            for (int n = 0; n < nodes; n++) {
                void *p = numa_alloc_onnode(sBytes, n);
                if (!p) {
                    for (int i = 0; i < n; i++) {
                        numa_free(shards[i], shardBytes[i]);
                    }
                    shards.clear();
                    shardBytes.clear();
                    return false;
                }
                shards[n] = (uint16_t*)p;
                shardBytes[n] = sBytes;
            }
            numCb = cbTotal;
            cbPerNode = per;
            return true;
#else
            return false;
#endif
        }

        void Free() {
            if (data) {
#ifdef USE_NUMAS
                free_interleaved(data, bytes);
#elif defined(USE_CPU_NUMA)
                if (numaAllocated) {
                    numa_free(data, bytes);
                } else {
                    alignedAllocator<uint8_t, 64>().deallocate((uint8_t*)data, bytes);
                }
#else
                alignedAllocator<uint8_t, 64>().deallocate((uint8_t*)data, bytes);
#endif
                data = nullptr;
                bytes = 0;
                numaAllocated = false;
            }
#ifdef USE_CPU_NUMA
            for (size_t i = 0; i < shards.size(); i++) {
                if (shards[i]) {
                    numa_free(shards[i], shardBytes[i]);
                }
            }
            shards.clear();
            shardBytes.clear();
            numCb = 0;
            cbPerNode = 0;
#endif
        }

        TransposedWeightBuf() = default;
        ~TransposedWeightBuf() { Free(); }
        TransposedWeightBuf(TransposedWeightBuf &&o) noexcept
            : data(o.data), bytes(o.bytes), numaAllocated(o.numaAllocated),
              shards(std::move(o.shards)), shardBytes(std::move(o.shardBytes)),
              numCb(o.numCb), cbPerNode(o.cbPerNode) {
            o.data = nullptr;
            o.bytes = 0;
            o.numaAllocated = false;
            o.numCb = 0;
            o.cbPerNode = 0;
        }
        TransposedWeightBuf& operator=(TransposedWeightBuf &&o) noexcept {
            if (this != &o) {
                Free();
                data = o.data;
                bytes = o.bytes;
                numaAllocated = o.numaAllocated;
                shards = std::move(o.shards);
                shardBytes = std::move(o.shardBytes);
                numCb = o.numCb;
                cbPerNode = o.cbPerNode;
                o.data = nullptr;
                o.bytes = 0;
                o.numaAllocated = false;
                o.numCb = 0;
                o.cbPerNode = 0;
            }
            return *this;
        }
        TransposedWeightBuf(const TransposedWeightBuf&) = delete;
        TransposedWeightBuf& operator=(const TransposedWeightBuf&) = delete;
    };

    struct FastllmMoeDataManager {
            std::vector <float, alignedAllocator<float, 64> > gateUpOutput, swigluOutput, downOutput, reduceOutput;
            std::vector <uint8_t, alignedAllocator<uint8_t, 64> > realInput, expandInput, downInput;
            // 块转置权重的缓存，key 为原始权重的 cpuData 指针（加载后固定）
            std::mutex weightTLock;
            std::unordered_map <const uint16_t *, TransposedWeightBuf> transposedWeights;
    } fastllmMoeDataManager;

    // 将 BF16 权重 [k][m]（行主序）的 cb 区间 [cbStart, cbEnd) 转置为块布局
    // [cb][m/8][64][8]，写入 dst 从局部下标 0 开始
    static void TransposeBF16MoEWeightsBlock64Range(const uint16_t *src, uint16_t *dst,
                                                    int k, int m, int cbStart, int cbEnd) {
        for (int cb = cbStart; cb < cbEnd; cb++) {
            uint16_t *dstBase = dst + (size_t)(cb - cbStart) * (m / 8) * 512;
            for (int mb = 0; mb < m; mb += 8) {
                uint16_t *d = dstBase + (mb / 8) * 512;
                for (int c = 0; c < 64; c++) {
                    memcpy(d + c * 8, src + (size_t)(cb * 64 + c) * m + mb, 16);
                }
            }
        }
    }

    // 转置权重的访问视图：单块（numaCnt==1）或 per-node 分片（numaCnt>=2）
    struct TransposedWeightView {
        const uint16_t *single = nullptr;
        const uint16_t *shards[8] = {};
        int cbStart[8] = {}, cbEnd[8] = {};
        int numCb = 0;
        int numaCnt = 1;

        bool Valid() const { return numCb > 0; }

        int NodeOf(int cb) const {
            for (int n = 0; n < numaCnt; n++) {
                if (cb >= cbStart[n] && cb < cbEnd[n]) {
                    return n;
                }
            }
            return 0;
        }

        int LocalCb(int cb) const { return cb - cbStart[NodeOf(cb)]; }

        const uint16_t *ShardBase(int cb) const {
            if (numaCnt <= 1) {
                return single;
            }
            return shards[NodeOf(cb)];
        }
    };

    static TransposedWeightView BuildTransposedWeightView(const TransposedWeightBuf &buf, int numCb) {
        TransposedWeightView view;
        view.numCb = numCb;
        if (!buf.shards.empty()) {
            view.numaCnt = (int)buf.shards.size();
            for (int n = 0; n < view.numaCnt; n++) {
                view.shards[n] = buf.shards[n];
                view.cbStart[n] = n * buf.cbPerNode;
                view.cbEnd[n] = (n + 1) * buf.cbPerNode;
            }
        } else {
            view.numaCnt = 1;
            view.single = buf.data;
            view.cbStart[0] = 0;
            view.cbEnd[0] = numCb;
        }
        return view;
    }

    // 获取（或生成）块转置权重；不支持时返回 numCb==0 的视图
    static TransposedWeightView GetTransposedBF16Weight(const uint16_t *src, int k, int m) {
        TransposedWeightView view;
        if (src == nullptr || k <= 0 || m <= 0 || k % 64 != 0 || m % 8 != 0) {
            return view;
        }
        int numCb = k / 64;
        {
            std::lock_guard <std::mutex> lock(fastllmMoeDataManager.weightTLock);
            auto it = fastllmMoeDataManager.transposedWeights.find(src);
            if (it != fastllmMoeDataManager.transposedWeights.end()) {
                return BuildTransposedWeightView(it->second, numCb);
            }
        }

        TransposedWeightBuf buf;
        int nodes = 1;
#ifdef USE_CPU_NUMA
        InitCpuNumaBinding();
        CpuNumaState &state = GetCpuNumaState();
        nodes = state.enabled ? std::min(state.numaCnt, 8) : 1;
#endif
        if (!buf.AllocateSharded(numCb, m, nodes)) {
            if (!buf.AllocateSingle((size_t)k * m * 2)) {
                return view;
            }
            TransposeBF16MoEWeightsBlock64Range(src, buf.data, k, m, 0, numCb);
        } else {
            int per = numCb / nodes;
            for (int n = 0; n < nodes; n++) {
                TransposeBF16MoEWeightsBlock64Range(src, buf.shards[n], k, m, n * per, (n + 1) * per);
            }
        }

        std::lock_guard <std::mutex> lock(fastllmMoeDataManager.weightTLock);
        auto ret = fastllmMoeDataManager.transposedWeights.emplace(src, std::move(buf));
        return BuildTransposedWeightView(ret.first->second, numCb);
    }

    // decode 专用 GEMV：1 个 token × 64 输出列，权重为块转置布局，外层按 m 连续读 1024B
    static inline __m256 FastllmBF16ToFP32AVX2(__m128i bf16_data) {
        __m256i fp32_data = _mm256_cvtepu16_epi32(bf16_data);
        fp32_data = _mm256_slli_epi32(fp32_data, 16);
        return _mm256_castsi256_ps(fp32_data);
    }

    static void GemvBF16Block64AVX2(const uint16_t *input, const uint16_t *weightT,
                                    float *output, int m, int cb) {
#ifdef __AVX2__
        const uint16_t *base = weightT + (size_t)cb * (m / 8) * 512;
        for (int g = 0; g < 8; g++) {
            __m256 acc[8];
            for (int c = 0; c < 8; c++) {
                acc[c] = _mm256_setzero_ps();
            }
            for (int mb = 0; mb < m; mb += 8) {
                __m128i a = _mm_loadu_si128((const __m128i*)(input + mb));
                __m256 af = FastllmBF16ToFP32AVX2(a);
                const uint16_t *wp = base + (mb / 8) * 512 + g * 64;
                for (int c = 0; c < 8; c++) {
                    __m128i w = _mm_loadu_si128((const __m128i*)(wp + c * 8));
                    acc[c] = _mm256_fmadd_ps(af, FastllmBF16ToFP32AVX2(w), acc[c]);
                }
            }
            for (int c = 0; c < 8; c++) {
                output[g * 8 + c] = Floatsum(acc[c]);
            }
        }
#else
        for (int j = 0; j < 64; j++) {
            float sum = 0.0f;
            for (int l = 0; l < m; l++) {
                sum += bf16tofp32.dict[input[l]] * bf16tofp32.dict[weightT[cb * (m / 8) * 512 + (l / 8) * 512 + j * 8 + (l % 8)]];
            }
            output[j] = sum;
        }
#endif
    }

    struct MultiThreadGemvBF16Block64Op : MultiThreadBaseOp {
        const uint16_t *input, *weightT;
        float *output;
        int m, cb, outCb, node;

        MultiThreadGemvBF16Block64Op(const uint16_t *input, const uint16_t *weightT,
                                     float *output, int m, int cb, int outCb, int node)
                : input(input), weightT(weightT), output(output), m(m), cb(cb), outCb(outCb), node(node) {}

        void Run() override {
            GemvBF16Block64AVX2(input, weightT, output + outCb * 64, m, cb);
        }
    };

    // decode 转置 GEMV 任务的调度：NUMA 启用且全部为转置任务时，按 node 静态分片，
    // 每个 node 的任务只丢给该 node 的线程做 work-stealing（权重零远程）；否则退回
    // 全局 DynamicScheduleTasks。
    static void ScheduleCpuMoEGemm(std::vector<MultiThreadBaseOp*> &ops, int transposedCount) {
#ifdef USE_CPU_NUMA
        InitCpuNumaBinding();
        CpuNumaState &state = GetCpuNumaState();
        if (state.enabled && transposedCount == (int)ops.size() && transposedCount > 0) {
            std::vector<std::vector<MultiThreadBaseOp*>> perNode(state.numaCnt);
            for (auto *op : ops) {
                auto *g = (MultiThreadGemvBF16Block64Op*)op;
                int n = (g->node >= 0 && g->node < state.numaCnt) ? g->node : 0;
                perNode[n].push_back(g);
            }
            for (int n = 0; n < state.numaCnt; n++) {
                if (!perNode[n].empty()) {
                    DynamicScheduleTasksRange(perNode[n], state.nodeThreadStart[n],
                                              state.nodeThreadCount[n], GetAlivePool());
                }
            }
            return;
        }
#endif
        DynamicScheduleTasks(ops);
    }

    void FastllmGemm (int n, int m, int k, 
        const void *A, long lda, // A [n * m], lda = bytes for 1 row in A
        const void *B, long ldb, // B [k * m], ldb = bytes for 1 row in B
        void *C, long ldc, // C[n * k], ldc = bytes for 1 row in C
        int st, int end, // calc C[0 : n, st : end]
        DataType AType, DataType BType, DataType CType
    ) {
        bool finish = false;
// printf("into fastllm gemm %s %s %s\n", GetDataTypeName(AType).c_str(), GetDataTypeName(BType).c_str(), GetDataTypeName(CType).c_str());
        if (AType >= DataType::DATA_GGUF_FORMAT && AType < DataType::DATA_GGUF_FORMAT_END) {
            if (CType == DataType::FLOAT32) {
                LinearQ8K_GGUF_Kernel((uint8_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end, AType, BType);
                finish = true;
            }
        } else if (AType == DataType::INF_INT8_PERCHANNEL) {
        	if (CType == DataType::FLOAT32) {
        		if (BType == DataType::INT4_PERCHANNEL) {
                    LinearINT8PERCHANNEL_INT4PERCHANNEL_Kernel((uint8_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    finish = true;
        		} else if (BType == DataType::INT8_PERCHANNEL) {
                    LinearINT8PERCHANNEL_INT8PERCHANNEL_Kernel((uint8_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    finish = true;
                }
        	}
        } else if (AType == DataType::INF_INT8_GROUP128) {
        	if (CType == DataType::FLOAT32) {
        		if (BType == DataType::INT4_GROUP128) {
                    LinearINT8GROUP128_INT4GROUP128_Kernel((uint8_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    finish = true;
        		}
        	}
        } else if (AType == DataType::INF_INT8_GROUP32) {
            if (CType == DataType::FLOAT32 && BType == DataType::INT4_GROUP32) {
                LinearINT8GROUP32_INT4GROUP32_Kernel((uint8_t*)A, (uint8_t*)B, nullptr,
                                                     (float*)C, n, m, ldc / sizeof(float), st, end);
                finish = true;
            }
        } else if (AType == DataType::FLOAT32) {
            if (CType == DataType::FLOAT32) {
                if (BType == DataType::FLOAT32) {
                    if (lda == (long)m * sizeof(float) &&
                        ldb == (long)m * sizeof(float)) {
                        MultiThreadLinearFloat32Float32Op(
                            (float*)A, (float*)B, nullptr, (float*)C,
                            n, m, (int)(ldc / sizeof(float)), st, end).Run();
                    } else {
                        for (int i = 0; i < n; i++) {
                            float *floatA = (float*)((uint8_t*)A + i * lda);
                            float *floatC = (float*)((uint8_t*)C + i * ldc);
                            for (int j = st; j < end; j++) {
                                float *floatB = (float*)((uint8_t*)B + j * ldb);
                                float sum = 0.0f;
                                for (int l = 0; l < m; l++) {
                                    sum += floatA[l] * floatB[l];
                                }
                                floatC[j] = sum;
                            }
                        }
                    }
                    finish = true;
                } else if (BType == DataType::BFLOAT16) {
                    extern bool LinearFloat32BFloat16_AVX2_Kernel(
                        float *inputData, uint16_t *weightData,
                        float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
                    static const bool profileF32BF16 =
                        std::getenv("FASTLLM_PROFILE_F32BF16") != nullptr;
                    auto t0 = profileF32BF16 ?
                        std::chrono::steady_clock::now() :
                        std::chrono::steady_clock::time_point();
                    bool usedAvx2 = false;
                    if (cpuInstructInfo.hasAVX2 && n >= 1) {
                        usedAvx2 = LinearFloat32BFloat16_AVX2_Kernel(
                            (float*)A, (uint16_t*)B, nullptr, (float*)C,
                            n, m, (int)(ldc / sizeof(float)), st, end);
                    }
                    if (profileF32BF16) {
                        auto t1 = std::chrono::steady_clock::now();
                        printf(
                            "[fastllm-profile-f32bf16] n=%d m=%d cols=%d "
                            "avx2=%d %.3f ms\n",
                            n, m, end - st, usedAvx2 ? 1 : 0,
                            std::chrono::duration<double, std::milli>(
                                t1 - t0).count());
                    }
                    if (!usedAvx2) {
                        for (int i = 0; i < n; i++) {
                            float *floatA = (float*)((uint8_t*)A + i * lda);
                            float *floatC = (float*)((uint8_t*)C + i * ldc);
                            for (int j = st; j < end; j++) {
                                uint16_t *floatB = (uint16_t*)((uint8_t*)B + j * ldb);
                                float sum = 0.0f;
                                for (int l = 0; l < m; l++) {
                                    sum += floatA[l] * bf16tofp32.dict[floatB[l]];
                                }
                                floatC[j] = sum;
                            }
                        }
                    }
                    finish = true;
                } else if (BType == DataType::FLOAT16) {
                    MultiThreadLinearFloat32Float16Op (
                        (float*)A, (uint16_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end
                    ).Run();
                    finish = true;
                } else if (BType == DataType::FP8_E4M3_BLOCK_128) {
                    const int blockSize = 128;
                    int blocks = (m - 1) / blockSize + 1;
                    for (int i = 0; i < n; i++) {
                        float *floatA = (float*)((uint8_t*)A + i * lda);
                        float *floatC = (float*)((uint8_t*)C + i * ldc);
                        for (int j = st; j < end; j++) {
                            uint8_t *rowStart = (uint8_t*)B + j * ldb;
                            float now = 0.0f;
                            for (int block = 0; block < blocks; block++) {
                                uint8_t *blockStart = rowStart + block * (blockSize + sizeof(float));
                                float scale;
                                memcpy(&scale, blockStart + blockSize, sizeof(float));
                                int l = block * blockSize;
                                int blockEnd = std::min(m, l + blockSize);
                                for (; l < blockEnd; l++) {
                                    now += scale * floatA[l] * fp8e4m3tofp32.dict[blockStart[l - block * blockSize]];
                                }
                            }
                            floatC[j] = now;
                        }
                    }
                    finish = true;
                } else if (BType == DataType::FP8_E4M3_PERCHANNEL) {
                    for (int i = 0; i < n; i++) {
                        float *floatA = (float*)((uint8_t*)A + i * lda);
                        float *floatC = (float*)((uint8_t*)C + i * ldc);
                        for (int j = st; j < end; j++) {
                            uint8_t *rowStart = (uint8_t*)B + j * ldb;
                            float scale;
                            memcpy(&scale, rowStart + m, sizeof(float));
                            float now = 0.0f;
                            for (int l = 0; l < m; l++) {
                                now += scale * floatA[l] * fp8e4m3tofp32.dict[rowStart[l]];
                            }
                            floatC[j] = now;
                        }
                    }
                    finish = true;
                } else if (BType == DataType::NVFP4_BLOCK_32_E8M0) {
                    // This compact layout is internal to the NUMA BF16
                    // decode path. Keep the generic FP32 entry correct by
                    // converting through BF16 when it is called directly.
                    std::vector<uint16_t> bf16A_temp((size_t)n * m);
                    for (int i = 0; i < n; i++) {
                        Float32ToBFloat16(
                            (float*)((uint8_t*)A + (size_t)i * lda),
                            bf16A_temp.data() + (size_t)i * m, m);
                    }
                    if (n <= 31 && cpuInstructInfo.hasAVX512BF16 &&
                        FastllmGemmBFloat16NVFP4Block32E8M0_AVX512BF16(
                            bf16A_temp.data(), m * (long)sizeof(uint16_t),
                            B, ldb, C, ldc, n, m, k, st, end)) {
                        finish = true;
                        return;
                    }
                    std::vector<uint16_t> bf16B_temp(
                        (size_t)(end - st) * m);
                    NVFP4Block32RowsToBFloat16(
                        B, ldb, bf16B_temp.data(), m, st, end);
                    MultiThreadLinearBFloat16BFloat16Op(
                        bf16A_temp.data(), bf16B_temp.data(), nullptr,
                        ((float*)C) + st, n, m, ldc / sizeof(float),
                        0, end - st).Run();
                    finish = true;
                } else if (BType == DataType::NVFP4_BLOCK_16 ||
                           BType == DataType::NVFP4_BLOCK_16_E8M0) {
                    bool scaleE8M0 = BType == DataType::NVFP4_BLOCK_16_E8M0;
                    if (n > 31) {
                        std::vector<uint16_t> bf16B_temp((size_t)(end - st) * m);
                        NVFP4Block16RowsToBFloat16(B, ldb, bf16B_temp.data(), m, st, end, scaleE8M0);
                        std::vector<uint16_t> bf16A_temp((size_t)n * m);
                        for (int i = 0; i < n; i++) {
                            Float32ToBFloat16((float*)((uint8_t*)A + (size_t)i * lda), bf16A_temp.data() + (size_t)i * m, m);
                        }
                        MultiThreadLinearBFloat16BFloat16Op(
                            bf16A_temp.data(), bf16B_temp.data(), nullptr, ((float*)C) + st,
                            n, m, ldc / sizeof(float), 0, end - st
                        ).Run();
                        finish = true;
                        return;
                    }
                    if (cpuInstructInfo.hasAVX512BF16) {
                        if (scaleE8M0 && FastllmGemmFloat32NVFP4Block16E8M0_AVX512BF16(A, lda, B, ldb, C, ldc, n, m, k, st, end)) {
                            finish = true;
                            return;
                        }
                        if (!scaleE8M0 && FastllmGemmFloat32NVFP4Block16_AVX512BF16(A, lda, B, ldb, C, ldc, n, m, k, st, end)) {
                            finish = true;
                            return;
                        }
                    }
                    if (scaleE8M0) {
                        GemmNVFP4Block16_CPU_Run<false, true>(A, lda, B, ldb, C, ldc, n, m, st, end);
                    } else {
                        GemmNVFP4Block16_CPU_Run<false>(A, lda, B, ldb, C, ldc, n, m, st, end);
                    }
                    finish = true;
                }
            }
        } else if (AType == DataType::BFLOAT16) {
            if (CType == DataType::FLOAT32) {
                if (BType == DataType::FLOAT32) {
                    for (int i = 0; i < n; i++) {
                        uint16_t *bf16A = (uint16_t*)((uint8_t*)A + i * lda);
                        float *floatC = (float*)((uint8_t*)C + i * ldc);
                        for (int j = st; j < end; j++) {
                            float *floatB = (float*)((uint8_t*)B + j * ldb);
                            float sum = 0.0f;
                            int l = 0;
#ifdef __AVX2__
                            __m256 vsum = _mm256_setzero_ps();
                            for (; l + 7 < m; l += 8) {
                                __m128i bf16 = _mm_loadu_si128((const __m128i*)(bf16A + l));
                                __m256i extended = _mm256_cvtepu16_epi32(bf16);
                                __m256 va = _mm256_castsi256_ps(_mm256_slli_epi32(extended, 16));
                                __m256 vb = _mm256_loadu_ps(floatB + l);
                                vsum = _mm256_fmadd_ps(va, vb, vsum);
                            }
                            sum += Floatsum(vsum);
#endif
                            for (; l < m; l++) {
                                sum += bf16tofp32.dict[bf16A[l]] * floatB[l];
                            }
                            floatC[j] = sum;
                        }
                    }
                    finish = true;
                } else if (BType == DataType::BFLOAT16) {
                    // LinearBFloat16BFloat16_Kernel((uint16_t*)A, (uint16_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    MultiThreadLinearBFloat16BFloat16Op (
                        (uint16_t*)A, (uint16_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end
                    ).Run();
                    finish = true;
                } else if (BType == FP8_E4M3_BLOCK_128) {
                    // A是BFLOAT16, B是FP8_E4M3_BLOCK_128格式（fp8数据+scale）, C是FLOAT32
                    // 为需要计算的行分配临时bf16缓冲区
                    if (n > 31) {
                        std::vector<uint16_t> bf16B_temp((end - st) * m);
                        // 转换fp8到bf16，仅转换需要计算的行[st:end]
                        int block_size = 128;
                        int num_blocks = (m + block_size - 1) / block_size;
                        int last_block_size = (m % block_size == 0) ? block_size : (m % block_size);
                        for (int j = st; j < end; j++) {
                            uint8_t *rowStart = (uint8_t*)B + j * ldb;  // ldb应该是每行的总字节数
                            uint16_t *bf16B_row = bf16B_temp.data() + (j - st) * m;
                            
                            // 按block进行处理
                            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                                // 计算当前block的大小（最后一个block可能不完整）
                                int current_block_size = (block_idx == num_blocks - 1) ? last_block_size : block_size;
                                
                                // 计算当前block的起始位置
                                // 每个block占用 128字节(fp8) + 4字节(float scale)
                                uint8_t *block_start = rowStart + block_idx * (block_size + sizeof(float));
                                uint8_t *fp8_ptr = block_start;
                                float *scale_ptr = (float*)(block_start + block_size);
                                
                                // 转换当前block中的每个fp8到bf16
                                int base_idx = block_idx * block_size;
                                for (int l = 0; l < current_block_size; l++) {
                                    // fp8转fp32并乘以scale
                                    float fp32_val = fp8e4m3tofp32.dict[fp8_ptr[l]] * (*scale_ptr);
                                    
                                    // fp32转bf16
                                    uint32_t val;
                                    memcpy(&val, &fp32_val, sizeof(val));
                                    bf16B_row[base_idx + l] = (uint16_t)(val >> 16);
                                }
                            }
                        }

                        MultiThreadLinearBFloat16BFloat16Op (
                            (uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st
                        ).Run();
                        finish = true;
                        // LinearBFloat16BFloat16_Kernel((uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st);
                    } else {
                        LinearBFloat16_FP8E4M3BLOCK128_Kernel((uint16_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                        finish = true;
                    }
                } else if (BType == FP8_E4M3_PERCHANNEL) {
                    // A是BFLOAT16, B是FP8_E4M3_PERCHANNEL格式（fp8数据+scale）, C是FLOAT32
                    // 为需要计算的行分配临时bf16缓冲区
                    if (n > 31) {
                        std::vector<uint16_t> bf16B_temp((end - st) * m);
                        // 转换fp8到bf16，仅转换需要计算的行[st:end]
                        int block_size = m;
                        int num_blocks = (m + block_size - 1) / block_size;
                        int last_block_size = (m % block_size == 0) ? block_size : (m % block_size);
                        for (int j = st; j < end; j++) {
                            uint8_t *rowStart = (uint8_t*)B + j * ldb;  // ldb应该是每行的总字节数
                            uint16_t *bf16B_row = bf16B_temp.data() + (j - st) * m;
                            
                            // 按block进行处理
                            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                                // 计算当前block的大小（最后一个block可能不完整）
                                int current_block_size = (block_idx == num_blocks - 1) ? last_block_size : block_size;
                                
                                // 计算当前block的起始位置
                                // 每个block占用 128字节(fp8) + 4字节(float scale)
                                uint8_t *block_start = rowStart + block_idx * (block_size + sizeof(float));
                                uint8_t *fp8_ptr = block_start;
                                float *scale_ptr = (float*)(block_start + block_size);
                                
                                // 转换当前block中的每个fp8到bf16
                                int base_idx = block_idx * block_size;
                                for (int l = 0; l < current_block_size; l++) {
                                    // fp8转fp32并乘以scale
                                    float fp32_val = fp8e4m3tofp32.dict[fp8_ptr[l]] * (*scale_ptr);
                                    
                                    // fp32转bf16
                                    uint32_t val;
                                    memcpy(&val, &fp32_val, sizeof(val));
                                    bf16B_row[base_idx + l] = (uint16_t)(val >> 16);
                                }
                            }
                        }

                        MultiThreadLinearBFloat16BFloat16Op (
                            (uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st
                        ).Run();
                        finish = true;
                    } else {
                        LinearBFloat16_FP8E4M3PERCHANNEL_Kernel((uint16_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                        finish = true;
                    }
                } else if (BType == NVFP4_BLOCK_32_E8M0) {
                    if (n <= 31 && cpuInstructInfo.hasAVX512BF16 &&
                        FastllmGemmBFloat16NVFP4Block32E8M0_AVX512BF16(
                            A, lda, B, ldb, C, ldc,
                            n, m, k, st, end)) {
                        finish = true;
                        return;
                    }
                    std::vector<uint16_t> bf16B_temp(
                        (size_t)(end - st) * m);
                    NVFP4Block32RowsToBFloat16(
                        B, ldb, bf16B_temp.data(), m, st, end);
                    MultiThreadLinearBFloat16BFloat16Op(
                        (uint16_t*)A, bf16B_temp.data(), nullptr,
                        ((float*)C) + st, n, m, ldc / sizeof(float),
                        0, end - st).Run();
                    finish = true;
                } else if (BType == NVFP4_BLOCK_16 ||
                           BType == NVFP4_BLOCK_16_E8M0) {
                    bool scaleE8M0 = BType == DataType::NVFP4_BLOCK_16_E8M0;
                    if (n > 31) {
                        std::vector<uint16_t> bf16B_temp((size_t)(end - st) * m);
                        NVFP4Block16RowsToBFloat16(B, ldb, bf16B_temp.data(), m, st, end, scaleE8M0);
                        MultiThreadLinearBFloat16BFloat16Op(
                            (uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st,
                            n, m, ldc / sizeof(float), 0, end - st
                        ).Run();
                        finish = true;
                        return;
                    }
                    if (cpuInstructInfo.hasAVX512BF16) {
                        if (scaleE8M0 && FastllmGemmBFloat16NVFP4Block16E8M0_AVX512BF16(A, lda, B, ldb, C, ldc, n, m, k, st, end)) {
                            finish = true;
                            return;
                        }
                        if (!scaleE8M0 && FastllmGemmBFloat16NVFP4Block16_AVX512BF16(A, lda, B, ldb, C, ldc, n, m, k, st, end)) {
                            finish = true;
                            return;
                        }
                    }
                    if (scaleE8M0) {
                        GemmNVFP4Block16_CPU_Run<true, true>(A, lda, B, ldb, C, ldc, n, m, st, end);
                    } else {
                        GemmNVFP4Block16_CPU_Run<true>(A, lda, B, ldb, C, ldc, n, m, st, end);
                    }
                    finish = true;
                } else if (BType == AWQ_4BIT_128) {
                    // A是BFLOAT16, B是AWQ_4BIT_128格式（uint4权重+zero+scale）, C是FLOAT32
                    // 为需要计算的行分配临时bf16缓冲区
                    /* if (n > 31) {
                        std::vector<uint16_t> bf16B_temp((end - st) * m);
                        bool success = AWQ4BIT128_TO_BFloat16_Kernel((uint8_t*)B, bf16B_temp.data(), m, st, end, ldb);
                        LinearBFloat16BFloat16_Kernel((uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st);
                    } else {
                        LinearBFloat16_AWQ4BIT128_Kernel((uint16_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    } */ 
                } else if (BType >= DataType::DATA_GGUF_FORMAT && BType < DataType::DATA_GGUF_FORMAT_END) {
                    std::vector <float> fp32B_temp((end - st) * m);
                    std::vector <uint16_t> bf16B_temp((end - st) * m);
                    ggml_type weightType = (ggml_type)((int)BType - (int)DataType::DATA_GGUF_FORMAT);

                    auto toFloat = ggml_type_to_float(weightType);
                    AssertInFastLLM(toFloat != nullptr, "WeightImportGGUFTensor: weight (type " + std::string(ggml_type_name(weightType)) + ") can't convert to fp32.");
                    toFloat(((uint8_t*)B) + ldb * st, fp32B_temp.data(), (end - st) * m);
                    Float32ToBFloat16(fp32B_temp.data(), bf16B_temp.data(), (end - st) * m);
                    MultiThreadLinearBFloat16BFloat16Op (
                            (uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st
                    ).Run();
                    finish = true;
                }
            }
        }
        
        if (!finish) {
            ErrorInFastLLM("FastllmGemm Error: \nAType = " + GetDataTypeName(AType) + "\nBType = " + GetDataTypeName(BType) + "\nCType = " + GetDataTypeName(CType));
        }
    }

    static void AddBiasToFloatOutput(float *outputData, float *biasData, int n, int k) {
        if (biasData == nullptr) {
            return;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                outputData[(uint64_t)i * k + j] += biasData[j];
            }
        }
    }

    static ggml_type GetLinearGGMLType(const Data &weight) {
        if (weight.ggmlTensor != nullptr) {
            return ((ggml_tensor*)weight.ggmlTensor)->type;
        }
        return (ggml_type)weight.ggmlType;
    }

    static bool CanRunBFloat16NativeLinearWeight(const Data &weight) {
        if (weight.dataType == DataType::FLOAT32) {
            return true;
        }
        if (weight.dataType != DataType::DATA_GGUF_FORMAT) {
            return false;
        }
        ggml_type type = GetLinearGGMLType(weight);
        if ((int)type < 0 || type >= GGML_TYPE_COUNT) {
            return false;
        }
        ggml_type dotType = ggml_type_vec_dot_type(type);
        return type == GGML_TYPE_F32 ||
               (ggml_is_quantized(type) &&
                (dotType == GGML_TYPE_Q8_0 || dotType == GGML_TYPE_Q8_1 ||
                 dotType == GGML_TYPE_Q8_K || dotType == GGML_TYPE_Q8_K32));
    }

    static void RunLinearBFloat16GGUF(uint16_t *inputData, Data &weight,
                                      float *outputData, float *biasData,
                                      int n, int m, int k,
                                      AliveThreadPool *pool, int startTid, int threadNum) {
        std::vector<float> floatInput((uint64_t)n * m);
        BFloat16ToFloat32(inputData, floatInput.data(), n * m);
        RunLinearFloat32GGUF(floatInput.data(), (uint8_t*)weight.cpuData, outputData, biasData,
                             &weight, n, m, k, pool, startTid, threadNum);
    }

    static void RunLinearBFloat16NativeToFloat32(uint16_t *inputData, Data &weight,
                                                float *outputData, float *biasData,
                                                int n, int m, int k,
                                                AliveThreadPool *pool, int startTid, int threadNum) {
        if (weight.dataType == DataType::DATA_GGUF_FORMAT && ggml_is_quantized(GetLinearGGMLType(weight))) {
            RunLinearBFloat16GGUF(inputData, weight, outputData, biasData, n, m, k, pool, startTid, threadNum);
            return;
        }
        DataType weightType = weight.dataType == DataType::DATA_GGUF_FORMAT ?
            DataType::FLOAT32 : weight.dataType;
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadGemmOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            if (i == threadNum - 1) {
                end = k;
            }
            ops.push_back(new MultiThreadGemmOp(
                (uint8_t*)inputData, DataType::BFLOAT16,
                (uint8_t*)weight.cpuData, weightType,
                (uint8_t*)outputData, DataType::FLOAT32,
                n, m, k, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
        AddBiasToFloatOutput(outputData, biasData, n, k);
    }

    void MultiThreadGemmOp::Run() {
            FastllmGemm(
                n, m, k,
                inputData, GetDataBytes(inputDataType, 1, m),
                weightData, GetDataBytes(weightDataType, 1, m),
                outputData, GetDataBytes(outputDataType, 1, k),
                st, end,
                inputDataType, weightDataType, outputDataType
            );
    }

    static inline void CrossSwigluFloat32Chunk(
            const float *input, int len, float *output) {
        int i = 0;
#ifdef __aarch64__
        float32x4_t c1 = vdupq_n_f32(1.0f);
        for (; i + 3 < len; i += 4) {
            float32x4x2_t xy = vld2q_f32(input + i * 2);
            float32x4_t vx = xy.val[0];
            float32x4_t vy = xy.val[1];
            vx = vdivq_f32(vx, vaddq_f32(c1, exp_ps(vnegq_f32(vx))));
            vst1q_f32(output + i, vmulq_f32(vx, vy));
        }
#endif
#ifdef __AVX2__
        const __m256i deinterleaveIdx =
            _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        for (; i + 7 < len; i += 8) {
            const __m256 lo = _mm256_loadu_ps(input + i * 2);
            const __m256 hi = _mm256_loadu_ps(input + i * 2 + 8);
            __m256 x = _mm256_shuffle_ps(lo, hi, 0x88);
            __m256 y = _mm256_shuffle_ps(lo, hi, 0xDD);
            x = _mm256_permutevar8x32_ps(x, deinterleaveIdx);
            y = _mm256_permutevar8x32_ps(y, deinterleaveIdx);

            const __m256 expNegX = exp256_ps(
                _mm256_sub_ps(_mm256_setzero_ps(), x));
            const __m256 den = _mm256_add_ps(
                _mm256_set1_ps(1.0f), expNegX);
            __m256 r = _mm256_rcp_ps(den);
            r = _mm256_mul_ps(
                r, _mm256_fnmadd_ps(den, r, _mm256_set1_ps(2.0f)));
            const __m256 silu = _mm256_mul_ps(x, r);
            _mm256_storeu_ps(output + i, _mm256_mul_ps(silu, y));
        }
#endif
        for (; i < len; i++) {
            const size_t inputOffset = (size_t)i * 2;
            const float x = input[inputOffset];
            output[i] = (x / (1.0f + expf(-x))) *
                input[inputOffset + 1];
        }
    }

    void MultiThreadGemmAndCrossSwigluOp::Run() {
            // 1. 先执行 GEMM，结果写到 gateUpOutputData
            //    gateUpOutputData 已经偏移到了当前 NUMA 分片的起始列 (base)
            //    GEMM 写入列范围 [st, end)，ldc = k * sizeof(float) (整行步长)
            FastllmGemm(
                n, m, k,
                inputData, GetDataBytes(inputDataType, 1, m),
                weightData, GetDataBytes(weightDataType, 1, m),
                gateUpOutputData, GetDataBytes(gateUpOutputDataType, 1, k),
                st, end,
                inputDataType, weightDataType, gateUpOutputDataType
            );

            if (skipCrossSwiglu) {
                return;
            }

            // 2. 对 GEMM 刚写出的列做 CrossSwiglu
            //    gateUpOutputData 指向 gateUpOutput[..., globalColOffset]
            //    GEMM 写了 [st, end) 列，在全局坐标下是 [globalColOffset + st, globalColOffset + end)
            //    gateUpOutput 交错布局: gate[0], up[0], gate[1], up[1], ...
            //    swigluOutput 布局: [n, interDim], interDim = k / 2
            int interDim = k / 2;
            int localCols = end - st;         // 本次 GEMM 写出的列数
            int swigluCols = localCols / 2;   // 做完 swiglu 后的列数
            int globalSt = globalColOffset + st;
            int swigluColSt = globalSt / 2;   // swiglu 输出的列起始位置
            const long ldc = GetDataBytes(gateUpOutputDataType, 1, k);

            // Group-32 activation quantization needs a complete group to
            // calculate its scale and sum. NUMA MergeMOE aligns these tasks
            // to 64 gate/up columns, so compute each 32-value SwiGLU group in
            // a small L1-resident buffer and quantize it immediately. This
            // avoids both the large float intermediate write and a second
            // thread-pool pass over all routed experts.
            if (dstOutputData != nullptr &&
                    dstOutputDataType == DataType::INF_INT8_GROUP32) {
                constexpr int groupSize = 32;
                AssertInFastLLM(
                    globalSt % (groupSize * 2) == 0 &&
                        localCols % (groupSize * 2) == 0 &&
                        swigluColSt % groupSize == 0 &&
                        swigluCols % groupSize == 0,
                    "MultiThreadGemmAndCrossSwigluOp: INF_INT8_GROUP32 "
                    "destination requires group-aligned columns.\n");
                const size_t dstRowBytes = GetDataBytes(
                    dstOutputDataType, 1, interDim);
                const size_t dstGroupBytes = GetDataBytes(
                    DataType::INF_INT8_PERCHANNEL, 1, groupSize);
                for (int row = 0; row < n; row++) {
                    const float *gateUpRow = (const float*)(
                        gateUpOutputData + ldc * row);
                    const float *cur = gateUpRow + st;
                    uint8_t *dst = dstOutputData + dstRowBytes * row +
                        (size_t)(swigluColSt / groupSize) * dstGroupBytes;
                    for (int group = 0; group < swigluCols;
                         group += groupSize) {
                        alignas(32) float values[groupSize];
                        CrossSwigluFloat32Chunk(
                            cur + group * 2, groupSize, values);
                        ConvertFromFloat32(
                            dst, DataType::INF_INT8_GROUP32,
                            values, 1, groupSize);
                        dst += dstGroupBytes;
                    }
                }
                return;
            }

            for (int row = 0; row < n; row++) {
                // gateUpOutputData 已经偏移了 base 列，所以这里从 st 开始读
                float *gateUpRow = (float*)(gateUpOutputData + ldc * row);
                float *cur = gateUpRow + st;  // 指向 GEMM 写出的 [st] 位置
                float *out = swigluOutputData + (size_t)row * interDim + swigluColSt;
                CrossSwigluFloat32Chunk(cur, swigluCols, out);

                // 3. 如果指定了目标类型，立即将 swiglu 输出的列范围转换写入 dstOutputData
                if (dstOutputData != nullptr) {
                    size_t dstRowBytes = GetDataBytes(dstOutputDataType, 1, interDim);
                    if (dstOutputDataType == DataType::FLOAT32) {
                        memcpy((uint8_t*)dstOutputData + dstRowBytes * row + swigluColSt * sizeof(float),
                               out, swigluCols * sizeof(float));
                    } else if (dstOutputDataType == DataType::FLOAT16) {
                        Float32ToFloat16(out, (uint16_t*)((uint8_t*)dstOutputData + dstRowBytes * row) + swigluColSt, swigluCols);
                    } else if (dstOutputDataType == DataType::BFLOAT16) {
                        Float32ToBFloat16(out, (uint16_t*)((uint8_t*)dstOutputData + dstRowBytes * row) + swigluColSt, swigluCols);
                    } else {
                        // 对于需要整行量化的类型（INF_INT8 等），逐列分块无法处理，
                        // 需要外部在所有列写完后单独调用 ConvertFromFloat32
                        ErrorInFastLLM("MultiThreadGemmAndCrossSwigluOp: Unsupported destination type " + GetDataTypeName(dstOutputDataType));
                    }
                }
            }
    }
            
    void MultiThreadReduceBatchOp::Run() {
            for (int i = batch_st; i < batch_end; i++) {
                bool initialized = preZeroed;
                float *dst = lastOutput + (size_t)i * hidden_size;
                for (int expert_idx = 0; expert_idx < k; expert_idx++) {
                    int curPos = pos[i * k + expert_idx];
                    if (curPos == -1) continue; // 跳过无效位置
                    float weight = weights[curPos];
                    const float *src =
                        ((float*)downOutData) + (size_t)curPos * hidden_size;
                    if (!initialized) {
#ifdef __AVX2__
                        const __m256 vw = _mm256_set1_ps(weight);
                        int h = hidden_st;
                        for (; h + 7 < hidden_end; h += 8) {
                            _mm256_storeu_ps(dst + h,
                                _mm256_mul_ps(vw, _mm256_loadu_ps(src + h)));
                        }
                        for (; h < hidden_end; h++) {
                            dst[h] = weight * src[h];
                        }
#else
                        for (int h = hidden_st; h < hidden_end; h++) {
                            dst[h] = weight * src[h];
                        }
#endif
                        initialized = true;
                    } else {
#ifdef __AVX2__
                        const __m256 vw = _mm256_set1_ps(weight);
                        int h = hidden_st;
                        for (; h + 7 < hidden_end; h += 8) {
                            _mm256_storeu_ps(dst + h,
                                _mm256_fmadd_ps(vw, _mm256_loadu_ps(src + h),
                                    _mm256_loadu_ps(dst + h)));
                        }
                        for (; h < hidden_end; h++) {
                            dst[h] += weight * src[h];
                        }
#else
                        for (int h = hidden_st; h < hidden_end; h++) {
                            dst[h] += weight * src[h];
                        }
#endif
                    }
                }
                if (!initialized) {
#ifdef __AVX2__
                    const __m256 vzero = _mm256_setzero_ps();
                    int h = hidden_st;
                    for (; h + 7 < hidden_end; h += 8) {
                        _mm256_storeu_ps(dst + h, vzero);
                    }
                    for (; h < hidden_end; h++) {
                        dst[h] = 0.0f;
                    }
#else
                    for (int h = hidden_st; h < hidden_end; h++) {
                        dst[h] = 0.0f;
                    }
#endif
                }
            }
    }

    void MultiThreadReduceBatch(uint8_t *downOutData, DataType downOutDataType,
                    float *weights, float *lastOutput,
                    int *pos, int bsz, int k,
                    int hidden_size, bool preZeroed) {
        auto *pool = GetAlivePool();
        int threadNum = pool->threads.size();
        
        // 决定如何划分：尝试创建一个接近正方形的网格
        int batch_blocks = 1, hidden_blocks = threadNum;
        
        // 简单的启发式：如果bsz足够大，尝试在两个维度上划分
        if (bsz >= 4 && threadNum >= 4) {
            // 找到最佳的2D网格划分
            for (int b = 2; b <= std::min(bsz, threadNum); b++) {
                if (threadNum % b == 0) {
                    int h = threadNum / b;
                    if (h <= hidden_size) {
                        batch_blocks = b;
                        hidden_blocks = h;
                    }
                }
            }
        }
        
        std::vector<fastllm::MultiThreadReduceBatchOp*> ops;
        ops.reserve(threadNum);
        
        int batch_per = bsz / batch_blocks;
        int hidden_per = hidden_size / hidden_blocks;
        
        int op_idx = 0;
        for (int b = 0; b < batch_blocks; b++) {
            int batch_st = b * batch_per;
            int batch_end = (b == batch_blocks - 1) ? bsz : (b + 1) * batch_per;
            
            for (int h = 0; h < hidden_blocks; h++) {
                int hidden_st = h * hidden_per;
                int hidden_end = (h == hidden_blocks - 1) ? hidden_size : (h + 1) * hidden_per;
                
                ops.push_back(new MultiThreadReduceBatchOp(
                    downOutData, downOutDataType,
                    weights, lastOutput,
                    pos, bsz, k,
                    hidden_size,
                    batch_st, batch_end,
                    hidden_st, hidden_end,
                    preZeroed));
                
                pool->PushOp(op_idx++, ops.back());
            }
        }
        
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    extern void Float32ToInfInt8PerChannelAVX2(const float* srcData, uint8_t* dstData, size_t columns);
    void Float32ToInfInt8PerChannel(const float* srcData, uint8_t* dstData, size_t columns) {
        if (cpuInstructInfo.hasAVX2) {
            Float32ToInfInt8PerChannelAVX2(srcData, dstData, columns);
            return;
        }
        // 目标内存布局：
        // [int8 * columns] [float scale] [int sum]
        int8_t* quantizedData = (int8_t*)dstData;
        float* scalePtr = (float*)(dstData + columns);
        int* sumPtr = (int*)(dstData + columns + sizeof(float));
        
        // 1. 找到这一行的最大绝对值
        float maxAbs = 0.0f;
        for (size_t i = 0; i < columns; i++) {
            float absVal = std::abs(srcData[i]);
            if (absVal > maxAbs) {
                maxAbs = absVal;
            }
        }
        
        // 2. 计算scale（对称量化，范围是 -127 到 127）
        float scale;
        if (maxAbs > 0) {
            scale = maxAbs / 127.0f;
        } else {
            scale = 1.0f;  // 避免除零
        }
        
        // 3. 量化并计算sum
        int sum = 0;
        for (size_t i = 0; i < columns; i++) {
            // 量化: q = round(x / scale)
            int quantized = std::round(srcData[i] / scale);
            
            // 裁剪到int8范围 [-127, 127]（对称量化通常不使用-128）
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;
            
            quantizedData[i] = (int8_t)quantized;
            sum += quantized;
        }
        
        // 4. 存储scale和sum
        *scalePtr = scale;
        *sumPtr = sum;
    }

    void ConvertFromFloat32(void *dstData, DataType dstDataType, const float *floatData, size_t rows, size_t columns) {
        if (dstDataType == DataType::FLOAT32) {
            memcpy(dstData, floatData, rows * columns * sizeof(float));
        } else if (dstDataType == DataType::FLOAT16) {
            Float32ToFloat16((float*)floatData, (uint16_t*)dstData, rows * columns);
        } else if (dstDataType == DataType::BFLOAT16) {
            Float32ToBFloat16((float*)floatData, (uint16_t*)dstData, rows * columns);
        } else if (dstDataType == DataType::INF_INT8_PERCHANNEL) {
            size_t rowCount = GetDataBytes(dstDataType, 1, columns);
            for (int i = 0; i < rows; i++) {
                Float32ToInfInt8PerChannel (
                    (float*)floatData + i * columns, 
                    (uint8_t*)dstData + i * rowCount, 
                    columns
                );
            }
        } else if (dstDataType == DataType::INF_INT8_GROUP128 ||
                   dstDataType == DataType::INF_INT8_GROUP32) {
            size_t groupCnt = dstDataType == DataType::INF_INT8_GROUP128 ? 128 : 32;
            rows *= (columns / groupCnt);
            columns = groupCnt;
            size_t rowCount = GetDataBytes(INF_INT8_PERCHANNEL, 1, columns);
            for (int i = 0; i < rows; i++) {
                Float32ToInfInt8PerChannel (
                    (float*)floatData + i * columns, 
                    (uint8_t*)dstData + i * rowCount, 
                    columns
                );
            }
        } else if (dstDataType >= DataType::DATA_GGUF_FORMAT && dstDataType < DataType::DATA_GGUF_FORMAT_END) {
            auto ggmlType = (ggml_type)((int)dstDataType - (int)DataType::DATA_GGUF_FORMAT);
            size_t rowCount = ggml_row_size(ggmlType, columns);
            for (int i = 0; i < rows; i++) {
                iqk_quantize_row_q8_K (
                        (float*)floatData + i * columns, (uint8_t*)dstData + i * rowCount, columns, 
                        ggmlType, ggmlType
                );
            }
        } else {
            ErrorInFastLLM("ConvertFromFloat32 failed with type" + GetDataTypeName(dstDataType));
        }
    }

#ifdef __AVX2__
    namespace {
        inline __m256 LoadBFloat16AsFloat32(const uint16_t *source) {
            const __m128i packed =
                _mm_loadu_si128((const __m128i*)source);
            const __m256i expanded = _mm256_slli_epi32(
                _mm256_cvtepu16_epi32(packed), 16);
            return _mm256_castsi256_ps(expanded);
        }

        inline int HorizontalSumInt32x8(const __m256i value) {
            const __m128i sum128 = _mm_add_epi32(
                _mm256_castsi256_si128(value),
                _mm256_extractf128_si256(value, 1));
            const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            const __m128i sum64 = _mm_add_epi32(hi64, sum128);
            const __m128i hi32 = _mm_shuffle_epi32(
                sum64, _MM_SHUFFLE(2, 3, 0, 1));
            return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
        }

        inline float HorizontalMaxFloat32x8(__m256 value) {
            __m128 max4 = _mm_max_ps(
                _mm256_extractf128_ps(value, 1),
                _mm256_castps256_ps128(value));
            max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
            max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
            return _mm_cvtss_f32(max4);
        }

        template <bool q8K32>
        void QuantizeBFloat16RowQ8K(
                const uint16_t *source, block_q8_K *destination,
                size_t columns) {
            AssertInFastLLM(
                columns % QK_K == 0,
                "BF16 to Q8_K conversion requires a multiple of QK_K.");
            const size_t blocks = columns / QK_K;
            const __m256 signBit = _mm256_set1_ps(-0.0f);
            const __m256i permutation =
                _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

            for (size_t block = 0; block < blocks; block++) {
                const uint16_t *blockSource = source + block * QK_K;
                block_q8_K &output = destination[block];
                __m256 maxAbs = _mm256_setzero_ps();
                for (int offset = 0; offset < QK_K; offset += 8) {
                    const __m256 value =
                        LoadBFloat16AsFloat32(blockSource + offset);
                    maxAbs = _mm256_max_ps(
                        maxAbs, _mm256_andnot_ps(signBit, value));
                }

                const float maxScalar = HorizontalMaxFloat32x8(maxAbs);
                const float scale = maxScalar / 127.0f;
                const float inverseScale =
                    maxScalar != 0.0f ? 127.0f / maxScalar : 0.0f;
                output.d = scale;
                const __m256 multiplier =
                    _mm256_set1_ps(inverseScale);
                int blockSumInt32 = 0;
                float blockSumFloat32 = 0.0f;

                for (int group = 0; group < QK_K / 32; group++) {
                    const uint16_t *groupSource =
                        blockSource + group * 32;
                    __m256 value0 = _mm256_mul_ps(
                        multiplier,
                        LoadBFloat16AsFloat32(groupSource));
                    __m256 value1 = _mm256_mul_ps(
                        multiplier,
                        LoadBFloat16AsFloat32(groupSource + 8));
                    __m256 value2 = _mm256_mul_ps(
                        multiplier,
                        LoadBFloat16AsFloat32(groupSource + 16));
                    __m256 value3 = _mm256_mul_ps(
                        multiplier,
                        LoadBFloat16AsFloat32(groupSource + 24));
                    value0 = _mm256_round_ps(value0, _MM_ROUND_NEAREST);
                    value1 = _mm256_round_ps(value1, _MM_ROUND_NEAREST);
                    value2 = _mm256_round_ps(value2, _MM_ROUND_NEAREST);
                    value3 = _mm256_round_ps(value3, _MM_ROUND_NEAREST);
                    __m256i quant0 = _mm256_cvtps_epi32(value0);
                    __m256i quant1 = _mm256_cvtps_epi32(value1);
                    __m256i quant2 = _mm256_cvtps_epi32(value2);
                    __m256i quant3 = _mm256_cvtps_epi32(value3);

                    if constexpr (q8K32) {
                        const int sum = HorizontalSumInt32x8(
                            _mm256_add_epi32(
                                _mm256_add_epi32(quant0, quant1),
                                _mm256_add_epi32(quant2, quant3)));
                        float *sums = (float*)output.bsums;
                        sums[group] = scale * sum;
                        blockSumFloat32 += sums[group];
                    } else {
                        output.bsums[group * 2] =
                            HorizontalSumInt32x8(
                                _mm256_add_epi32(quant0, quant1));
                        output.bsums[group * 2 + 1] =
                            HorizontalSumInt32x8(
                                _mm256_add_epi32(quant2, quant3));
                        blockSumInt32 += output.bsums[group * 2] +
                                         output.bsums[group * 2 + 1];
                    }

                    quant0 = _mm256_packs_epi32(quant0, quant1);
                    quant2 = _mm256_packs_epi32(quant2, quant3);
                    quant0 = _mm256_packs_epi16(quant0, quant2);
                    quant0 = _mm256_permutevar8x32_epi32(
                        quant0, permutation);
                    _mm256_storeu_si256(
                        (__m256i*)(output.qs + group * 32), quant0);
                }
                output.sum = q8K32 ? blockSumFloat32 :
                             scale * blockSumInt32;
            }
        }

        bool TryConvertBFloat16ToQ8K(
                void *dstData, DataType dstDataType,
                const uint16_t *source, size_t rows, size_t columns) {
            if (dstDataType < DataType::DATA_GGUF_FORMAT ||
                dstDataType >= DataType::DATA_GGUF_FORMAT_END) {
                return false;
            }
            const ggml_type type = (ggml_type)(
                (int)dstDataType - (int)DataType::DATA_GGUF_FORMAT);
            if (type != GGML_TYPE_Q8_K && type != GGML_TYPE_Q8_K32) {
                return false;
            }
            const size_t rowBytes = GetDataBytes(
                dstDataType, 1, columns);
            for (size_t row = 0; row < rows; row++) {
                block_q8_K *destination = (block_q8_K*)(
                    (uint8_t*)dstData + row * rowBytes);
                if (type == GGML_TYPE_Q8_K32) {
                    QuantizeBFloat16RowQ8K<true>(
                        source + row * columns, destination, columns);
                } else {
                    QuantizeBFloat16RowQ8K<false>(
                        source + row * columns, destination, columns);
                }
            }
            return true;
        }
    }
#endif

    void ConvertFromBFloat16(
            void *dstData, DataType dstDataType,
            const uint16_t *bfloat16Data, size_t rows, size_t columns) {
        if (rows == 0 || columns == 0) {
            return;
        }
        if (dstDataType == DataType::BFLOAT16) {
            memcpy(dstData, bfloat16Data,
                   rows * columns * sizeof(uint16_t));
            return;
        }
        if (dstDataType == DataType::FLOAT32) {
            float *destination = (float*)dstData;
            size_t count = rows * columns;
            size_t index = 0;
#ifdef __AVX2__
            for (; index + 7 < count; index += 8) {
                __m128i packed = _mm_loadu_si128(
                    (const __m128i*)(bfloat16Data + index));
                _mm256_storeu_ps(destination + index,
                    _mm256_castsi256_ps(_mm256_slli_epi32(
                        _mm256_cvtepu16_epi32(packed), 16)));
            }
#endif
            for (; index < count; index++) {
                destination[index] =
                    BFloat16BitsToFloat32(bfloat16Data[index]);
            }
            return;
        }
#ifdef __AVX2__
        if (TryConvertBFloat16ToQ8K(
                dstData, dstDataType, bfloat16Data, rows, columns)) {
            return;
        }
#endif

        // Preserve support for every existing activation format. This uses
        // only one row of FLOAT32 scratch instead of materializing the full
        // routed-expert activation table.
        const size_t rowBytes = GetDataBytes(dstDataType, 1, columns);
        std::vector<float, alignedAllocator<float, 64>> floatRow(columns);
        for (size_t row = 0; row < rows; row++) {
            const uint16_t *source = bfloat16Data + row * columns;
            for (size_t column = 0; column < columns; column++) {
                floatRow[column] = BFloat16BitsToFloat32(source[column]);
            }
            ConvertFromFloat32(
                (uint8_t*)dstData + row * rowBytes, dstDataType,
                floatRow.data(), 1, columns);
        }
    }

    void MultiThreadConvertFromBFloat16Op::Run() {
        const size_t rowBytes = GetDataBytes(dstDataType, 1, columns);
        ConvertFromBFloat16(
            (uint8_t*)dstData + startRow * rowBytes, dstDataType,
            bfloat16Data + startRow * columns,
            endRow - startRow, columns);
    }

    void RunMultiThreadConvertFromBFloat16(
            void *dstData, DataType dstDataType,
            const uint16_t *bfloat16Data, size_t rows, size_t columns,
            AliveThreadPool *pool) {
        if (rows * columns < 10000 || pool == nullptr) {
            ConvertFromBFloat16(
                dstData, dstDataType, bfloat16Data, rows, columns);
            return;
        }

        const int firstThread =
            pool->curActivateThreadInterval.first;
        int threadCount = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        threadCount = std::min(threadCount, (int)rows);
        if (threadCount <= 1) {
            ConvertFromBFloat16(
                dstData, dstDataType, bfloat16Data, rows, columns);
            return;
        }

        const size_t rowsPerThread = rows / threadCount;
        size_t startRow = 0;
        std::vector<MultiThreadConvertFromBFloat16Op*> operations;
        operations.reserve(threadCount);
        for (int thread = 0; thread < threadCount; thread++) {
            const size_t endRow = thread == threadCount - 1 ? rows :
                                  startRow + rowsPerThread;
            operations.push_back(new MultiThreadConvertFromBFloat16Op(
                dstData, dstDataType, bfloat16Data, columns,
                startRow, endRow));
            startRow = endRow;
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->PushOp(firstThread + thread, operations[thread]);
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->Wait(firstThread + thread);
            delete operations[thread];
        }
    }

    void MultiThreadConvertFromFloat32Op::Run() {
            // 计算每行的字节大小
            size_t rowSize = GetDataBytes(dstDataType, 1, columns);
            
            // 调用原始函数处理指定行范围
            void *dstStart = (char*)dstData + startRow * rowSize;
            const float *srcStart = floatData + startRow * columns;
            size_t rowsToProcess = endRow - startRow;
            
            ConvertFromFloat32(dstStart, dstDataType, srcStart, rowsToProcess, columns);
    }

    // 对应的多线程运行函数
    void RunMultiThreadConvertFromFloat32(void *dstData, DataType dstDataType, 
                                                const float *floatData, size_t rows, 
                                                size_t columns, AliveThreadPool *pool) {
        // 如果数据量较小，直接单线程处理
        if (rows * columns < 10000) {
            ConvertFromFloat32(dstData, dstDataType, floatData, rows, columns);
            return;
        }
        
        int threadNum = pool->threads.size();
        threadNum = std::min(threadNum, (int)rows);  // 线程数不超过行数
        
        // 如果行数太少，减少线程数
        if (rows < threadNum) {
            ConvertFromFloat32(dstData, dstDataType, floatData, rows, columns);
            return;
        }
        
        size_t rowsPerThread = rows / threadNum;
        size_t curRow = 0;
        
        std::vector<MultiThreadConvertFromFloat32Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            size_t endRow = (i == threadNum - 1) ? rows : curRow + rowsPerThread;
            ops.push_back(new MultiThreadConvertFromFloat32Op(
                dstData, dstDataType, floatData, columns, curRow, endRow));
            curRow = endRow;
        }
        
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }
        
    void WorkStealingOp::Run() {
            // 首先执行自己的任务
            processOwnTasks();
            
            // 然后从其他线程偷取任务
            stealFromOthers();
            
            // 标记完成
            myState->completed.store(true, std::memory_order_release);
    }
        
    void WorkStealingOp::processOwnTasks() {
            while (true) {
                int taskId = myState->curr.fetch_add(1, std::memory_order_acq_rel);
                if (taskId >= myState->end) {
                    break;
                }
                if (taskId < myState->tasks.size()) {
                    myState->tasks[taskId]->Run();
                }
            }
    }
        
    void WorkStealingOp::stealFromOthers() {
            // 从当前线程开始，环形遍历其他线程
            for (int offset = 1; offset < totalThreads; offset++) {
                int targetId = (threadId + offset) % totalThreads;
                
                TaskState* otherState = (*allStates)[targetId];
                if (otherState == nullptr) continue;
                
                // 检查是否还有任务可偷
                while (true) {
                    int taskId = otherState->curr.fetch_add(1, std::memory_order_acq_rel);
                    if (taskId >= otherState->end) {
                        break;
                    }
                    if (taskId < otherState->tasks.size()) {
                        otherState->tasks[taskId]->Run();
                    }
                }
            }
    }

    // 重构的动态任务调度函数，支持work-stealing
    void DynamicScheduleTasks(std::vector<MultiThreadBaseOp*>& ops) {
        auto *pool = GetAlivePool();
        int numThreads = pool->threads.size(); // 假设线程池有获取线程数的方法
        
        // 创建任务状态数组
        using TaskState = typename WorkStealingOp::TaskState;
        std::vector<TaskState*> taskStates(numThreads, nullptr);
        
        // 为每个线程分配任务状态
        for (int i = 0; i < numThreads; i++) {
            taskStates[i] = new TaskState();
            taskStates[i]->curr.store(0, std::memory_order_relaxed);
            taskStates[i]->end = 0;
            taskStates[i]->completed.store(false, std::memory_order_relaxed);
        }
        
        // 分配任务到各个线程
        int totalOps = ops.size();
        if (totalOps > 0) {
            // 计算每个线程的任务数量
            int tasksPerThread = totalOps / numThreads;
            int remainingTasks = totalOps % numThreads;
            
            int taskIndex = 0;
            for (int i = 0; i < numThreads; i++) {
                int numTasks = tasksPerThread + (i < remainingTasks ? 1 : 0);
                
                if (numTasks > 0) {
                    // 分配任务到该线程
                    taskStates[i]->tasks.clear();
                    taskStates[i]->tasks.reserve(numTasks);
                    
                    for (int j = 0; j < numTasks && taskIndex < totalOps; j++) {
                        taskStates[i]->tasks.push_back(ops[taskIndex++]);
                    }
                    
                    taskStates[i]->curr.store(0, std::memory_order_relaxed);
                    taskStates[i]->end = taskStates[i]->tasks.size();
                } else {
                    taskStates[i]->end = 0;
                }
            }
        }
        
        // 创建work-stealing ops并提交到线程池
        std::vector<WorkStealingOp*> wsOps(numThreads);
        for (int i = 0; i < numThreads; i++) {
            wsOps[i] = new WorkStealingOp(
                i, &taskStates, taskStates[i], numThreads
            );
            
            pool->PushOp(i, wsOps[i]);
        }
        
        // 等待所有线程完成
        for (int i = 0; i < numThreads; i++) {
            pool->Wait(i);
        }
        
        // 清理资源
        for (int i = 0; i < numThreads; i++) {
            delete wsOps[i];
            delete taskStates[i];
        }
        
        // 删除原始ops
        for (auto* op : ops) {
            delete op;
        }
    }

    void MultiThreadRepackWeightsOp::Run() {
        for (int i = st; i < end; i++) {
            if (weights[i] != nullptr) {
                weights[i]->Repack();
            }
        }
    }

    void CpuMergeMOE::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuLinearOp());
 // auto ttt = std::chrono::system_clock::now();
 // std::vector <std::pair <std::string, float> > record;
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &index = *(datas.find("index")->second);
        Data &score = *(datas.find("score")->second);
        Data &w1 = *(datas.find("w1")->second);
        Data &w2 = *(datas.find("w2")->second);
        Data &w3 = *(datas.find("w3")->second);
        Data **weights = (Data**)(datas.find("weights")->second);
        Data **biass = (Data**)(datas.find("biass")->second);
        float sharedScale = floatParams.find("sharedScale") != floatParams.end() ? floatParams.find("sharedScale")->second : 1.0f;        
        float swigluLimit = floatParams.find("swigluLimit") != floatParams.end() ?
                            floatParams.find("swigluLimit")->second : 0.0f;
        bool deepSeekV4Mode = intParams.find("deepSeekV4Mode") != intParams.end() &&
                              intParams.find("deepSeekV4Mode")->second != 0;
        bool useGeglu = intParams.find("gateType") != intParams.end() &&
                        intParams.find("gateType")->second == (int)MoeGateGeglu;
        output.Allocate();

        // index: [n, topk], score: [n, topk]
        int n = index.dims[0];
        int topk = index.dims[1];
        int weightsBatch = intParams.find("weights___batch") != intParams.end() ? intParams.find("weights___batch")->second : (topk + 1) * 2;
        int routedExpertCount = std::max(0, weightsBatch / 2 - 1);
        auto normalizeExpertIdx = [&](int expertIdx) {
            if (routedExpertCount <= 0) {
                return 0;
            }
            return std::max(0, std::min(expertIdx, routedExpertCount - 1));
        };
        
        ToDataType(score, DataType::FLOAT32);
        int32_t *indexData = (int32_t*)index.cpuData;
        float *scoreData = (float*)score.cpuData;
        bool profileDetail = std::getenv("FASTLLM_PROFILE_DETAIL") != nullptr &&
                             (std::getenv("FASTLLM_PROFILE") != nullptr ||
                              std::getenv("FASTLLM_PROFILE_DEEPSEEKV4") != nullptr ||
                              std::getenv("FASTLLM_PROFILE_CPU_MOE") != nullptr);
        if (profileDetail && std::getenv("FASTLLM_PROFILE_CPU_MOE_BRANCH") != nullptr) {
            printf("[fastllm-profile-cpu-moe] branch input=%s weight=%s bs=%d topk=%d weights_batch=%d\n",
                   GetDataTypeName(input.dataType).c_str(), GetDataTypeName(weights[2]->dataType).c_str(),
                   input.dims.empty() ? -1 : input.dims[0], topk, weightsBatch);
            fflush(stdout);
        }
        double profileLast = CpuProfileNowMs();
        double profileQuantInMs = 0.0, profilePrepareMs = 0.0, profileGateMs = 0.0;
        double profileSwigluQuantMs = 0.0, profileDownMs = 0.0, profileReduceMs = 0.0, profileOutputMs = 0.0;
        int profileExpertCalls = 0;
        auto profileLap = [&](double &bucket) {
            if (!profileDetail) {
                return;
            }
            double now = CpuProfileNowMs();
            bucket += now - profileLast;
            profileLast = now;
        };

        if (weights[2]->dataType == DataType::DATA_GGUF_FORMAT && 
            !weights[2]->IsRepacked) {
            std::vector<Data*> repackWeights;
            std::set<Data*> uniqueWeights;
            for (int i = 0; i < weightsBatch; i++) {
                if (weights[i] != nullptr &&
                    weights[i]->dataType == DataType::DATA_GGUF_FORMAT &&
                    !weights[i]->IsRepacked &&
                    uniqueWeights.insert(weights[i]).second) {
                    repackWeights.push_back(weights[i]);
                }
            }
            int len = (int)repackWeights.size();
            if (len == 0) {
                profileLap(profilePrepareMs);
            } else {
                auto *pool = GetAlivePool();
                int threadNum = std::min((int)pool->threads.size(), len);
                int per = len / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadRepackWeightsOp*> ops;
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per;
                    if (i == threadNum - 1) {
                        end = len;
                    }
                    ops.push_back(new MultiThreadRepackWeightsOp(repackWeights.data(), cur, end));
                    cur = end;
                }
                for (int i = 0; i < ops.size(); i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < ops.size(); i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            }
        }

        if ((input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16) && 
                (weights[2]->dataType == DataType::INT4_GROUP 
                || weights[2]->dataType == DataType::INT4_NOZERO 
                || weights[2]->dataType == DataType::INT8) &&
            input.dims[0] < 32) {
            int permuteType = 1;
            if (weights[2]->dataType == DataType::INT8) {
                permuteType = 0;
            }
            
            int32_t *indexData = (int32_t*)index.cpuData;
            float *scoreData = (float*)score.cpuData;
            
            int outer = n;
            float *floatInput = (float*)input.cpuData;
            std::vector <float> vInputs;
            output.Allocate(0.0f);

            if (input.dataType == DataType::FLOAT16) {
                int len = input.Count(0);
                vInputs.resize(len);
                for (int i = 0; i < len; i++) {
                    vInputs[i] = fp16tofp32.dict[((uint16_t*)input.cpuData)[i]];
                }
                floatInput = vInputs.data();
            }
            if (profileDetail) {
                profileLast = CpuProfileNowMs();
            }
            
            for (int o = 0; o < outer; o++) {
                std::vector <std::pair <int, float> > v;
                for (int j = 0; j < topk; j++) {
                    // index 存储的是专家索引（从0开始），需要+1因为0表示shared expert
                    int expertIdx = normalizeExpertIdx(indexData[o * topk + j]);
                    float expertScore = scoreData[o * topk + j];
                    v.push_back(std::make_pair(expertIdx + 1, expertScore));
                }
                if (weights[0] != nullptr) {
                    v.push_back(std::make_pair(0, sharedScale));
                }
                int n = input.dims[0], m = input.dims[1];
                int group = weights[2]->group, groupCnt = weights[2]->groupCnt;
                if (weights[2]->dataType != DataType::INT4_GROUP) {
                    group = 1;
                    groupCnt = m;
                }
                float *inputData = floatInput + o * m;

                std::vector<LowBitConfig> &inputConfigs = moeIntSingleVarManager.inputConfigs;
                std::vector<uint8_t> &uinput = moeIntSingleVarManager.uinput;
                std::vector <float> &inputSums = moeIntSingleVarManager.inputSums;
                std::vector <float> &iscales = moeIntSingleVarManager.iscales;
                std::vector <float> &izeros = moeIntSingleVarManager.izeros;
// record.push_back(std::make_pair("before OnlineQuantization", GetSpan(ttt, std::chrono::system_clock::now())));
                OnlineQuantization(inputData, uinput, inputConfigs, 1, m, group, groupCnt, 
                                    inputSums, iscales, izeros, permuteType);
                profileLap(profileQuantInMs);
// record.push_back(std::make_pair("OnlineQuantization", GetSpan(ttt, std::chrono::system_clock::now())));
                std::vector <std::vector <float> > &middles = moeIntSingleVarManager.middles;
                std::vector <std::vector <float> > &results = moeIntSingleVarManager.results;
                middles.resize(v.size());
                results.resize(v.size());
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    weights[idx * 2]->CalcWeightSum();
                    weights[idx * 2 + 1]->CalcWeightSum();
                }
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    middles[j].resize(weights[idx * 2]->dims[0]);
                    results[j].resize(weights[idx * 2 + 1]->dims[0]);
                }
                profileExpertCalls += (int)v.size();
                std::vector<fastllm::MultiThreadBaseOp*> ops;
                auto *pool = GetAlivePool();
                int threads = pool->threads.size();
                ops.resize(threads);

                std::vector <std::vector <LowBitConfig> > &inputConfigsDown = moeIntSingleVarManager.inputConfigsDown;
                std::vector <std::vector <uint8_t> > &uinputsDown = moeIntSingleVarManager.uinputsDown;
                std::vector <std::vector <float> > &inputSumsDown = moeIntSingleVarManager.inputSumsDown;
                std::vector <std::vector <float> > &iscalesDown = moeIntSingleVarManager.iscalesDown;
                std::vector <std::vector <float> > &izerosDown = moeIntSingleVarManager.izerosDown;
                inputConfigsDown.resize(v.size());
                uinputsDown.resize(v.size());
                inputSumsDown.resize(v.size());
                iscalesDown.resize(v.size());
                izerosDown.resize(v.size());
                profileLap(profilePrepareMs);
 // record.push_back(std::make_pair("prepare", GetSpan(ttt, std::chrono::system_clock::now())));
                for (int st = 0; st < v.size(); st++) {
                    int k = weights[v[st].first * 2]->dims[0];
                    int end = st, selSum = 1; // 一共处理selSum * k个输出

                    int curSum = 1;
                    for (int l = st + 1; l < v.size(); l++) {
                        int curK = weights[v[l].first * 2]->dims[0];
                        if (curK % k != 0) {
                            break;
                        }
                        curSum += (curK / k);
                        if (threads % curSum == 0) {
                            end = l;
                            selSum = curSum;
                        }
                    }
                    int base = threads / selSum;
                    int threadSt = 0;
// float xxx = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        Data *weight = weights[idx * 2];
                        uint8_t *weightData = (uint8_t *) weight->cpuData;
                        float *outputData = middles[l].data();
                        float *biasData = nullptr;
                        int curK = weight->dims[0];
                        int curThread = (curK / k) * base;
// xxx += m * curK;
                        if (weight->dataType == DataType::INT8) {
                            LaunchLinearInt8Int8(uinput.data(), weightData, outputData, 1, m, curK,
                                                weight->weightSum.data(), weight->zeros.data(), weight->scales.data(), biasData, 
                                                inputSums.data(), iscales.data(), izeros.data(), 
                                                ops, pool, threadSt, curThread);
                        } else {
                            MultiplyInt4GroupMultiThreadLaunch(uinput.data(), weightData, outputData, 1, m, curK,
                                                weight->weightSum.data(), weight->mins.data(), weight->scales.data(), biasData, 
                                                inputSums, iscales, izeros,
                                                inputConfigs, threadSt, curThread, group, groupCnt, ops, pool);
                        }
                        threadSt += curThread;
                    }
                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                        ops[j] = nullptr;
                    }
                    profileLap(profileGateMs);
// record.push_back(std::make_pair("mul0", GetSpan(ttt, std::chrono::system_clock::now())));
// float spend = record.back().second - record[record.size() - 2].second;
//printf("speed = %f gops.\n", xxx / spend / 1e9);
                    // swiglu
                    threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int spatial = weights[idx * 2]->dims[0], mid = spatial / 2;
                        float *outputData = middles[l].data();
                        int curK = weights[idx * 2]->dims[0];
                        ops[l - st] = new fastllm::MultiThreadMultiOps();
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(
                            useGeglu ? (fastllm::MultiThreadBaseOp*)new fastllm::MultiThreadGegluOp(outputData, mid, mid, outputData, 1, spatial, spatial)
                                     : (fastllm::MultiThreadBaseOp*)new fastllm::MultiThreadSwigluOp(outputData, mid, mid, outputData, 1, spatial, spatial));
                        Data *weightDown = weights[idx * 2 + 1];
                        int groupDown = weightDown->group, groupCntDown = weightDown->groupCnt;
                        if (weightDown->dataType != DataType::INT4_GROUP) {
                            groupDown = 1;
                            groupCntDown = mid;
                        }
                        auto &inputConfigs = inputConfigsDown[l];
                        auto &inputSums = inputSumsDown[l];
                        auto &iscales = iscalesDown[l];
                        auto &izeros = izerosDown[l];
                        auto &uinputDown = uinputsDown[l];
                        inputConfigs.resize(n * groupDown);
                        uinputDown.resize(n * mid);   
                        inputSums.resize(n * groupDown);
                        iscales.resize(n * groupDown);
                        izeros.resize(n * groupDown);

                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new MultiThreadOnlineQuantizationOp(
                                    middles[l].data(), uinputDown.data(), inputConfigs.data(),
                                    1, mid, groupDown, groupCntDown,
                                    inputSums.data(), iscales.data(), izeros.data(), permuteType));
                        pool->PushOp(l - st, ops[l - st]);
                    }
                    for (int l = st; l <= end; l++) {
                        pool->Wait(l - st);
                        delete ops[l - st];
                        ops[l - st] = nullptr;
                    }
                    profileLap(profileSwigluQuantMs);
// record.push_back(std::make_pair("swiglu", GetSpan(ttt, std::chrono::system_clock::now())));
 // record.push_back(std::make_pair("quant", GetSpan(ttt, std::chrono::system_clock::now())));
                    threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int mid = weights[idx * 2]->dims[0] / 2;
                        int curK = weights[idx * 2]->dims[0];
                        Data *weightDown = weights[idx * 2 + 1];
                        int groupDown = weightDown->group, groupCntDown = weightDown->groupCnt;
                        auto &inputConfigs = inputConfigsDown[l];
                        auto &inputSums = inputSumsDown[l];
                        auto &iscales = iscalesDown[l];
                        auto &izeros = izerosDown[l];
                        auto &uinputDown = uinputsDown[l];
                        int curThread = (curK / k) * base;
                        if (weightDown->dataType != DataType::INT4_GROUP) {
                            groupDown = 1;
                            groupCntDown = mid;
                        }
                        if (weightDown->dataType == DataType::INT8) {
                            LaunchLinearInt8Int8(uinputDown.data(), (uint8_t*)weightDown->cpuData, results[l].data(), 1, mid, m,
                                                    weightDown->weightSum.data(), weightDown->zeros.data(), weightDown->scales.data(), nullptr, 
                                                    inputSums.data(), iscales.data(), izeros.data(),
                                                    ops, pool, threadSt, curThread);
                        } else {
                            MultiplyInt4GroupMultiThreadLaunch(uinputDown.data(), (uint8_t*)weightDown->cpuData, results[l].data(), 1, mid, m,
                                                    weightDown->weightSum.data(), weightDown->mins.data(), weightDown->scales.data(), nullptr, 
                                                    inputSums, iscales, izeros,
                                                    inputConfigs, threadSt, curThread, groupDown, groupCntDown, ops, pool);
                        }
                        threadSt += curThread;               
                    }

                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                        ops[j] = nullptr;
                    }
                    profileLap(profileDownMs);
 // record.push_back(std::make_pair("mul1", GetSpan(ttt, std::chrono::system_clock::now())));
                    st = end;
                }
// record.push_back(std::make_pair("finish", GetSpan(ttt, std::chrono::system_clock::now())));
                float *fLastOutput = ((float*)output.cpuData) + o * m;
                std::vector <float> tempOutput;
                if (output.dataType == DataType::FLOAT16) {
                    tempOutput.resize(m, 0);
                    fLastOutput = tempOutput.data();
                }
/*
                std::vector <float> vv;
                vv.resize(v.size());
                for (int i = 0; i < v.size(); i++) {
                    vv[i] = v[i].second;
                }
                RunMultiThreadReduce (
                    (int)vv.size(), results.data(), vv.data(), fLastOutput, 
                    nullptr, m, pool
                );
*/

                for (int j = 0; j < v.size(); j++) {
                    float value = v[j].second;
                    float *curOutput = (float*)results[j].data();
                    int i = 0;
#ifdef __AVX2__
                    __m256 value_vec = _mm256_set1_ps(value);

                    // 每次处理 8 个浮点数（AVX2 寄存器可以容纳 8 个 float）
                    for (; i <= m - 8; i += 8) {
                        // 加载 curOutput 的 8 个浮点数
                        __m256 curOutput_vec = _mm256_loadu_ps(&curOutput[i]);

                        // 加载 fLastOutput 的 8 个浮点数
                        __m256 fLastOutput_vec = _mm256_loadu_ps(&fLastOutput[i]);

                        // 计算 curOutput * value
                        __m256 result_vec = _mm256_mul_ps(curOutput_vec, value_vec);

                        // 累加到 fLastOutput
                        fLastOutput_vec = _mm256_add_ps(fLastOutput_vec, result_vec);

                        // 将结果存回 fLastOutput
                        _mm256_storeu_ps(&fLastOutput[i], fLastOutput_vec);
                    }
#endif
                    // 处理剩余的不足 8 个的元素
                    for (; i < m; i++) {
                        fLastOutput[i] += curOutput[i] * value;
                    }
                }
                profileLap(profileReduceMs);
 // record.push_back(std::make_pair("get f32 output", GetSpan(ttt, std::chrono::system_clock::now())));
                if (output.dataType == DataType::FLOAT16) {
                    Float32ToFloat16(tempOutput.data(), ((uint16_t*)output.cpuData) + o * m, m);
                }
                profileLap(profileOutputMs);
// record.push_back(std::make_pair("finish output", GetSpan(ttt, std::chrono::system_clock::now())));
// for (int i = 0; i < record.size(); i++) {
    //printf("%s spend %f s.\n", record[i].first.c_str(), record[i].second);
// }
            }
            if (profileDetail) {
                double total = profileQuantInMs + profilePrepareMs + profileGateMs +
                               profileSwigluQuantMs + profileDownMs + profileReduceMs + profileOutputMs;
                printf("[fastllm-profile-cpu-moe] lowbit_small outer=%d topk=%d experts=%d quant_in=%.3f prepare=%.3f gate=%.3f swiglu_quant=%.3f down=%.3f reduce=%.3f output=%.3f total=%.3f\n",
                       outer, topk, profileExpertCalls, profileQuantInMs, profilePrepareMs, profileGateMs,
                       profileSwigluQuantMs, profileDownMs, profileReduceMs, profileOutputMs, total);
                fflush(stdout);
            }
        } else if ((input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16 ||
                    input.dataType == DataType::BFLOAT16) &&
                (weights[2]->dataType == DataType::DATA_GGUF_FORMAT)) {
            int outer = n;
            float *floatInput = (float*)input.cpuData;
            std::vector <float> vInputs;
            output.Allocate(0.0f);

            if (input.dataType == DataType::FLOAT16) {
                int len = input.Count(0);
                vInputs.resize(len);
                for (int i = 0; i < len; i++) {
                    vInputs[i] = fp16tofp32.dict[((uint16_t*)input.cpuData)[i]];
                }
                floatInput = vInputs.data();
            } else if (input.dataType == DataType::BFLOAT16) {
                int len = input.Count(0);
                vInputs.resize(len);
                BFloat16ToFloat32((uint16_t*)input.cpuData, vInputs.data(), len);
                floatInput = vInputs.data();
            }
            for (int o = 0; o < outer; o++) {
                std::vector <std::pair <int, float> > v;
                for (int j = 0; j < topk; j++) {
                    int expertIdx = normalizeExpertIdx(indexData[o * topk + j]);
                    float expertScore = scoreData[o * topk + j];
                    v.push_back(std::make_pair(expertIdx + 1, expertScore));
                }
                if (weights[0] != nullptr) {
                    v.push_back(std::make_pair(0, sharedScale));
                }
                int nRow = input.dims[0], m = input.dims[1];
                float *inputData = floatInput + o * m;

                std::vector <std::vector <float> > &middles = moeIntSingleVarManager.middles;
                std::vector <std::vector <float> > &results = moeIntSingleVarManager.results;
                middles.resize(v.size());
                results.resize(v.size());
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    middles[j].resize(weights[idx * 2]->dims[0]);
                    results[j].resize(weights[idx * 2 + 1]->dims[0]);
                }
                std::vector<fastllm::MultiThreadBaseOp*> ops;
                auto *pool = GetAlivePool();
                int threads = pool->threads.size();
                ops.resize(threads);

                std::vector <std::vector <uint8_t> > &q8kInputsDown = moeIntSingleVarManager.uinputsDown;
                q8kInputsDown.resize(v.size());
                std::vector <std::vector <uint8_t> > &q8kInputsGate = moeIntSingleVarManager.uinputsGate;
                std::vector <int> q8kInputGateIds(v.size());
                std::vector <int> q8kInputGateTypes;
                q8kInputsGate.clear();
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    ggml_type gateType = (ggml_type)weights[idx * 2]->ggmlType;
                    int id = -1;
                    for (int t = 0; t < q8kInputGateTypes.size(); t++) {
                        if (q8kInputGateTypes[t] == gateType) {
                            id = t;
                            break;
                        }
                    }
                    if (id < 0) {
                        id = (int)q8kInputsGate.size();
                        q8kInputGateTypes.push_back(gateType);
                        q8kInputsGate.push_back(std::vector <uint8_t>());
                        q8kInputsGate.back().resize(ggml_row_size(ggml_type_vec_dot_type(gateType), m));
                        iqk_quantize_row_q8_K (
                            inputData, q8kInputsGate.back().data(), m,
                            ggml_type_vec_dot_type(gateType), gateType
                        );
                    }
                    q8kInputGateIds[j] = id;
                }

                for (int st = 0; st < v.size(); st++) {
                    int k = weights[v[st].first * 2]->dims[0];
                    int end = st, selSum = 1; // 一共处理selSum * k个输出

                    int curSum = 1;
                    for (int l = st + 1; l < v.size(); l++) {
                        int curK = weights[v[l].first * 2]->dims[0];
                        if (curK % k != 0) {
                            break;
                        }
                        curSum += (curK / k);
                        if (threads % curSum == 0) {
                            end = l;
                            selSum = curSum;
                        }
                    }
                    int base = threads / selSum;
                    int threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        Data *weight = weights[idx * 2];
                        uint8_t *weightData = (uint8_t *) weight->cpuData;
                        float *outputData = middles[l].data();
                        float *biasData = nullptr;
                        int curK = weight->dims[0];
                        int curThread = (curK / k) * base;
                        
                        LaunchLinearQ8KGGUF(q8kInputsGate[q8kInputGateIds[l]].data(), weightData,
                            outputData, biasData, weight, 1, m, curK, ops, pool, threadSt, curThread);
                        threadSt += curThread;
                    }
                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                        ops[j] = nullptr;
                    }

                    // swiglu
                    threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int spatial = weights[idx * 2]->dims[0], mid = spatial / 2;
                        float *outputData = middles[l].data();
                        int curK = weights[idx * 2]->dims[0];
                        auto &uinputDown = q8kInputsDown[l];
                        int rowCount = mid / QK_K; // 每行有多少个block
                        uinputDown.resize(ggml_row_size(ggml_type_vec_dot_type((ggml_type)weights[idx * 2 + 1]->ggmlType), mid));

                        ops[l - st] = new fastllm::MultiThreadMultiOps();
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(
                            useGeglu ? (fastllm::MultiThreadBaseOp*)new fastllm::MultiThreadGegluOp(outputData, mid, mid, outputData, 1, spatial, spatial)
                                     : (fastllm::MultiThreadBaseOp*)new fastllm::MultiThreadSwigluOp(outputData, mid, mid, outputData, 1, spatial, spatial)); 
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadFloat32ToQ8KOp(middles[l].data(), (uint8_t*)uinputDown.data(), mid, (ggml_type)weights[idx * 2 + 1]->ggmlType));
                        /* ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new MultiThreadOnlineQuantizationOp(
                                    middles[l].data(), uinputDown.data(), inputConfigs.data(),
                                    1, mid, groupDown, groupCntDown,
                                    inputSums.data(), iscales.data(), izeros.data(), permuteType)); */
                        pool->PushOp(l - st, ops[l - st]);
                    }
                    for (int l = st; l <= end; l++) {
                        pool->Wait(l - st);
                        delete ops[l - st];
                        ops[l - st] = nullptr;
                    }
/*
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int spatial = weights[idx * 2]->dims[0], mid = spatial / 2;
                        
                        auto &uinputDown = q8kInputsDown[l];
                        int rowCount = mid / QK_K; // 每行有多少个block

                        uinputDown.resize(ggml_row_size(ggml_type_vec_dot_type((ggml_type)weights[idx * 2 + 1]->ggmlType), m));
                        iqk_quantize_row_q8_K (
                                middles[l].data(), uinputDown.data(), mid, 
                                ggml_type_vec_dot_type((ggml_type)weights[idx * 2 + 1]->ggmlType),
                                (ggml_type)weights[idx * 2 + 1]->ggmlType
                        );
                    }
*/
                    threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int mid = weights[idx * 2]->dims[0] / 2;
                        int curK = weights[idx * 2]->dims[0];
                        Data *weightDown = weights[idx * 2 + 1];

                        auto &uinputDown = q8kInputsDown[l];
                        int curThread = (curK / k) * base;

                        LaunchLinearQ8KGGUF(uinputDown.data(), weightDown->cpuData, results[l].data(), nullptr, weightDown, 
                            1, mid, m, ops, pool, threadSt, curThread);
                        threadSt += curThread;               
                    }

                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                        ops[j] = nullptr;
                    }
                    st = end;
                }

                float *fLastOutput = ((float*)output.cpuData) + o * m;
                std::vector <float> tempOutput;
                if (output.dataType == DataType::FLOAT16 || output.dataType == DataType::BFLOAT16) {
                    tempOutput.resize(m, 0);
                    fLastOutput = tempOutput.data();
                }
                
                for (int j = 0; j < v.size(); j++) {
                    float value = v[j].second;
                    float *curOutput = (float*)results[j].data();
                    int i = 0;
#ifdef __AVX2__
                    __m256 value_vec = _mm256_set1_ps(value);

                    // 每次处理 8 个浮点数（AVX2 寄存器可以容纳 8 个 float）
                    for (; i <= m - 8; i += 8) {
                        // 加载 curOutput 的 8 个浮点数
                        __m256 curOutput_vec = _mm256_loadu_ps(&curOutput[i]);

                        // 加载 fLastOutput 的 8 个浮点数
                        __m256 fLastOutput_vec = _mm256_loadu_ps(&fLastOutput[i]);

                        // 计算 curOutput * value
                        __m256 result_vec = _mm256_mul_ps(curOutput_vec, value_vec);

                        // 累加到 fLastOutput
                        fLastOutput_vec = _mm256_add_ps(fLastOutput_vec, result_vec);

                        // 将结果存回 fLastOutput
                        _mm256_storeu_ps(&fLastOutput[i], fLastOutput_vec);
                    }
#endif
                    // 处理剩余的不足 8 个的元素
                    for (; i < m; i++) {
                        fLastOutput[i] += curOutput[i] * value;
                    }
                }
 // record.push_back(std::make_pair("get f32 output", GetSpan(ttt, std::chrono::system_clock::now())));
                if (output.dataType == DataType::FLOAT16) {
                    Float32ToFloat16(tempOutput.data(), ((uint16_t*)output.cpuData) + o * m, m);
                } else if (output.dataType == DataType::BFLOAT16) {
                    Float32ToBFloat16(tempOutput.data(), ((uint16_t*)output.cpuData) + o * m, m);
                }
// record.push_back(std::make_pair("finish output", GetSpan(ttt, std::chrono::system_clock::now())));
// for (int i = 0; i < record.size(); i++) {
    //printf("%s spend %f s.\n", record[i].first.c_str(), record[i].second);
// }
            }
        } else if ((input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16) && 
                (weights[2]->dataType == DataType::FLOAT16) &&
                input.dims[0] < 32) {
            int outer = n;
            float *floatInput = (float*)input.cpuData;
            std::vector <float> vInputs;
            output.Allocate(0.0f);
            double fp16PrepareMs = 0.0, fp16GateMs = 0.0, fp16SwigluMs = 0.0;
            double fp16DownMs = 0.0, fp16ReduceMs = 0.0, fp16OutputMs = 0.0;
            double fp16RoutedGateMs = 0.0, fp16SharedGateMs = 0.0;
            double fp16RoutedDownMs = 0.0, fp16SharedDownMs = 0.0;
            int fp16ExpertCalls = 0;

            if (input.dataType == DataType::FLOAT16) {
                int len = input.Count(0);
                vInputs.resize(len);
                for (int i = 0; i < len; i++) {
                    vInputs[i] = fp16tofp32.dict[((uint16_t*)input.cpuData)[i]];
                }
                floatInput = vInputs.data();
            }
            if (profileDetail) {
                profileLast = CpuProfileNowMs();
            }
            for (int o = 0; o < outer; o++) {
                std::vector <std::pair <int, float> > v;
                for (int j = 0; j < topk; j++) {
                    int expertIdx = normalizeExpertIdx(indexData[o * topk + j]);
                    float expertScore = scoreData[o * topk + j];
                    v.push_back(std::make_pair(expertIdx + 1, expertScore));
                }
                if (weights[0] != nullptr) {
                    v.push_back(std::make_pair(0, sharedScale));
                }
                int m = input.dims[1];
                float *inputData = floatInput + o * m;
            
                auto &middles = moeFloatSingleVarManager.middles;
                auto &results = moeFloatSingleVarManager.results;
                middles.resize(v.size());
                results.resize(v.size());
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    middles[j].resize(weights[idx * 2]->dims[0]);
                    results[j].resize(weights[idx * 2 + 1]->dims[0]);
                }
                fp16ExpertCalls += (int)v.size();
                profileLap(fp16PrepareMs);
                std::vector<fastllm::MultiThreadBaseOp*> ops;
                auto *pool = GetAlivePool();
                int threads = pool->threads.size();
                ops.resize(threads);

                for (int st = 0; st < v.size(); st++) {
                    int k = weights[v[st].first * 2]->dims[0];
                    int end = st, selSum = 1; // 一共处理selSum * k个输出

                    int curSum = 1;
                    for (int l = st + 1; l < v.size(); l++) {
                        int curK = weights[v[l].first * 2]->dims[0];
                        if (curK % k != 0) {
                            break;
                        }
                        curSum += (curK / k);
                        if (threads % curSum == 0) {
                            end = l;
                            selSum = curSum;
                        }
                    }
                    int base = threads / selSum;
                    int threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        Data *weight = weights[idx * 2];
                        float *outputData = middles[l].data();
                        float *biasData = nullptr;
                        int curK = weight->dims[0];
                        int curThread = (curK / k) * base;
                        LaunchLinearFloat32Float16(inputData, *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
                        threadSt += curThread;
                    }
                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                        ops[j] = nullptr;
                    }
                    double beforeGateLap = profileLast;
                    profileLap(fp16GateMs);
                    if (profileDetail) {
                        double span = profileLast - beforeGateLap;
                        bool onlyShared = true;
                        for (int l = st; l <= end; l++) {
                            if (v[l].first != 0) {
                                onlyShared = false;
                                break;
                            }
                        }
                        if (onlyShared) {
                            fp16SharedGateMs += span;
                        } else {
                            fp16RoutedGateMs += span;
                        }
                    }

                    // swiglu
                    threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int spatial = weights[idx * 2]->dims[0], mid = spatial / 2;
                        float *outputData = middles[l].data();
                        int curK = weights[idx * 2]->dims[0];
                        ops[l - st] = new fastllm::MultiThreadMultiOps();
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(
                            useGeglu ? (fastllm::MultiThreadBaseOp*)new fastllm::MultiThreadGegluOp(outputData, mid, mid, outputData, 1, spatial, spatial)
                                     : (fastllm::MultiThreadBaseOp*)new fastllm::MultiThreadSwigluOp(outputData, mid, mid, outputData, 1, spatial, spatial));
                        pool->PushOp(l - st, ops[l - st]);
                    }
                    for (int l = st; l <= end; l++) {
                        pool->Wait(l - st);
                        delete ops[l - st];
                        ops[l - st] = nullptr;
                    }
                    profileLap(fp16SwigluMs);

                    threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int mid = weights[idx * 2]->dims[0] / 2;
                        int curK = weights[idx * 2]->dims[0];
                        Data *weightDown = weights[idx * 2 + 1];
                        int curThread = (curK / k) * base;
                        LaunchLinearFloat32Float16((float*)middles[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
                        threadSt += curThread;               
                    }

                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                        ops[j] = nullptr;
                    }
                    double beforeDownLap = profileLast;
                    profileLap(fp16DownMs);
                    if (profileDetail) {
                        double span = profileLast - beforeDownLap;
                        bool onlyShared = true;
                        for (int l = st; l <= end; l++) {
                            if (v[l].first != 0) {
                                onlyShared = false;
                                break;
                            }
                        }
                        if (onlyShared) {
                            fp16SharedDownMs += span;
                        } else {
                            fp16RoutedDownMs += span;
                        }
                    }
                    st = end;
                }
                float *fLastOutput = ((float*)output.cpuData) + o * m;
                std::vector <float> tempOutput;
                if (output.dataType == DataType::FLOAT16) {
                    tempOutput.resize(m, 0);
                    fLastOutput = tempOutput.data();
                }
                for (int j = 0; j < v.size(); j++) {
                    float value = v[j].second;
                    float *curOutput = (float*)results[j].data();
                    int i = 0;
#ifdef __AVX2__
                    __m256 value_vec = _mm256_set1_ps(value);

                    // 每次处理 8 个浮点数（AVX2 寄存器可以容纳 8 个 float）
                    for (; i <= m - 8; i += 8) {
                        // 加载 curOutput 的 8 个浮点数
                        __m256 curOutput_vec = _mm256_loadu_ps(&curOutput[i]);

                        // 加载 fLastOutput 的 8 个浮点数
                        __m256 fLastOutput_vec = _mm256_loadu_ps(&fLastOutput[i]);

                        // 计算 curOutput * value
                        __m256 result_vec = _mm256_mul_ps(curOutput_vec, value_vec);

                        // 累加到 fLastOutput
                        fLastOutput_vec = _mm256_add_ps(fLastOutput_vec, result_vec);

                        // 将结果存回 fLastOutput
                        _mm256_storeu_ps(&fLastOutput[i], fLastOutput_vec);
                    }
#endif
                    // 处理剩余的不足 8 个的元素
                    for (; i < m; i++) {
                        fLastOutput[i] += curOutput[i] * value;
                    }
                }
                profileLap(fp16ReduceMs);
                if (output.dataType == DataType::FLOAT16) {
                    Float32ToFloat16(tempOutput.data(), ((uint16_t*)output.cpuData) + o * m, m);
                }
                profileLap(fp16OutputMs);
            }
            if (profileDetail) {
                double total = fp16PrepareMs + fp16GateMs + fp16SwigluMs + fp16DownMs + fp16ReduceMs + fp16OutputMs;
                printf("[fastllm-profile-cpu-moe] fp16_small outer=%d topk=%d experts=%d prepare=%.3f gate=%.3f gate_routed=%.3f gate_shared=%.3f swiglu=%.3f down=%.3f down_routed=%.3f down_shared=%.3f reduce=%.3f output=%.3f total=%.3f\n",
                       outer, topk, fp16ExpertCalls, fp16PrepareMs, fp16GateMs,
                       fp16RoutedGateMs, fp16SharedGateMs, fp16SwigluMs, fp16DownMs,
                       fp16RoutedDownMs, fp16SharedDownMs, fp16ReduceMs, fp16OutputMs, total);
                fflush(stdout);
            }
        } else if ((input.dataType == DataType::FLOAT32 || input.dataType == DataType::BFLOAT16) &&
                (weights[2]->dataType == DataType::FP8_E4M3 ||
                 weights[2]->dataType == DataType::NVFP4 ||
                 weights[2]->dataType == DataType::BFLOAT16) &&
                input.dims[0] < 32) {
            int outer = n;
            float *floatInput = input.dataType == DataType::FLOAT32 ? (float*)input.cpuData : nullptr;
            output.Allocate(0.0f);

            for (int o = 0; o < outer; o++) {
                std::vector <std::pair <int, float> > v;
                for (int j = 0; j < topk; j++) {
                    int expertIdx = normalizeExpertIdx(indexData[o * topk + j]);
                    float expertScore = scoreData[o * topk + j];
                    v.push_back(std::make_pair(expertIdx + 1, expertScore));
                }
                if (weights[0] != nullptr) {
                    v.push_back(std::make_pair(0, sharedScale));
                }
                if (deepSeekV4Mode) {
                    std::stable_sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
                        if (a.first == 0 || b.first == 0) {
                            return b.first == 0 && a.first != 0;
                        }
                        return a.first < b.first;
                    });
                }
                int m = input.dims[1];
                float *inputData = floatInput == nullptr ? nullptr : floatInput + (uint64_t)o * m;
                auto &bf16Input = moeFloatSingleVarManager.bf16Input;
                uint16_t *bf16InputData = nullptr;
                std::vector<float> floatRow;
                if (input.dataType == DataType::BFLOAT16) {
                    bf16InputData = ((uint16_t*)input.cpuData) + (uint64_t)o * m;
                    if (!cpuInstructInfo.hasAVX512BF16) {
                        floatRow.resize(m);
                        BFloat16ToFloat32(bf16InputData, floatRow.data(), m);
                        inputData = floatRow.data();
                    }
                } else {
                    bf16Input.resize(m);
                    Float32ToBFloat16(inputData, bf16Input.data(), m);
                    bf16InputData = bf16Input.data();
                }
                
                auto &middles = moeFloatSingleVarManager.middles;
                auto &swigluResults = moeFloatSingleVarManager.swigluResults;
                auto &results = moeFloatSingleVarManager.results;
                middles.resize(v.size());
                swigluResults.resize(v.size());
                results.resize(v.size());
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    middles[j].resize(weights[idx * 2]->dims[0]);
                    swigluResults[j].resize(weights[idx * 2]->dims[0]);
                    results[j].resize(weights[idx * 2 + 1]->dims[0]);
                }
                profileLap(profilePrepareMs);
                std::vector<fastllm::MultiThreadBaseOp*> ops;
                auto *pool = GetAlivePool();
                int threads = pool->curActivateThreadInterval.second -
                              pool->curActivateThreadInterval.first;
                ops.resize(threads);

                bool useDeepSeekV4MoeFast =
                    deepSeekV4Mode && !useGeglu && cpuInstructInfo.hasAVX512BF16 &&
                    std::getenv("FASTLLM_DSV4_DISABLE_CPU_MOE_FAST") == nullptr;
                if (useDeepSeekV4MoeFast) {
                    constexpr int gateRowsPerTask = 128;
                    constexpr int downRowsPerTask = 256;

                    DeepSeekV4MoeLinearTaskStorage gateTaskStorage;
                    ReserveDeepSeekV4MoeLinearTasks(
                        gateTaskStorage, v, weights, 0,
                        gateRowsPerTask);
                    for (int l = 0; l < v.size(); l++) {
                        int idx = v[l].first;
                        Data &weight = *weights[idx * 2];
                        AppendDeepSeekV4MoeLinearTasks(
                            gateTaskStorage, bf16InputData, weight,
                            middles[l].data(),
                            m, weight.dims[0], gateRowsPerTask);
                    }
                    std::vector<MultiThreadBaseOp*> gateTasks;
                    gateTaskStorage.BuildPointers(gateTasks);
                    ScheduleDeepSeekV4MoeTasks(gateTasks, false);
                    profileExpertCalls += v.size();
                    profileLap(profileGateMs);

                    // Each task owns one official FP8 activation block. This
                    // parallelizes SwiGLU, routing, quantization and BF16
                    // conversion without changing any dtype boundary.
                    std::vector<MultiThreadDeepSeekV4MoeDownPrepareOp>
                        downPrepareTaskStorage;
                    size_t downPrepareTaskCount = 0;
                    for (int l = 0; l < v.size(); l++) {
                        int idx = v[l].first;
                        int mid = weights[idx * 2]->dims[0] / 2;
                        downPrepareTaskCount += (mid + 127) / 128;
                    }
                    downPrepareTaskStorage.reserve(
                        downPrepareTaskCount);
                    for (int l = 0; l < v.size(); l++) {
                        int idx = v[l].first;
                        int spatial = weights[idx * 2]->dims[0];
                        int mid = spatial / 2;
                        Data &weightDown = *weights[idx * 2 + 1];
                        bool quantize = weightDown.dataType == DataType::FP8_E4M3 ||
                                        weightDown.dataType == DataType::NVFP4;
                        bool routed = idx != 0;
                        float routeWeight = routed ? v[l].second : 1.0f;
                        uint16_t *downInputData =
                            (uint16_t*)(swigluResults[l].data() + mid);
                        for (int st = 0; st < mid; st += 128) {
                            int end = std::min(st + 128, mid);
                            downPrepareTaskStorage.emplace_back(
                                middles[l].data(), swigluResults[l].data(),
                                downInputData, mid, st, end,
                                routed, routeWeight, swigluLimit,
                                true, quantize, true);
                        }
                    }
                    std::vector<MultiThreadBaseOp*> downPrepareTasks;
                    downPrepareTasks.reserve(
                        downPrepareTaskStorage.size());
                    for (auto &task : downPrepareTaskStorage) {
                        downPrepareTasks.push_back(&task);
                    }
                    ScheduleDeepSeekV4MoeTasks(
                        downPrepareTasks, false);
                    profileLap(profileSwigluQuantMs);

                    DeepSeekV4MoeLinearTaskStorage downTaskStorage;
                    ReserveDeepSeekV4MoeLinearTasks(
                        downTaskStorage, v, weights, 1,
                        downRowsPerTask);
                    for (int l = 0; l < v.size(); l++) {
                        int idx = v[l].first;
                        int mid = weights[idx * 2]->dims[0] / 2;
                        Data &weightDown = *weights[idx * 2 + 1];
                        uint16_t *downInputData =
                            (uint16_t*)(swigluResults[l].data() + mid);
                        AppendDeepSeekV4MoeLinearTasks(
                            downTaskStorage, downInputData, weightDown,
                            results[l].data(), mid, m, downRowsPerTask);
                    }
                    std::vector<MultiThreadBaseOp*> downTasks;
                    downTaskStorage.BuildPointers(downTasks);
                    ScheduleDeepSeekV4MoeTasks(downTasks, false);
                    profileLap(profileDownMs);
                } else for (int st = 0; st < v.size(); st++) {
                    int k = weights[v[st].first * 2]->dims[0];
                    int end = st, selSum = 1; // 一共处理selSum * k个输出

                    int curSum = 1;
                    for (int l = st + 1; l < v.size(); l++) {
                        int curK = weights[v[l].first * 2]->dims[0];
                        if (curK % k != 0) {
                            break;
                        }
                        curSum += (curK / k);
                        if (threads % curSum == 0) {
                            end = l;
                            selSum = curSum;
                        }
                    }
                    int base = threads / selSum;
                    int threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        Data *weight = weights[idx * 2];
                        float *outputData = middles[l].data();
                        float *biasData = nullptr;
                        int curK = weight->dims[0];
                        int curThread = (curK / k) * base;
                        if (weight->dataType == DataType::FP8_E4M3) {
                            LaunchLinearBFloat16FP8E4M3(bf16InputData, *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
                        } else if (weight->dataType == DataType::NVFP4) {
                            if (cpuInstructInfo.hasAVX512BF16) {
                                LaunchLinearBFloat16NVFP4(bf16InputData, *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
                            } else {
                                LaunchLinearFloat32NVFP4(inputData, *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
                            }
                        } else if (weight->dataType == DataType::BFLOAT16) {
                            LaunchLinearBFloat16BFloat16(bf16InputData, *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
                        } else {
                            // TODO: other
                        }
                        threadSt += curThread;
                    }
                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                        ops[j] = nullptr;
                    }
                    profileExpertCalls += end - st + 1;
                    profileLap(profileGateMs);

                    // The official DeepSeek-V4 expert has observable dtype and
                    // operation-order boundaries: w1/w3 return BF16, routed
                    // activations are clamped, the route weight is applied
                    // before w2, and quantized w2 receives FP8 activations.
                    bool runActivationInline =
                        !deepSeekV4Mode &&
                        weights[v[st].first * 2]->dataType ==
                            DataType::BFLOAT16;
                    threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int spatial = weights[idx * 2]->dims[0], mid = spatial / 2;
                        Data *weightDown = weights[idx * 2 + 1];
                        float *outputData = middles[l].data();
                        float *swigluData = swigluResults[l].data();
                        int curK = weights[idx * 2]->dims[0];
                        if (deepSeekV4Mode && !useGeglu) {
                            const bool routed = idx != 0;
                            const float routeWeight = routed ? v[l].second : 1.0f;
                            for (int i = 0; i < mid; i++) {
                                float gate = RoundFloat32ToBFloat16RNE(outputData[i]);
                                float up = RoundFloat32ToBFloat16RNE(outputData[mid + i]);
                                if (routed && swigluLimit > 0.0f) {
                                    gate = std::min(gate, swigluLimit);
                                    up = std::max(-swigluLimit, std::min(up, swigluLimit));
                                }
                                float h = (gate / (1.0f + std::exp(-gate))) * up;
                                swigluData[i] = RoundFloat32ToBFloat16RNE(routeWeight * h);
                            }
                            if (weightDown->dataType == DataType::FP8_E4M3 ||
                                weightDown->dataType == DataType::NVFP4) {
                                QuantizeDequantizeFP8E4M3Block128(swigluData, mid);
                            }
                            if (weightDown->dataType == DataType::FP8_E4M3 ||
                                (weightDown->dataType == DataType::NVFP4 && cpuInstructInfo.hasAVX512BF16) ||
                                weightDown->dataType == DataType::BFLOAT16) {
                                Float32ToBFloat16(swigluData, (uint16_t*)middles[l].data(), mid);
                            }
                            continue;
                        }

                        if (runActivationInline) {
                            if (useGeglu) {
                                MultiThreadGegluOp(
                                    outputData, mid, mid, swigluData,
                                    1, spatial, spatial).Run();
                            } else {
                                MultiThreadSwigluOp(
                                    outputData, mid, mid, swigluData,
                                    1, spatial, spatial).Run();
                            }
                            Float32ToBFloat16(
                                swigluData,
                                (uint16_t*)middles[l].data(), mid);
                            continue;
                        }

                        ops[l - st] = new fastllm::MultiThreadMultiOps();
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(
                            useGeglu ? (fastllm::MultiThreadBaseOp*)new fastllm::MultiThreadGegluOp(outputData, mid, mid, swigluData, 1, spatial, spatial)
                                     : (fastllm::MultiThreadBaseOp*)new fastllm::MultiThreadSwigluOp(outputData, mid, mid, swigluData, 1, spatial, spatial));
                        if (weightDown->dataType == DataType::FP8_E4M3 ||
                            (weightDown->dataType == DataType::NVFP4 && cpuInstructInfo.hasAVX512BF16) ||
                            weightDown->dataType == DataType::BFLOAT16) {
                            ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadFloat32ToBFloat16Op(swigluData, (uint16_t*)middles[l].data(), mid));
                        }
                        pool->PushOp(l - st, ops[l - st]);
                    }
                    for (int l = st; l <= end; l++) {
                        if ((deepSeekV4Mode && !useGeglu) ||
                            runActivationInline) {
                            continue;
                        }
                        pool->Wait(l - st);
                        delete ops[l - st];
                        ops[l - st] = nullptr;
                    }
                    profileLap(profileSwigluQuantMs);
                    threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int mid = weights[idx * 2]->dims[0] / 2;
                        int curK = weights[idx * 2]->dims[0];
                        Data *weightDown = weights[idx * 2 + 1];
                        int curThread = (curK / k) * base;
                        if (weightDown->dataType == DataType::FP8_E4M3) {
                            LaunchLinearBFloat16FP8E4M3((uint16_t*)middles[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
                        } else if (weightDown->dataType == DataType::NVFP4) {
                            if (cpuInstructInfo.hasAVX512BF16) {
                                LaunchLinearBFloat16NVFP4((uint16_t*)middles[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
                            } else {
                                LaunchLinearFloat32NVFP4(swigluResults[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
                            }
                        } else if (weightDown->dataType == DataType::BFLOAT16) {
                            LaunchLinearBFloat16BFloat16((uint16_t*)middles[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
                        } else {
                            // TODO: other
                        }
                        threadSt += curThread;               
                    }

                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                        ops[j] = nullptr;
                    }
                    profileLap(profileDownMs);
                    st = end;
                }
                float *fLastOutput = ((float*)output.cpuData) + o * m;
                std::vector <float> tempOutput;
                if (output.dataType == DataType::FLOAT16 || output.dataType == DataType::BFLOAT16) {
                    tempOutput.resize(m, 0);
                    fLastOutput = tempOutput.data();
                }
                if (useDeepSeekV4MoeFast) {
                    int reduceThreads = std::min(
                        {(int)pool->threads.size(),
                         16, m});
                    std::vector<MultiThreadDeepSeekV4MoeReduceOp> reduceOps;
                    reduceOps.reserve(reduceThreads);
                    int per = m / reduceThreads;
                    int cur = 0;
                    for (int i = 0; i < reduceThreads; i++) {
                        int end = cur + per +
                            (cur + per * (reduceThreads - i) < m);
                        if (i == reduceThreads - 1) {
                            end = m;
                        }
                        reduceOps.emplace_back(
                            &results, fLastOutput, (int)v.size(),
                            cur, end);
                        cur = end;
                    }
                    for (int i = 0; i < reduceThreads; i++) {
                        pool->PushOp(i, &reduceOps[i]);
                    }
                    for (int i = 0; i < reduceThreads; i++) {
                        pool->Wait(i);
                    }
                } else for (int j = 0; j < v.size(); j++) {
                    float value = deepSeekV4Mode ? 1.0f : v[j].second;
                    float *curOutput = (float*)results[j].data();
                    if (deepSeekV4Mode) {
                        for (int d = 0; d < m; d++) {
                            curOutput[d] = RoundFloat32ToBFloat16RNE(curOutput[d]);
                        }
                    }
                    int i = 0;
#ifdef __AVX2__
                    __m256 value_vec = _mm256_set1_ps(value);

                    // 每次处理 8 个浮点数（AVX2 寄存器可以容纳 8 个 float）
                    for (; i <= m - 8; i += 8) {
                        // 加载 curOutput 的 8 个浮点数
                        __m256 curOutput_vec = _mm256_loadu_ps(&curOutput[i]);

                        // 加载 fLastOutput 的 8 个浮点数
                        __m256 fLastOutput_vec = _mm256_loadu_ps(&fLastOutput[i]);

                        // 计算 curOutput * value
                        __m256 result_vec = _mm256_mul_ps(curOutput_vec, value_vec);

                        // 累加到 fLastOutput
                        fLastOutput_vec = _mm256_add_ps(fLastOutput_vec, result_vec);

                        // 将结果存回 fLastOutput
                        _mm256_storeu_ps(&fLastOutput[i], fLastOutput_vec);
                    }
#endif
                    // 处理剩余的不足 8 个的元素
                    for (; i < m; i++) {
                        fLastOutput[i] += curOutput[i] * value;
                    }
                }
                profileLap(profileReduceMs);
                if (output.dataType == DataType::FLOAT16) {
                    Float32ToFloat16(tempOutput.data(), ((uint16_t*)output.cpuData) + o * m, m);
                } else if (output.dataType == DataType::BFLOAT16) {
                    uint16_t *dst = ((uint16_t*)output.cpuData) + o * m;
                    if (deepSeekV4Mode) {
                        for (int d = 0; d < m; d++) {
                            dst[d] = Float32ToBFloat16RNEBits(tempOutput[d]);
                        }
                    } else {
                        Float32ToBFloat16(tempOutput.data(), dst, m);
                    }
                }
                profileLap(profileOutputMs);
            }
            if (profileDetail) {
                double total = profilePrepareMs + profileGateMs + profileSwigluQuantMs +
                               profileDownMs + profileReduceMs + profileOutputMs;
                printf("[fastllm-profile-cpu-moe] quant_small outer=%d topk=%d experts=%d prepare=%.3f gate=%.3f swiglu_quant=%.3f down=%.3f reduce=%.3f output=%.3f total=%.3f\n",
                       outer, topk, profileExpertCalls, profilePrepareMs, profileGateMs,
                       profileSwigluQuantMs, profileDownMs, profileReduceMs,
                       profileOutputMs, total);
                fflush(stdout);
            }
        } else if ((input.dataType == DataType::FLOAT32 ||
                    input.dataType == DataType::BFLOAT16) &&
                   (output.dataType == DataType::FLOAT32 ||
                    output.dataType == DataType::BFLOAT16) &&
                   weights[2]->dataType == DataType::BFLOAT16) {
 auto st = std::chrono::system_clock::now();
            Data gate, attenPart, moePart;
            int bs = input.dims[0];
            int m = weightsBatch / 2 - 1; // num experts

            {
                auto *pool = GetAlivePool();

                int dim = output.dims[1];
                int inputDim = input.dims[1];
                int interDim = weights[2]->dims[0] / 2;
                int outputDim = output.dims[1];

                std::vector <std::vector <std::pair <int, float> > > expertTasks; // expertTasks[i]代表专家i的task, expertTasks[i][j] = (第j个任务对应的行数， 权重)
                expertTasks.resize(m + 1);
                for (int b = 0; b < bs; b++) {
                    expertTasks[0].push_back(std::make_pair(b, sharedScale));
                    for (int j = 0; j < topk; j++) {
                        int expertIdx = normalizeExpertIdx(indexData[b * topk + j]);
                        float value = scoreData[b * topk + j];
                        expertTasks[expertIdx + 1].push_back(std::make_pair(b, value));
                    }
                }

                int totalLines = 0;
                for (int e = 0; e < expertTasks.size(); e++) {
                    if (weights[e * 2] != nullptr) {
                        totalLines += expertTasks[e].size();
                    }
                }
//printf("prepare spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                DataType startDataType = DataType::BFLOAT16;
                DataType downInputDataType = DataType::BFLOAT16;

                // 从 fastllmMoeDataManager 获取缓存的 vector，并根据需要调整大小
                auto& realInput = fastllmMoeDataManager.realInput;
                auto& expandInput = fastllmMoeDataManager.expandInput;
                auto& gateUpOutput = fastllmMoeDataManager.gateUpOutput;
                auto& swigluOutput = fastllmMoeDataManager.swigluOutput;
                auto& downInput = fastllmMoeDataManager.downInput;
                auto& downOutput = fastllmMoeDataManager.downOutput;
                auto& reduceOutput = fastllmMoeDataManager.reduceOutput;

                // 计算所需大小
                size_t realInputSize = GetDataBytes(startDataType, bs, inputDim);
                size_t expandInputSize = GetDataBytes(startDataType, totalLines, inputDim);
                size_t gateUpOutputSize = totalLines * interDim * 2;
                size_t swigluOutputSize = totalLines * interDim;
                size_t downInputSize = GetDataBytes(downInputDataType, totalLines, outputDim);
                size_t downOutputSize = totalLines * outputDim;
                size_t reduceOutputSize = bs * outputDim;

                // 只在当前容量不足时才进行 resize
                if (realInput.size() < realInputSize) {
                    realInput.resize(realInputSize);
                }
                if (expandInput.size() < expandInputSize) {
                    expandInput.resize(expandInputSize);
                }
                if (gateUpOutput.size() < gateUpOutputSize) {
                    gateUpOutput.resize(gateUpOutputSize);
                }
                if (swigluOutput.size() < swigluOutputSize) {
                    swigluOutput.resize(swigluOutputSize);
                }
                if (downInput.size() < downInputSize) {
                    downInput.resize(downInputSize);
                }
                if (downOutput.size() < downOutputSize) {
                    downOutput.resize(downOutputSize);
                }
                if (reduceOutput.size() < reduceOutputSize) {
                    reduceOutput.resize(reduceOutputSize);
                }

//printf("malloc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 0. input -> realInput
                if (input.dataType == DataType::FLOAT32) {
                    RunMultiThreadConvertFromFloat32(realInput.data(), DataType::BFLOAT16,
                                                     (float*)input.cpuData, bs, inputDim,
                                                     GetAlivePool());
                } else {
                    memcpy(realInput.data(), input.cpuData, realInputSize);
                }
//printf("Float32ToBFloat16 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                // 1. realInput -> expandInput
                std::vector <MultiThreadMemcpyMultiLinesTask> memcpyTasks;
                memcpyTasks.resize(totalLines);
                {
                    int offset = 0;
                    uint8_t* realInputPtr = realInput.data();
                    uint8_t* expandInputPtr = expandInput.data();
                    int bytesPerLine = GetDataBytes(startDataType, 1, inputDim);
                    
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr) {
                            for (auto& task : expertTasks[e]) {
                                int rowIdx = task.first;

                                memcpyTasks[offset] = MultiThreadMemcpyMultiLinesTask(
                                    expandInputPtr + offset * bytesPerLine, 
                                    realInputPtr + rowIdx * bytesPerLine, 
                                    bytesPerLine
                                );
                                offset++;
                            }
                        }
                    }
                }
                RunMultiThreadMemcpyMultiLines(memcpyTasks, GetAlivePool());
//printf("expand spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 2. gateUp
                {
long long ops = 0;
                    int offset = 0;
                    int stride = 64;
                    std::vector<MultiThreadBaseOp*> gemmOps;
                    int transposedCount = 0;
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr && expertTasks[e].size() > 0) {
                            int lines = expertTasks[e].size();

                            // Prepare input pointer for this expert's batch
                            uint16_t* expertInputPtr = (uint16_t*)(expandInput.data() + offset * GetDataBytes(startDataType, 1, inputDim));
                            
                            // Prepare output pointer for this expert's batch
                            float* expertGateUpOutputPtr = gateUpOutput.data() + offset * interDim * 2;
                            
                            // Get weight data (assuming weights are stored as BFloat16)
                            uint16_t* weightPtr = (uint16_t*)(weights[e * 2]->cpuData);

                            TransposedWeightView tw;
                            if (lines == 1 && weights[e * 2]->dataType == DataType::BFLOAT16) {
                                tw = GetTransposedBF16Weight(weightPtr, interDim * 2, inputDim);
                            }
                            if (tw.Valid()) {
                                int numCb = interDim * 2 / 64;
                                for (int cb = 0; cb < numCb; cb++) {
                                    gemmOps.push_back(new MultiThreadGemvBF16Block64Op(
                                        expertInputPtr, tw.ShardBase(cb), expertGateUpOutputPtr,
                                        inputDim, tw.LocalCb(cb), cb, tw.NodeOf(cb)
                                    ));
                                    transposedCount++;
                                }
                            } else {
                                for (int st = 0; st < interDim * 2; st += stride) {
                                    int end = std::min(st + stride, interDim * 2);
                                    gemmOps.push_back(new MultiThreadGemmOp(
                                        (uint8_t*)expertInputPtr, DataType::BFLOAT16,
                                        (uint8_t*)weightPtr, DataType::BFLOAT16,
                                        (uint8_t*)expertGateUpOutputPtr, DataType::FLOAT32,
                                        lines, inputDim, interDim * 2, st, end
                                    ));
                                }
                            }
ops += (long long)lines * inputDim * interDim * 2;
                            offset += lines;
                        }
                    }
                    ScheduleCpuMoEGemm(gemmOps, transposedCount);
//printf("ops = %f g\n", (float)ops / 1e9);
                }
//printf("gateup spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                // 3. swiglu
                if (useGeglu) {
                    GegluMultiThread((float *) gateUpOutput.data(), interDim, interDim, ((float *) swigluOutput.data()),
                                     totalLines, interDim * 2, interDim, GetAlivePool());
                } else {
                    SwigluMultiThread((float *) gateUpOutput.data(), interDim, interDim, ((float *) swigluOutput.data()),
                                      totalLines, interDim * 2, interDim, GetAlivePool());
                }
//printf("swiglu spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                // 4. swigluOutput -> downInput
                RunMultiThreadConvertFromFloat32(downInput.data(), DataType::BFLOAT16, (float*)swigluOutput.data(), totalLines, interDim, GetAlivePool());

//printf("Float32ToBFloat16 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 5. down
                {
                    int offset = 0;
                    int stride = 64;
                    std::vector <MultiThreadBaseOp*> gemmOps;
                    int transposedCount = 0;
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2 + 1] != nullptr && expertTasks[e].size() > 0) {
                            int lines = expertTasks[e].size();
                            
                            // Prepare input pointer for this expert's batch
                            uint16_t* expertDownInputPtr = (uint16_t*)(downInput.data() + offset * GetDataBytes(downInputDataType, 1, interDim));
                            
                            // Prepare output pointer for this expert's batch
                            float* expertDownOutputPtr = downOutput.data() + offset * dim;
                            
                            // Get weight data (assuming weights are stored as BFloat16)
                            uint16_t* weightPtr = (uint16_t*)(weights[e * 2 + 1]->cpuData);

                            TransposedWeightView tw;
                            if (lines == 1 && weights[e * 2 + 1]->dataType == DataType::BFLOAT16) {
                                tw = GetTransposedBF16Weight(weightPtr, dim, interDim);
                            }
                            if (tw.Valid()) {
                                int numCb = dim / 64;
                                for (int cb = 0; cb < numCb; cb++) {
                                    gemmOps.push_back(new MultiThreadGemvBF16Block64Op(
                                        (const uint16_t*)expertDownInputPtr, tw.ShardBase(cb),
                                        (float*)expertDownOutputPtr, interDim,
                                        tw.LocalCb(cb), cb, tw.NodeOf(cb)
                                    ));
                                    transposedCount++;
                                }
                            } else {
                                for (int st = 0; st < dim; st += stride) {
                                    int end = std::min(st + stride, dim);
                                    gemmOps.push_back(new MultiThreadGemmOp (
                                        (uint8_t*)expertDownInputPtr, DataType::BFLOAT16, 
                                        (uint8_t*)weightPtr, DataType::BFLOAT16, 
                                        (uint8_t*)expertDownOutputPtr, DataType::FLOAT32, 
                                        lines, interDim, dim, st, end
                                    ));
                                }
                            }
                            offset += lines;
                        }
                    }
                    ScheduleCpuMoEGemm(gemmOps, transposedCount);
                }

//printf("down spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 6. reduce
                {
                    // 准备数据结构
                    int total_tasks = 0;
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr) {
                            total_tasks += expertTasks[e].size();
                        }
                    }
                    // 假设每个样本最多选择k个专家
                    int k = 0; // 需要确定每个样本选择的专家数量
                    std::vector<int> samples_expert_count(bs, 0);
                    // 第一遍：统计每个样本的专家数量
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr) {
                            for (auto& task : expertTasks[e]) {
                                int rowIdx = task.first;
                                samples_expert_count[rowIdx]++;
                                k = std::max(k, samples_expert_count[rowIdx]);
                            }
                        }
                    }
                    // 分配内存
                    std::vector<int> pos(bs * k, -1);  // 初始化为-1表示无效位置
                    std::vector<float> task_weights(total_tasks, 0.0f);
                    std::vector<int> sample_expert_idx(bs, 0);  // 记录每个样本当前填充到第几个专家
                    // 第二遍：填充pos和weights数组
                    int offset = 0;
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr) {
                            for (auto& task : expertTasks[e]) {
                                int rowIdx = task.first;
                                float weight = task.second;
                                
                                // 在pos数组中记录这个任务的位置
                                int expert_idx = sample_expert_idx[rowIdx]++;
                                pos[rowIdx * k + expert_idx] = offset;
                                task_weights[offset] = weight;
                                
                                offset++;
                            }
                        }
                    }

                    // 调用多线程函数
                    MultiThreadReduceBatch(
                        (uint8_t*)downOutput.data(),  // downOutData
                        DataType::FLOAT32,             // downOutDataType (假设是float32)
                        task_weights.data(),           // weights
                        output.dataType == DataType::FLOAT32 ? (float*)output.cpuData : reduceOutput.data(),           // lastOutput
                        pos.data(),                    // pos
                        bs,                           // bsz
                        k,                            // k (每个样本的专家数)
                        dim                           // hidden_size
                    );
                    // 注意：如果某些样本的专家数少于k，需要特殊处理
                    // 可以在MultiThreadReduceBatchOp::Run()中添加检查：
                    // if (curPos == -1) continue; // 跳过无效位置
                }
//printf("reduce spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 7. reduceOutput -> last Output
                if (output.dataType != DataType::FLOAT32) {
                    if (output.dataType == DataType::FLOAT16) {
                        Float32ToFloat16(reduceOutput.data(), (uint16_t*)output.cpuData, output.Count(0));
                    } else if (output.dataType == DataType::BFLOAT16) {
                        Float32ToBFloat16(reduceOutput.data(), (uint16_t*)output.cpuData, output.Count(0));
                    }
                }
//printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            }
        } else {
  // auto st = std::chrono::system_clock::now();
  // auto veryst = std::chrono::system_clock::now();
  // std::map <std::string, float> cnt;
            // normal
            Data gate, attenPart, moePart;
            int m = weightsBatch / 2 - 1; // num experts

            if (input.dims[0] == 1) {
                output.Allocate(0.0f);
                for (int j = 0; j < topk; j++) {
                    int expertIdx = normalizeExpertIdx(indexData[j]);
                    float value = scoreData[j];

                    Linear(input, *weights[(expertIdx + 1) * 2], Data(), w3);
                    if (useGeglu) {
                        Geglu(w3, w1);
                    } else {
                        Swiglu(w3, w1);
                    }
                    Linear(w1, *weights[(expertIdx + 1) * 2 + 1], Data(), w2);
                    AddTo(output, w2, value);
                }

                if (weights[0] != nullptr) {
                    Linear(input, *weights[0], Data(), w3);
                    if (useGeglu) {
                        Geglu(w3, w1);
                    } else {
                        Swiglu(w3, w1);
                    }
                    Linear(w1, *weights[1], Data(), w2);
                    AddTo(output, w2, sharedScale);
                }
            } else {
                int bs = input.dims[0], dim = output.dims[1];
                int inputDim = input.dims[1];
                std::vector <float> tempResult, middleResult;
                tempResult.resize(bs * dim, 0.0f);
                middleResult.resize(bs * dim, 0.0f);
                std::vector <std::vector <std::pair <int, float> > > expertTasks; // expertTasks[i]代表专家i的task, expertTasks[i][j] = (第j个任务对应的行数， 权重)
                expertTasks.resize(m + 1);
                Data &tempInput = w2;
                tempInput.ToDevice(input.dataDevice);
                // w2 is a reusable scratch tensor and may still own storage
                // allocated for a different dtype. expansionSize is measured
                // in elements, so merely changing dataType can make Allocate
                // incorrectly reuse a buffer with too few bytes.
                if (tempInput.dataType != input.dataType) {
                    tempInput.FreeSpace();
                    tempInput.dataType = input.dataType;
                }
                tempInput.Resize(input.dims);
  // cnt["prepare 0"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                tempInput.Allocate();
  // cnt["allocate"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                for (int b = 0; b < bs; b++) {
                    expertTasks[0].push_back(std::make_pair(b, sharedScale));
                    for (int j = 0; j < topk; j++) {
                        int expertIdx = normalizeExpertIdx(indexData[b * topk + j]);
                        float value = scoreData[b * topk + j];
                        expertTasks[expertIdx + 1].push_back(std::make_pair(b, value));
                    }
                }
  // cnt["prepare"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                for (int e = 0; e < expertTasks.size(); e++) {
                    auto &task = expertTasks[e];
                    if (task.size() == 0) {
                        continue;
                    }
                    if (weights[e * 2] == nullptr) {
                        continue;
                    }

                    auto canForceFloatOutput = [](const Data &curInput, const Data &curWeight) {
                        if (curInput.dataType == DataType::FLOAT32) {
                            return true;
                        }
                        if (curInput.dataType != DataType::BFLOAT16) {
                            return false;
                        }
                        return CanRunBFloat16NativeLinearWeight(curWeight) ||
                               curWeight.dataType == DataType::BFLOAT16 ||
                               curWeight.dataType == DataType::FLOAT16 ||
                               curWeight.dataType == DataType::FP8_E4M3 ||
                               curWeight.dataType == DataType::NVFP4;
                    };
                    auto reshapeMoeLinear = [&](Data &curInput, Data &curWeight, Data &curOutput) {
                        curWeight.weightType = WeightType::LINEAR;
                        DataType outputType = canForceFloatOutput(curInput, curWeight) ?
                                              DataType::FLOAT32 : curInput.dataType;
                        if (curOutput.dataType != outputType) {
                            curOutput.FreeSpace();
                            curOutput.dataType = outputType;
                        }
                        std::vector<int> dims = curInput.dims;
                        dims.back() = curWeight.dims[0];
                        curOutput.Resize(dims);
                    };

                    tempInput.Resize({(int)task.size(), inputDim});
                    tempInput.Allocate();

                    std::vector <MultiThreadMemcpyMultiLinesTask> memcpyTasks;
                    for (int i = 0; i < (int)task.size(); i++) {
                        memcpyTasks.push_back(MultiThreadMemcpyMultiLinesTask(tempInput.cpuData + i * inputDim * input.unitSize, input.cpuData + task[i].first * inputDim * input.unitSize, inputDim * input.unitSize));
                    }
                    RunMultiThreadMemcpyMultiLines(memcpyTasks, GetAlivePool());
                    reshapeMoeLinear(tempInput, *weights[e * 2], w3);
 // cnt["linear 0 prepare"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                    DoCpuLinear(tempInput, *weights[e * 2], Data(), w3);
 // cnt["linear 0"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();                    
                    int mid = w3.dims[1] / 2;
                    if (w1.dataType != w3.dataType) {
                        w1.FreeSpace();
                        w1.dataType = w3.dataType;
                    }
                    w1.Resize({w3.dims[0], mid});
                    w1.Allocate();

                    if (w3.dataType == DataType::FLOAT32) {
                        if (useGeglu) {
                            GegluMultiThread((float *) w3.cpuData, mid, mid, ((float *) w1.cpuData),
                                             w3.dims[0], w3.dims[1], mid, GetAlivePool());
                        } else {
                            SwigluMultiThread((float *) w3.cpuData, mid, mid, ((float *) w1.cpuData),
                                              w3.dims[0], w3.dims[1], mid, GetAlivePool());
                        }
                    } else if (w3.dataType == DataType::FLOAT16) {
                        if (useGeglu) {
                            GegluMultiThreadFloat16((uint16_t *) w3.cpuData, mid, mid, ((uint16_t *) w1.cpuData),
                                                    w3.dims[0], w3.dims[1], mid, GetAlivePool());
                        } else {
                            SwigluMultiThreadFloat16((uint16_t *) w3.cpuData, mid, mid, ((uint16_t *) w1.cpuData),
                                                     w3.dims[0], w3.dims[1], mid, GetAlivePool());
                        }
                    } else {
                        if (useGeglu) {
                            GegluMultiThreadBFloat16((uint16_t *) w3.cpuData, mid, mid, ((uint16_t *) w1.cpuData),
                                                     w3.dims[0], w3.dims[1], mid, GetAlivePool());
                        } else {
                            SwigluMultiThreadBFloat16((uint16_t *) w3.cpuData, mid, mid, ((uint16_t *) w1.cpuData),
                                                      w3.dims[0], w3.dims[1], mid, GetAlivePool());
                        }
                    }
  // cnt["swiglu"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                    reshapeMoeLinear(w1, *weights[e * 2 + 1], w3);
  // cnt["linear 1 prepare"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                    DoCpuLinear(w1, *weights[e * 2 + 1], Data(), w3);
  // cnt["linear 1"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();                    
                    float *curOutput;
                    if (w3.dataType == DataType::FLOAT32) {
                        curOutput = (float*)w3.cpuData;
                    } else if (w3.dataType == DataType::FLOAT16) {
                        Float16ToFloat32((uint16_t*)w3.cpuData, middleResult.data(), w3.Count(0));
                        curOutput = middleResult.data();
                    } else if (w3.dataType == DataType::BFLOAT16) {
                        uint16_t *src = (uint16_t*)w3.cpuData;
                        int len = w3.Count(0);
                        for (int idx = 0; idx < len; idx++) {
                            uint32_t x = (uint32_t)src[idx] << 16;
                            middleResult[idx] = *(float*)&x;
                        }
                        curOutput = middleResult.data();
                    }

                    RunMultiThreadMoeReduce(&task, &tempResult, curOutput, dim, GetAlivePool());
  // cnt["reduce"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                }
                if (output.dataType == DataType::FLOAT32) {
                    memcpy(output.cpuData, tempResult.data(), output.GetBytes());
                } else if (output.dataType == DataType::FLOAT16) {
                    Float32ToFloat16(tempResult.data(), (uint16_t*)output.cpuData, output.Count(0));
                } else if (output.dataType == DataType::BFLOAT16) {
                    Float32ToBFloat16(tempResult.data(), (uint16_t*)output.cpuData, output.Count(0));
                }
  // cnt["output memcpy"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
            }
  //printf("moe spend %f s.\n", GetSpan(veryst, std::chrono::system_clock::now()));
  // for (auto &it : cnt) {
     //printf("%s spend %f s.\n", it.first.c_str(), it.second);
  // }
        }
    }

    void CpuMergeMLA::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qNope = *(datas.find("qNope")->second);
        Data &output = *(datas.find("output")->second);
        // int b = qNope.dims[0], s = q_nope.dims[1], h = q_nope.dims[2], d = q_nope.dims[3], c = qNope.dims.back();
        output.dataType = qNope.dataType;
        output.Resize(qNope.dims);
    }

    void CpuMergeMLA::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qNope = *(datas.find("qNope")->second);
        Data &qPe = *(datas.find("qPe")->second);
        Data &kvCache = *(datas.find("kvCache")->second);
        Data &peCache = *(datas.find("peCache")->second);
        Data &mask = *(datas.find("mask")->second);
        Data &output = *(datas.find("output")->second);
        float softmaxScale = floatParams.find("softmaxScale") != floatParams.end() ? floatParams.find("softmaxScale")->second : 1.0f;        
        int b = qPe.dims[0], s = qPe.dims[1], h = qPe.dims[2], c = qNope.dims.back(), t = kvCache.dims[1], r = qPe.dims[3];
        output.Allocate();

        // qNope: {b * s, h, c}
        // qPe: {b, s, h, r}
        // kvCache : {1, t, r}
        // peCache : {1, t, c}
        // output : {b * s, h, c}

        Data score0, score1;
        if (b == 1 && s == 1 && false) {
            // FastllmCudaMLA(qNope, qPe, kvCache, peCache, score0, output, softmaxScale);
        } else {
            if ((double)b * s * h * t * 2 * 4 > 1e9) {
                int parth = 1;
                Data qNopePart, qPePart;
                std::vector <Data> outputParts;
                std::vector <Data*> outputPartPointers;
                outputParts.resize((h - 1) / parth + 1);
                for (int i = 0; i < outputParts.size(); i++) {
                    outputPartPointers.push_back(&outputParts[i]);
                }
                for (int sth = 0; sth < h; sth += parth) {
                    int idx = sth / parth;
                    int curh = std::min(parth, h - sth);
                    Split(qNope, 1, sth, sth + curh, qNopePart);
                    Split(qPe, 2, sth, sth + curh, qPePart);
                    qNopePart.Reshape({b, s * curh, c});
                    MatMulTransB(qNopePart, peCache, score0);
                    score0.Reshape({b, s, curh, t});
                    qPePart.Reshape({b, s * curh, r});
                    MatMulTransB(qPePart, kvCache, score1);
                    score1.Reshape({b, s, curh, t});
                    AddTo(score1, score0);
                    Mul(score1, softmaxScale, score0);
                    if (mask.dims.size() > 0) {
                        score0.Reshape({b * s, curh, t});
                        ToDataType(mask, qNope.dataType);
                        AttentionMask(score0, mask, -10000);
                    }

                    Softmax(score0, score0, -1);
                    score0.Reshape({b, s * curh, t});
                    MatMul(score0, peCache, outputParts[idx]);
                    outputParts[idx].Reshape({b, s, curh, c});
                }
                CatBatch(outputPartPointers, 2, output);
                output.Reshape({b * s, h, c});
            } else {
                qNope.Reshape({b, s * h, c});
                MatMulTransB(qNope, peCache, score0);
                score0.Reshape({b, s, h, t});

                qPe.Reshape({qPe.dims[0], -1, qPe.dims[3]});
                MatMulTransB(qPe, kvCache, score1);
                score1.Reshape({b, s, h, t});
                AddTo(score1, score0);
                Mul(score1, softmaxScale, score0);

                if (mask.dims.size() > 0) {
                    score0.Reshape({b * s, h, t});
                    ToDataType(mask, qNope.dataType);
                    AttentionMask(score0, mask, -10000);
                }

                Softmax(score0, score0, -1);
                score0.Reshape({b, s * h, t});
                MatMul(score0, peCache, output);
            }
        }
    }

    void CpuMergeMLAPaged::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qNope = *(datas.find("qNope")->second);
        Data &output = *(datas.find("output")->second);
        output.dataType = qNope.dataType;
        output.Resize(qNope.dims);
    }

    void CpuMergeMLAPaged::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qNope = *(datas.find("qNope")->second);
        Data &qPe = *(datas.find("qPe")->second);
        Data &kvCachePaged = *(datas.find("kvCachePaged")->second);
        Data &peCachePaged = *(datas.find("peCachePaged")->second);
        Data &output = *(datas.find("output")->second);
        float softmaxScale = floatParams.find("softmaxScale") != floatParams.end() ? floatParams.find("softmaxScale")->second : 1.0f;
        int requestedKvLen = intParams.find("kvLen") != intParams.end() ?
            intParams.find("kvLen")->second : -1;

        AssertInFastLLM(kvCachePaged.isPagedKVCache && peCachePaged.isPagedKVCache,
            "CpuMergeMLAPaged: kvCachePaged and peCachePaged must be paged KV cache (isPagedKVCache=true).\n");
        AssertInFastLLM(kvCachePaged.pageIndex.size() == peCachePaged.pageIndex.size() &&
            kvCachePaged.lastPageLen == peCachePaged.lastPageLen && kvCachePaged.pageLen == peCachePaged.pageLen,
            "CpuMergeMLAPaged: kvCachePaged and peCachePaged must share same page layout.\n");

        // qNope 布局固定为 [h, b*s, c]；qPe 可能为 [b, h, s, r] 或 [b, s, h, r]，按 qNope 推断并兼容两种
        int hNope = qNope.dims[0], bsNope = qNope.dims[1], c = qNope.dims[2];
        int b, s, h, r = qPe.dims[3];
        bool qPeLayoutBhs = (qPe.dims[1] == hNope && (int)qPe.dims[0] * (int)qPe.dims[2] == bsNope);
        bool qPeLayoutBsh = (qPe.dims[2] == hNope && (int)qPe.dims[0] * (int)qPe.dims[1] == bsNope);
        AssertInFastLLM(qPeLayoutBhs || qPeLayoutBsh, "CpuMergeMLAPaged: qNope shape must be [h, b*s, c] and qPe [b,h,s,r] or [b,s,h,r].\n");
        if (qPeLayoutBhs) {
            b = qPe.dims[0]; h = qPe.dims[1]; s = qPe.dims[2];
        } else {
            b = qPe.dims[0]; s = qPe.dims[1]; h = qPe.dims[2];
        }
        int numPages = (int)kvCachePaged.pageIndex.size();
        int pageLen = kvCachePaged.pageLen;
        int fullKvLen = (numPages > 0) ?
            (numPages - 1) * pageLen + kvCachePaged.lastPageLen : 0;
        int kvLen = requestedKvLen > 0 ? requestedKvLen : fullKvLen;
        AssertInFastLLM(
            kvLen > 0 && kvLen <= fullKvLen && kvLen >= s,
            "CpuMergeMLAPaged: requested KV length is invalid.\n");
        numPages = (kvLen + pageLen - 1) / pageLen;
        int lastPageLen = (kvLen - 1) % pageLen + 1;

        output.Allocate();

        // pagedKVCacheData 为整体数据，形状 [maxPages, pageLen, 1, r] 或 [maxPages, pageLen, 1, c]；真实 kv/pe 由 pageIndex 指向的页拼成
        AssertInFastLLM(peCachePaged.pagedKVCacheData->dims.size() >= 4 && kvCachePaged.pagedKVCacheData->dims.size() >= 4,
            "CpuMergeMLAPaged: pagedKVCacheData shape must be [maxPages, pageLen, 1, feat].\n");
        int ckvDim2 = peCachePaged.pagedKVCacheData->dims[2];
        int ckvDim3 = peCachePaged.pagedKVCacheData->dims[3];
        int kpeDim2 = kvCachePaged.pagedKVCacheData->dims[2];
        int kpeDim3 = kvCachePaged.pagedKVCacheData->dims[3];
        AssertInFastLLM(ckvDim3 == c && kpeDim3 == r, "CpuMergeMLAPaged: cache feature dim mismatch (pastValue last dim=c, pastKey last dim=r).\n");
        int unitSizeCkv = peCachePaged.unitSize;
        int unitSizeKpe = kvCachePaged.unitSize;
        uint8_t *ckvData = peCachePaged.pagedKVCacheData->cpuData;
        uint8_t *kpeData = kvCachePaged.pagedKVCacheData->cpuData;
        size_t ckvPosStride = (size_t)ckvDim2 * ckvDim3 * unitSizeCkv;
        size_t ckvPageStride = (size_t)pageLen * ckvPosStride;
        size_t kpePosStride = (size_t)kpeDim2 * kpeDim3 * unitSizeKpe;
        size_t kpePageStride = (size_t)pageLen * kpePosStride;

        auto getCkvAt = [&](int kvPos) -> const uint8_t* {
            if (kvPos >= kvLen) return nullptr;
            int pi = kvPos / pageLen;
            int posInPage = kvPos % pageLen;
            if (pi >= numPages) return nullptr;
            int actualPage = peCachePaged.pageIndex[pi];
            if (pi == numPages - 1 && posInPage >= lastPageLen) return nullptr;
            return ckvData + (size_t)actualPage * ckvPageStride + (size_t)posInPage * ckvPosStride;
        };
        auto getKpeAt = [&](int kvPos) -> const uint8_t* {
            if (kvPos >= kvLen) return nullptr;
            int pi = kvPos / pageLen;
            int posInPage = kvPos % pageLen;
            if (pi >= numPages) return nullptr;
            int actualPage = kvCachePaged.pageIndex[pi];
            if (pi == numPages - 1 && posInPage >= lastPageLen) return nullptr;
            return kpeData + (size_t)actualPage * kpePageStride + (size_t)posInPage * kpePosStride;
        };

        auto dotF32 = [](const float* a, const float* b, int n) {
            float sum = 0.f;
#ifdef __AVX__
            __m256 vsum = _mm256_setzero_ps();
            int i = 0;
            for (; i + 7 < n; i += 8) {
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
            }
            sum = Floatsum(vsum);
            for (; i < n; i++) sum += a[i] * b[i];
#else
            for (int i = 0; i < n; i++) sum += a[i] * b[i];
#endif
            return sum;
        };

        auto dotF16 = [](const uint16_t* a, const uint16_t* b, int n) {
            float sum = 0.f;
            for (int i = 0; i < n; i++) sum += half_to_float(a[i]) * half_to_float(b[i]);
            return sum;
        };

        const bool causal = true;
        int base = kvLen - s;

        if (qNope.dataType == DataType::FLOAT32) {
            float *qNopePtr = (float*)qNope.cpuData;
            float *qPePtr = (float*)qPe.cpuData;
            float *outPtr = (float*)output.cpuData;
            std::fill(outPtr, outPtr + output.Count(0), 0.f);

            for (int bi = 0; bi < b; bi++) {
                for (int ti = 0; ti < s; ti++) {
                    for (int hi = 0; hi < h; hi++) {
                        int ni = bi * s + ti;
                        int qFlat = hi * (b * s) + ni;
                        const float *qn = qNopePtr + qFlat * c;
                        int qPeOff = qPeLayoutBhs ? (bi * h * s + hi * s + ti) * r : (ni * h + hi) * r;
                        const float *qp = qPePtr + qPeOff;
                        float *oHead = outPtr + qFlat * c;

                        int sMin = 0, sMax = kvLen - 1;
                        if (causal && base >= 0) {
                            sMax = std::min(ti + base, kvLen - 1);
                        } else if (causal) {
                            sMax = std::min(ti, kvLen - 1);
                        }

                        std::vector<float> logits(kvLen, -1e9f);
                        float m = -1e9f;
                        for (int j = sMin; j <= sMax; j++) {
                            const uint8_t *ckvJ = getCkvAt(j);
                            const uint8_t *kpeJ = getKpeAt(j);
                            if (!ckvJ || !kpeJ) continue;
                            float acc = dotF32(qn, (const float*)ckvJ, c) + dotF32(qp, (const float*)kpeJ, r);
                            logits[j] = acc * softmaxScale;
                            m = std::max(m, logits[j]);
                        }

                        float denom = 0.f;
                        for (int j = sMin; j <= sMax; j++) {
                            logits[j] = expf(logits[j] - m);
                            denom += logits[j];
                        }
                        if (denom < 1e-9f) denom = 1e-9f;
                        for (int j = sMin; j <= sMax; j++) logits[j] /= denom;

                        for (int j = sMin; j <= sMax; j++) {
                            const uint8_t *ckvJ = getCkvAt(j);
                            if (!ckvJ) continue;
                            const float *vj = (const float*)ckvJ;
                            float p = logits[j];
                            for (int d = 0; d < c; d++) oHead[d] += p * vj[d];
                        }
                    }
                }
            }
        } else {
            uint16_t *qNopePtr = (uint16_t*)qNope.cpuData;
            uint16_t *qPePtr = (uint16_t*)qPe.cpuData;
            uint16_t *outPtr = (uint16_t*)output.cpuData;
            std::fill(outPtr, outPtr + output.Count(0), float_to_half(0.f));

            std::vector<float> fqn(c), fqp(r), fo(c);
            for (int bi = 0; bi < b; bi++) {
                for (int ti = 0; ti < s; ti++) {
                    for (int hi = 0; hi < h; hi++) {
                        int ni = bi * s + ti;
                        int qFlat = hi * (b * s) + ni;
                        int qPeOff = qPeLayoutBhs ? (bi * h * s + hi * s + ti) * r : (ni * h + hi) * r;
                        Float16ToFloat32(qNopePtr + qFlat * c, fqn.data(), c);
                        Float16ToFloat32(qPePtr + qPeOff, fqp.data(), r);

                        int sMin = 0, sMax = kvLen - 1;
                        if (causal && base >= 0) {
                            sMax = std::min(ti + base, kvLen - 1);
                        } else if (causal) {
                            sMax = std::min(ti, kvLen - 1);
                        }

                        std::vector<float> logits(kvLen, -1e9f);
                        float m = -1e9f;
                        for (int j = sMin; j <= sMax; j++) {
                            const uint8_t *ckvJ = getCkvAt(j);
                            const uint8_t *kpeJ = getKpeAt(j);
                            if (!ckvJ || !kpeJ) continue;
                            float accNope = (unitSizeCkv == 2) ? dotF16(qNopePtr + qFlat * c, (const uint16_t*)ckvJ, c) : dotF32(fqn.data(), (const float*)ckvJ, c);
                            float accPe = (unitSizeKpe == 2) ? dotF16(qPePtr + qPeOff, (const uint16_t*)kpeJ, r) : dotF32(fqp.data(), (const float*)kpeJ, r);
                            float acc = accNope + accPe;
                            logits[j] = acc * softmaxScale;
                            m = std::max(m, logits[j]);
                        }

                        float denom = 0.f;
                        for (int j = sMin; j <= sMax; j++) {
                            logits[j] = expf(logits[j] - m);
                            denom += logits[j];
                        }
                        if (denom < 1e-9f) denom = 1e-9f;
                        for (int j = sMin; j <= sMax; j++) logits[j] /= denom;

                        std::fill(fo.begin(), fo.end(), 0.f);
                        for (int j = sMin; j <= sMax; j++) {
                            const uint8_t *ckvJ = getCkvAt(j);
                            if (!ckvJ) continue;
                            float p = logits[j];
                            if (unitSizeCkv == 2) {
                                const uint16_t *vj = (const uint16_t*)ckvJ;
                                for (int d = 0; d < c; d++) fo[d] += p * half_to_float(vj[d]);
                            } else {
                                const float *vj = (const float*)ckvJ;
                                for (int d = 0; d < c; d++) fo[d] += p * vj[d];
                            }
                        }
                        Float32ToFloat16(fo.data(), outPtr + qFlat * c, c);
                    }
                }
            }
        }
    }

    void CpuCopyKVCacheOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                   const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return;
    }

    void CpuCopyKVCacheOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &oldCache = *(datas.find("oldCache")->second);
        Data &newCache = *(datas.find("newCache")->second);

        int oldBsStart = intParams.find("oldBsStart") != intParams.end() ? intParams.find("oldBsStart")->second : -1;
        int newBsStart = intParams.find("newBsStart") != intParams.end() ? intParams.find("newBsStart")->second : -1;
        int bs = intParams.find("bs") != intParams.end() ? intParams.find("bs")->second : -1;
        int offset = intParams.find("offset") != intParams.end() ? intParams.find("offset")->second : -1;

        int unitSize = oldCache.unitSize;
        for (int o = 0; o < bs; o++) {
            uint8_t *cur = newCache.cpuData + (newBsStart + o) * newCache.strides[0] * unitSize;
            cur += offset * newCache.strides[1] * unitSize;
            uint8_t *old = oldCache.cpuData + (oldBsStart + o) * oldCache.strides[0] * unitSize;
            memcpy(cur, old, oldCache.dims[1] * oldCache.dims[2] * unitSize);
        }
    }

    void CpuEmbedding::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "Embedding's weight's dim should be 2.\n");
        AssertInFastLLM(weight.dataType == DataType::FLOAT32 ||
                        weight.dataType == DataType::FLOAT16 ||
                        weight.dataType == DataType::BFLOAT16, "Embedding's weight's type should be float32 or float16 or bfloat16.\n");
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Embedding's input's type should be float32 or float16.\n");

        weight.weightType = WeightType::EMBEDDING;
        int vocabSize = weight.dims[0], embSize = weight.dims[1];
        std::vector <int> dims = input.dims;
        dims.push_back(embSize);

        output.dataType = input.dataType;
        if (weight.dataType == DataType::FLOAT16) {
            output.dataType = DataType::FLOAT16;
        }
        output.Resize(dims);
    }

    void CpuEmbedding::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);;

        output.Allocate();

        int vocabSize = weight.dims[0], embSize = weight.dims[1];
        uint64_t inputLen = input.Count(0);

        float *inputData = (float*)input.cpuData;
        std::vector <float> tempInputData;
        if (input.dataType != DataType::FLOAT32) {
            tempInputData.resize(inputLen);
            inputData = tempInputData.data();

            if (input.dataType == DataType::FLOAT16) {
                for (int i = 0; i < inputLen; i++) {
                    inputData[i] = half_to_float(((uint16_t*)input.cpuData)[i]);
                }
            } else {
                ErrorInFastLLM("Embedding error: unsupport dataType.\n");
            }
        }

        auto writeFloatOutputRow = [&](int row, const float *src) {
            uint64_t offset = (uint64_t)row * embSize;
            if (output.dataType == DataType::FLOAT32) {
                memcpy((float*)output.cpuData + offset, src, (uint64_t)embSize * sizeof(float));
            } else if (output.dataType == DataType::FLOAT16) {
                Float32ToFloat16(const_cast<float*>(src), (uint16_t*)output.cpuData + offset, embSize);
            } else if (output.dataType == DataType::BFLOAT16) {
                Float32ToBFloat16(const_cast<float*>(src), (uint16_t*)output.cpuData + offset, embSize);
            } else {
                ErrorInFastLLM("Embedding error: unsupport output dataType.\n");
            }
        };

        auto writePackedOutputRow = [&](int row, const uint16_t *src, DataType srcType) {
            uint64_t offset = (uint64_t)row * embSize;
            if (srcType == output.dataType) {
                memcpy((uint8_t*)output.cpuData + offset * output.unitSize, src, (uint64_t)embSize * output.unitSize);
            } else if (srcType == DataType::FLOAT16) {
                if (output.dataType == DataType::FLOAT32) {
                    for (int j = 0; j < embSize; j++) {
                        ((float*)output.cpuData)[offset + j] = half_to_float(src[j]);
                    }
                } else if (output.dataType == DataType::BFLOAT16) {
                    Float16ToBFloat16(const_cast<uint16_t*>(src), (uint16_t*)output.cpuData + offset, embSize);
                } else {
                    ErrorInFastLLM("Embedding error: unsupport FLOAT16 output dataType.\n");
                }
            } else if (srcType == DataType::BFLOAT16) {
                if (output.dataType == DataType::FLOAT32) {
                    BFloat16ToFloat32(const_cast<uint16_t*>(src), (float*)output.cpuData + offset, embSize);
                } else if (output.dataType == DataType::FLOAT16) {
                    BFloat16ToFloat16(const_cast<uint16_t*>(src), (uint16_t*)output.cpuData + offset, embSize);
                } else {
                    ErrorInFastLLM("Embedding error: unsupport BFLOAT16 output dataType.\n");
                }
            } else {
                ErrorInFastLLM("Embedding error: unsupport weight dataType.\n");
            }
        };

        auto getToken = [&](int row) {
            float tokenValue = inputData[row];
            if (!std::isfinite(tokenValue)) {
                std::ostringstream oss;
                oss << "Embedding error: token is not finite. row = " << row
                    << ", value = " << tokenValue
                    << ", inputType = " << (int)input.dataType
                    << ", outputType = " << (int)output.dataType
                    << ", weightType = " << (int)weight.dataType << "\n";
                ErrorInFastLLM(oss.str());
            }
            int token = (int)(tokenValue + 1e-9);
            if (token < 0 || token >= vocabSize) {
                std::ostringstream oss;
                oss << "Embedding error: token out of range. row = " << row
                    << ", token = " << token
                    << ", vocabSize = " << vocabSize
                    << ", raw = " << tokenValue
                    << ", inputType = " << (int)input.dataType
                    << ", outputType = " << (int)output.dataType
                    << ", weightType = " << (int)weight.dataType << "\n";
                ErrorInFastLLM(oss.str());
            }
            return token;
        };

        if (GetLowMemMode() && !weight.fileName.empty()) {
            FILE *fi = fopen(weight.fileName.c_str(), "rb");
            if (fi == nullptr) {
                ErrorInFastLLM("Embedding error: failed to open low-memory weight file " + weight.fileName + ".\n");
            }
            if (weight.dataType == DataType::FLOAT32) {
                std::vector<float> weightRow(embSize);
                for (int i = 0; i < inputLen; i++) {
                    int token = getToken(i);
#if defined(_WIN32) or defined(_WIN64)
                    _fseeki64(fi, (long long)token * embSize * sizeof(float) + weight.filePos, 0);
#else
                    fseek(fi, (long long)token * embSize * sizeof(float) + weight.filePos, 0);
#endif
                    int ret = fread(weightRow.data(), sizeof(float), embSize, fi);
                    writeFloatOutputRow(i, weightRow.data());
                }
            } else {
                std::vector<uint16_t> weightRow(embSize);
                for (int i = 0; i < inputLen; i++) {
                    int token = getToken(i);
#if defined(_WIN32) or defined(_WIN64)
                    _fseeki64(fi, (long long)token * embSize * sizeof(uint16_t) + weight.filePos, 0);
#else
                    fseek(fi, (long long)token * embSize * sizeof(uint16_t) + weight.filePos, 0);
#endif
                    int ret = fread(weightRow.data(), sizeof(uint16_t), embSize, fi);
                    writePackedOutputRow(i, weightRow.data(), weight.dataType);
                }
            }
            fclose(fi);
        } else {
            if (weight.dataType == DataType::FLOAT32) {
                float *weightData = (float *) weight.cpuData;
                for (int i = 0; i < inputLen; i++) {
                    int token = getToken(i);
                    writeFloatOutputRow(i, weightData + (uint64_t)token * embSize);
                }
            } else {
                uint16_t *weightData = (uint16_t *) weight.cpuData;
                for (int i = 0; i < inputLen; i++) {
                    int token = getToken(i);
                    writePackedOutputRow(i, weightData + (uint64_t)token * embSize, weight.dataType);
                }
            }
        }
    }

    void CpuEmbeddingDirect::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "EmbeddingDirect's weight's dim should be 2.\n");
        AssertInFastLLM(weight.dataType == DataType::FLOAT32 ||
                        weight.dataType == DataType::FLOAT16 ||
                        weight.dataType == DataType::BFLOAT16, "EmbeddingDirect's weight's type should be float32 or float16 or bfloat16.\n");
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "EmbeddingDirect's input's type should be float32 or float16.\n");

        weight.weightType = WeightType::EMBEDDING;
        int vocabSize = weight.dims[0], embSize = weight.dims[1];
        std::vector <int> dims = input.dims;
        dims.push_back(embSize);

        output.dataType = weight.dataType;
        output.Resize(dims);
    }

    void CpuEmbeddingDirect::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);;

        output.Allocate();

        int vocabSize = weight.dims[0], embSize = weight.dims[1];
        uint64_t inputLen = input.Count(0);

        float *inputData = (float*)input.cpuData;
        std::vector <float> tempInputData;
        if (input.dataType != DataType::FLOAT32) {
            tempInputData.resize(inputLen);
            inputData = tempInputData.data();
            if (input.dataType == DataType::FLOAT16) {
                for (int i = 0; i < inputLen; i++) {
                    inputData[i] = half_to_float(((uint16_t*)input.cpuData)[i]);
                }
            } else {
                ErrorInFastLLM("EmbeddingDirect error: unsupport input dataType.\n");
            }
        }

        int unitSize = weight.unitSize;
        if (GetLowMemMode()) {
            FILE *fi = fopen(weight.fileName.c_str(), "rb");
            uint8_t *outputData = (uint8_t *) output.cpuData;
            for (int i = 0; i < inputLen; i++) {
                int token = (int) (inputData[i] + 1e-9);
#if defined(_WIN32) or defined(_WIN64)
                _fseeki64(fi, (long long)token * embSize * unitSize + weight.filePos, 0);
#else
                fseek(fi, (long long)token * embSize * unitSize + weight.filePos, 0);
#endif
                int ret = fread(outputData + (uint64_t)i * embSize * unitSize, unitSize, embSize, fi);
            }
            fclose(fi);
        } else {
            uint8_t *outputData = (uint8_t *) output.cpuData;
            uint8_t *weightData = (uint8_t *) weight.cpuData;
            for (int i = 0; i < inputLen; i++) {
                int token = (int) (inputData[i] + 1e-9);
                memcpy(outputData + (uint64_t)i * embSize * unitSize, weightData + (uint64_t)token * embSize * unitSize, (uint64_t)embSize * unitSize);
            }
        }
    }

    void CpuLayerNormOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &gamma = *(datas.find("gamma")->second);
        Data &beta = *(datas.find("beta")->second);

        output.Allocate();

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];

        float *mean = new float[inner], *var = new float[inner];
        float *inputData = (float *) input.cpuData;
        float *outputData = (float *) output.cpuData;
        float *gammaData = (float *) gamma.cpuData;
        float *betaData = (float *) beta.cpuData;

        if (inner == 1) {
            for (int i = 0; i < outer; i++) {
                float mean = 0.f, s2 = 0.f, var = 0.f;
                int j = 0;
#ifdef __aarch64__
                float32x4_t sums = vdupq_n_f32(0.0);
                    float32x4_t sums2 = vdupq_n_f32(0.0);
                    for (; j + 3 < channels; j += 4) {
                        float32x4_t vi = vld1q_f32(inputData + j);
                        sums = vaddq_f32(sums, vi);
                        sums2 = vaddq_f32(sums2, vmulq_f32(vi, vi));
                    }
                    mean = sums[0] + sums[1] + sums[2] + sums[3];
                    s2 = sums2[0] + sums2[1] + sums2[2] + sums2[3];
#endif
#ifdef __AVX2__
                __m256 sum_vec = _mm256_setzero_ps();
                __m256 squared_sum_vec = _mm256_setzero_ps();

                for (; j < channels - 7; j += 8) {
                    __m256 data_vec = _mm256_loadu_ps(inputData + j);
                    sum_vec = _mm256_add_ps(sum_vec, data_vec);

                    __m256 squared_data_vec = _mm256_mul_ps(data_vec, data_vec);
                    squared_sum_vec = _mm256_add_ps(squared_sum_vec, squared_data_vec);
                }

                float sum_array[8];
                _mm256_storeu_ps(sum_array, sum_vec);
                mean = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                            sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

                float squared_sum_array[8];
                _mm256_storeu_ps(squared_sum_array, squared_sum_vec);
                s2 = squared_sum_array[0] + squared_sum_array[1] +
                                    squared_sum_array[2] + squared_sum_array[3] +
                                    squared_sum_array[4] + squared_sum_array[5] +
                                    squared_sum_array[6] + squared_sum_array[7];
#endif
                for (; j < channels; j++) {
                    mean += inputData[j];
                    s2 += inputData[j] * inputData[j];
                }
                mean /= channels;
                var = sqrt(s2 / channels - mean*mean + 1e-10);
                j = 0;
#ifdef __aarch64__
                float32x4_t means = vdupq_n_f32(mean);
                    float32x4_t vars = vdupq_n_f32(1.0 / var);
                    for (; j + 3 < channels; j += 4) {
                        float32x4_t va = vld1q_f32(gammaData + j), vb = vld1q_f32(betaData + j);
                        float32x4_t vi = vld1q_f32(inputData + j);
                        float32x4_t vo = vaddq_f32(vmulq_f32(vmulq_f32(vsubq_f32(vi, means), vars), va), vb);
                        vst1q_f32(outputData + j, vo);
                    }
#endif
                for (; j < channels; j++) {
                    float a = gammaData[j], b = betaData[j];
                    outputData[j] = (inputData[j] - mean) / var * a + b;
                }

                inputData += channels;
                outputData += channels;
            }
            return;
        } else {
            for (int i = 0; i < outer; i++) {
                std::fill(mean, mean + inner, 0.f);
                std::fill(var, var + inner, 0.f);
                float *inputWalk = inputData;
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        mean[k] += *inputWalk++;
                    }
                }
                for (int k = 0; k < inner; k++) {
                    mean[k] /= channels;
                }
                inputWalk = inputData;
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        float x = (*inputWalk++) - mean[k];
                        var[k] += x * x;
                    }
                }
                for (int k = 0; k < inner; k++) {
                    var[k] = sqrt(var[k] / channels + 1e-5);
                }

                inputWalk = inputData;
                float *outputWalk = outputData;
                for (int j = 0; j < channels; j++) {
                    float a = gammaData[j], b = betaData[j];
                    for (int k = 0; k < inner; k++) {
                        *outputWalk++ = ((*inputWalk++) - mean[k]) / var[k] * a + b;
                    }
                }

                inputData += channels * inner;
                outputData += channels * inner;
            }
            delete[] mean;
            delete[] var;
        }
    }

    struct MultiThreadRMSNormFloatOp : MultiThreadBaseOp {
        float *input, *output, *weight;
        int outer, channels;
        float eps;        

        MultiThreadRMSNormFloatOp (float *output, float *input, float *weight, int outer, int channels, float eps) : 
            input(input), output(output), weight(weight), outer(outer), channels(channels), eps(eps) {}

        void Run() {
            for (int i = 0; i < outer; i++) {
                float mean = 0.f;
                int j = 0;
#ifdef __aarch64__
                float32x4_t sums = vdupq_n_f32(0.0);
                for (; j + 3 < channels; j += 4) {
                    float32x4_t vi = vld1q_f32(input + j);
                    sums = vaddq_f32(sums, vmulq_f32(vi, vi));
                }
                mean = sums[0] + sums[1] + sums[2] + sums[3];
#endif
                for (; j < channels; j++) {
                    mean += input[j] * input[j];
                }
                float scale = 1.0 / sqrt(mean / channels + eps);
                j = 0;
#ifdef __aarch64__
                float32x4_t vscale = vdupq_n_f32(scale);
                for (; j + 3 < channels; j += 4) {
                    float32x4_t vi = vld1q_f32(input + j);
                    float32x4_t vw = vld1q_f32(weight + j);
                    vst1q_f32(output + j, vmulq_f32(vmulq_f32(vi, vscale), vw));
                }
#endif
                for (; j < channels; j++) {
                    output[j] = input[j] * scale * weight[j];
                }

                input += channels;
                output += channels;
            }
        }
    };

    static void RunMultiThreadRMSNormFloat(float *output, float *input, float *weight, int outer, int channels, float eps, AliveThreadPool *pool) {
        if (outer == 1) {
            (MultiThreadRMSNormFloatOp(output, input, weight, outer, channels, eps)).Run();
            return;
        }
        int threadNum = pool->threads.size();
        int per = outer / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadRMSNormFloatOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
            ops.push_back(new MultiThreadRMSNormFloatOp(output + cur * channels, input + cur * channels, weight, end - cur, channels, eps));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    struct MultiThreadRMSNormFloat16Op : MultiThreadBaseOp {
        uint16_t *input, *output;
        float *weight;
        int outer, channels;
        float eps;

        MultiThreadRMSNormFloat16Op (uint16_t *output, uint16_t *input, float *weight, int outer, int channels, float eps) :
            input(input), output(output), weight(weight), outer(outer), channels(channels), eps(eps) {}

        void Run() {
            for (int i = 0; i < outer; i++) {
                float mean = 0.f;
                int j = 0;
                for (; j < channels; j++) {
                    float x = fp16tofp32.dict[input[j]];
                    mean += x * x;
                }
                float scale = 1.0 / sqrt(mean / channels + eps);
                j = 0;
                for (; j < channels; j++) {
                    output[j] = float_to_half(fp16tofp32.dict[input[j]] * scale * weight[j]);
                }

                input += channels;
                output += channels;
            }
        }
    };

    static void RunMultiThreadRMSNormFloat16(uint16_t *output, uint16_t *input, float *weight, int outer, int channels, float eps, AliveThreadPool *pool) {
        if (outer <= 1 || (long long)outer * channels < 65536) {
            (MultiThreadRMSNormFloat16Op(output, input, weight, outer, channels, eps)).Run();
            return;
        }
        int threadNum = std::min((int)pool->threads.size(), outer);
        int per = outer / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadRMSNormFloat16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
            ops.push_back(new MultiThreadRMSNormFloat16Op(output + cur * channels, input + cur * channels, weight, end - cur, channels, eps));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    struct MultiThreadRMSNormBFloat16Op : MultiThreadBaseOp {
        uint16_t *input, *output;
        float *weight;
        int outer, channels;
        float eps;

        MultiThreadRMSNormBFloat16Op (uint16_t *output, uint16_t *input, float *weight, int outer, int channels, float eps) :
            input(input), output(output), weight(weight), outer(outer), channels(channels), eps(eps) {}

        void Run() {
            for (int i = 0; i < outer; i++) {
                float mean = 0.f;
                int j = 0;
                for (; j < channels; j++) {
                    float x = bf16tofp32.dict[input[j]];
                    mean += x * x;
                }
                float scale = 1.0 / sqrt(mean / channels + eps);
                j = 0;
                for (; j < channels; j++) {
                    float val = bf16tofp32.dict[input[j]] * scale * weight[j];
                    uint32_t tmp;
                    memcpy(&tmp, &val, sizeof(tmp));
                    output[j] = (uint16_t)(tmp >> 16);
                }

                input += channels;
                output += channels;
            }
        }
    };

    static void RunMultiThreadRMSNormBFloat16(uint16_t *output, uint16_t *input, float *weight, int outer, int channels, float eps, AliveThreadPool *pool) {
        if (outer <= 1 || (long long)outer * channels < 65536) {
            (MultiThreadRMSNormBFloat16Op(output, input, weight, outer, channels, eps)).Run();
            return;
        }
        int threadNum = std::min((int)pool->threads.size(), outer);
        int per = outer / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadRMSNormBFloat16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
            ops.push_back(new MultiThreadRMSNormBFloat16Op(output + cur * channels, input + cur * channels, weight, end - cur, channels, eps));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuRMSNormOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-5;
        int dimsLen = input.dims.size();
        int axis = dimsLen - 1;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];

        if (input.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *outputData = (float *) output.cpuData;
            float *weightData = (float *) weight.cpuData;
            RunMultiThreadRMSNormFloat(outputData, inputData, weightData, outer, channels, eps, GetAlivePool());
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t *) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;
            float *weightData = (float *) weight.cpuData;

            RunMultiThreadRMSNormFloat16(outputData, inputData, weightData, outer, channels, eps, GetAlivePool());
        } else if (input.dataType == DataType::BFLOAT16) {
            uint16_t *inputData = (uint16_t *) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;
            float *weightData = (float *) weight.cpuData;

            RunMultiThreadRMSNormBFloat16(outputData, inputData, weightData, outer, channels, eps, GetAlivePool());
        } else {
            ErrorInFastLLM("RMSNorm error: unsupport dataType.\n");
        }
    }

    void CpuRMSNormPartOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-5;
        int start = intParams.find("start")->second;
        int end = intParams.find("end")->second;
        int dimsLen = input.dims.size();
        int axis = dimsLen - 1;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int partChannels = end - start;

        if (input.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *outputData = (float *) output.cpuData;
            float *weightData = (float *) weight.cpuData;

            for (int i = 0; i < outer; i++) {
                for (int j = 0; j < start; j++) {
                    outputData[j] = inputData[j];
                }
                float mean = 0.f;
                for (int j = start; j < end; j++) {
                    mean += inputData[j] * inputData[j];
                }
                float scale = 1.0 / sqrt(mean / partChannels + eps);
                for (int j = start; j < end; j++) {
                    outputData[j] = inputData[j] * scale * weightData[j - start];
                }
                for (int j = end; j < channels; j++) {
                    outputData[j] = inputData[j];
                }
                inputData += channels;
                outputData += channels;
            }
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t *) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;
            float *weightData = (float *) weight.cpuData;

            for (int i = 0; i < outer; i++) {
                for (int j = 0; j < start; j++) {
                    outputData[j] = inputData[j];
                }
                float mean = 0.f;
                for (int j = start; j < end; j++) {
                    float x = fp16tofp32.dict[inputData[j]];
                    mean += x * x;
                }
                float scale = 1.0 / sqrt(mean / partChannels + eps);
                for (int j = start; j < end; j++) {
                    outputData[j] = float_to_half(fp16tofp32.dict[inputData[j]] * scale * weightData[j - start]);
                }
                for (int j = end; j < channels; j++) {
                    outputData[j] = inputData[j];
                }
                inputData += channels;
                outputData += channels;
            }
        } else if (input.dataType == DataType::BFLOAT16) {
            uint16_t *inputData = (uint16_t *) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;
            float *weightData = (float *) weight.cpuData;

            for (int i = 0; i < outer; i++) {
                for (int j = 0; j < start; j++) {
                    outputData[j] = inputData[j];
                }
                float mean = 0.f;
                for (int j = start; j < end; j++) {
                    float x = bf16tofp32.dict[inputData[j]];
                    mean += x * x;
                }
                float scale = 1.0 / sqrt(mean / partChannels + eps);
                for (int j = start; j < end; j++) {
                    float val = bf16tofp32.dict[inputData[j]] * scale * weightData[j - start];
                    uint32_t tmp;
                    memcpy(&tmp, &val, sizeof(tmp));
                    outputData[j] = (uint16_t)(tmp >> 16);
                }
                for (int j = end; j < channels; j++) {
                    outputData[j] = inputData[j];
                }
                inputData += channels;
                outputData += channels;
            }
        } else {
            ErrorInFastLLM("RMSNormPart error: unsupport dataType.\n");
        }
    }

    bool CpuConv1DPerChannel::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return true;
    }

    void CpuConv1DPerChannel::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        int inputChannels = intParams.find("inputChannels")->second;
        int outputChannels = intParams.find("outputChannels")->second;
        int kernel = intParams.find("kernel")->second;
        int pad = intParams.find("pad")->second;
        int stride = intParams.find("stride")->second;
        
        AssertInFastLLM(weight.dims.size() == 3, "Conv1D's weight's shape's size should be 3.\n");
        AssertInFastLLM(input.dims[1] == inputChannels, "Conv1D's input's shape error.\n");

        weight.weightType = WeightType::CONV1D;

        std::vector <int> dims = input.dims;
        int inputLen = dims[2];
        int outputLen = (inputLen + pad + pad - kernel) / stride + 1;
        dims[1] = outputChannels;
        dims[2] = outputLen;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    struct MultiThreadConv1DPerChannelOp : MultiThreadBaseOp {
        float *floatInput, *floatOutput, *floatWeight, *floatBias;
        int batchSize, inputLength, outputLength;
        int inputChannels, outputChannels, kernelSize, padding, stride, groups;
        int channelsPerGroup, outputChannelsPerGroup;
        int st, end;

        MultiThreadConv1DPerChannelOp (float *floatInput, float *floatOutput, float *floatWeight,
                                       float *floatBias, int batchSize, int inputLength, int outputLength,
                                       int inputChannels, int outputChannels, int kernelSize, int padding,
                                       int stride, int groups, int channelsPerGroup,
                                       int outputChannelsPerGroup, int st, int end) :
            floatInput(floatInput), floatOutput(floatOutput), floatWeight(floatWeight), floatBias(floatBias),
            batchSize(batchSize), inputLength(inputLength), outputLength(outputLength),
            inputChannels(inputChannels), outputChannels(outputChannels), kernelSize(kernelSize),
            padding(padding), stride(stride), groups(groups), channelsPerGroup(channelsPerGroup),
            outputChannelsPerGroup(outputChannelsPerGroup), st(st), end(end) {}

        void Run() {
            for (int b = 0; b < batchSize; b++) {
                float *batchInput = floatInput + (long long)b * (inputChannels * inputLength);
                float *batchOutput = floatOutput + (long long)b * (outputChannels * outputLength);

                for (int g = st; g < end; g++) {
                    for (int oc = 0; oc < outputChannelsPerGroup; oc++) {
                        int globalOc = g * outputChannelsPerGroup + oc;
                        float *curWeight = floatWeight + globalOc * (channelsPerGroup * kernelSize);
                        float *curOutput = batchOutput + (long long)globalOc * outputLength;

                        for (int ol = 0; ol < outputLength; ol++) {
                            int il = ol * stride - padding;
                            float value = floatBias ? floatBias[globalOc] : 0.0f;

                            for (int ic = 0; ic < channelsPerGroup; ic++) {
                                int globalIc = g * channelsPerGroup + ic;
                                float *curInput = batchInput + (long long)globalIc * inputLength;

                                for (int k = 0; k < kernelSize; k++) {
                                    float inputValue = 0;
                                    int inputPos = il + k;

                                    if (inputPos >= 0 && inputPos < inputLength) {
                                        inputValue = curInput[inputPos];
                                    }

                                    value += inputValue * curWeight[ic * kernelSize + k];
                                }
                            }

                            curOutput[ol] = value;
                        }
                    }
                }
            }
        }
    };

    void CpuConv1DPerChannel::Run(const std::string &opType, const fastllm::DataDict &datas,
                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);
        output.Allocate();
        int inputChannels = intParams.find("inputChannels")->second;
        int outputChannels = intParams.find("outputChannels")->second;
        int kernelSize = intParams.find("kernel")->second;
        int padding = intParams.find("pad")->second;
        int stride = intParams.find("stride")->second;
        int groups = inputChannels;  // 组数等于通道数，实现逐通道卷积

        // 如果有groups参数，使用它
        if (intParams.find("groups") != intParams.end()) {
            groups = intParams.find("groups")->second;
        }

        std::vector<int> dims = input.dims;
        int batchSize = dims[0];
        int inputLength = dims[2];
        int outputLength = (inputLength + 2 * padding - kernelSize) / stride + 1;
        float *floatInput = (float*)input.cpuData;
        float *floatOutput = (float*)output.cpuData;

        float *floatWeight = (float*)weight.cpuData;
        float *floatBias = bias.dims.size() > 0 ? nullptr : (float*)bias.cpuData;

        std::vector <float> floatInputVector, floatOutputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            floatOutputVector.resize(output.Count(0));
            floatInput = (float*)floatInputVector.data();
            floatOutput = (float*)floatOutputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, floatInput, (int)floatInputVector.size());
        }

        int channelsPerGroup = inputChannels / groups;   // 对于逐通道卷积，这是1
        int outputChannelsPerGroup = outputChannels / groups;  // 对于逐通道卷积，这也是1

        if ((long long)batchSize * groups * outputLength * kernelSize * channelsPerGroup < 65536 ||
            groups <= 1) {
            MultiThreadConv1DPerChannelOp(floatInput, floatOutput, floatWeight, floatBias,
                                          batchSize, inputLength, outputLength, inputChannels,
                                          outputChannels, kernelSize, padding, stride, groups,
                                          channelsPerGroup, outputChannelsPerGroup, 0, groups).Run();
        } else {
            auto *pool = GetAlivePool();
            int threadNum = std::min((int)pool->threads.size(), groups);
            int per = groups / threadNum;
            std::vector<fastllm::MultiThreadConv1DPerChannelOp*> ops;
            int cur = 0;
            for (int i = 0; i < threadNum; i++) {
                int end = (i == threadNum - 1 ? groups : cur + per + (cur + per * (threadNum - i) < groups));
                ops.push_back(new MultiThreadConv1DPerChannelOp(
                    floatInput, floatOutput, floatWeight, floatBias,
                    batchSize, inputLength, outputLength, inputChannels,
                    outputChannels, kernelSize, padding, stride, groups,
                    channelsPerGroup, outputChannelsPerGroup, cur, end));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(floatOutput, (uint16_t*)output.cpuData, (int)floatOutputVector.size());
        }
    }

    bool CpuConv2DOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return true;
    }

    void CpuConv2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate(0.0f);

        int inputChannels = intParams.find("inputChannels")->second;
        int outputChannels = intParams.find("outputChannels")->second;
        int kernelH = intParams.find("kernelH")->second;
        int kernelW = intParams.find("kernelW")->second;
        int padH = intParams.find("padH")->second;
        int padW = intParams.find("padW")->second;
        int strideH = intParams.find("strideH")->second;
        int strideW = intParams.find("strideW")->second;
        
        std::vector <int> dims = input.dims;
        int inputHeight = dims[2], inputWidth = dims[3];
        int outputHeight = (inputHeight + padH + padH - kernelH) / strideH + 1;
        int outputWidth = (inputWidth + padW + padW - kernelW) / strideW + 1;

        float *floatInput = (float*)input.cpuData;
        float *floatWeight = (float*)weight.cpuData;
        float *floatBias = (float*)bias.cpuData;
        float *floatOutput = (float*)output.cpuData;
        for (int oc = 0; oc < outputChannels; oc++) {
            float *startWeight = (float*)floatWeight + oc * (inputChannels * kernelH * kernelW);
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    int ih = oh * strideH - padH;
                    int iw = ow * strideW - padW;
                    float value = floatBias[oc];
                    float *curWeight = startWeight;
                    for (int c = 0; c < inputChannels; c++) {
                        float *curInput = (float*)floatInput + c * inputHeight * inputWidth;
                        for (int h = 0; h < kernelH; h++) {
                            for (int w = 0; w < kernelW; w++) {
                                float inputValue = 0;
                                if (ih + h >= 0 && ih + h < inputHeight && iw + w >= 0 && iw + w < inputWidth) {
                                    inputValue = curInput[(ih + h) * inputWidth + (iw + w)];
                                }
                                value += inputValue * (*(curWeight++));
                            }
                        }
                    }

                    *(floatOutput++) = value;
                }
            }
        }
    }

    void CpuConv2DOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        int inputChannels = intParams.find("inputChannels")->second;
        int outputChannels = intParams.find("outputChannels")->second;
        int kernelH = intParams.find("kernelH")->second;
        int kernelW = intParams.find("kernelW")->second;
        int padH = intParams.find("padH")->second;
        int padW = intParams.find("padW")->second;
        int strideH = intParams.find("strideH")->second;
        int strideW = intParams.find("strideW")->second;
        
        AssertInFastLLM(weight.dims.size() == 4, "Conv2D's weight's shape's size should be 4.\n");
        AssertInFastLLM(input.dims[1] == inputChannels, "Conv2D's input's shape error.\n");

        weight.weightType = WeightType::CONV2D;

        std::vector <int> dims = input.dims;
        int inputHeight = dims[2], inputWidth = dims[3];
        int outputHeight = (inputHeight + padH + padH - kernelH) / strideH + 1;
        int outputWidth = (inputWidth + padW + padW - kernelW) / strideW + 1;
        dims[1] = outputChannels;
        dims[2] = outputHeight;
        dims[3] = outputWidth;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void DoCpuLinearReshape(Data &input, Data &weight, Data &output) {
        weight.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuLinearOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");

        DoCpuLinearReshape(input, weight, output);
    }

    void MultiThreadInt4GroupLinearOp::Run() {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                for (int g = 0; g < group; g++) {
                    int gst = g * groupCnt;
                    int gend = std::min((g + 1) * groupCnt, m);
                    int l = gst;
                    float curMin = fp16tofp32.dict[mins[j * group + g]];
                    float curScale = fp16tofp32.dict[scales[j * group + g]];
#ifdef __AVX2__
                    __m256 now_vec = _mm256_setzero_ps();
                    const __m256 scale_vec = _mm256_set1_ps(curScale);
                    const __m256 min_vec = _mm256_set1_ps(curMin);
                        
                    for (; l + 8 <= gend; l += 8) {
                        // 计算权重索引（每次处理4个字节）
                        size_t weight_offset = (j * m + l) / 2;
                        const uint8_t* weight_ptr = &weightData[weight_offset];
                            
                        // 加载4个权重字节
                        __m128i v = _mm_loadl_epi64((const __m128i*)weight_ptr);
                            
                        // 拆分高/低4位
                        __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi8(0x0F));
                        __m128i lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
                            
                        // 交错排列成 [hi0, lo0, hi1, lo1, ...]
                        __m128i hi_lo = _mm_unpacklo_epi8(hi, lo);
                            
                        // 将8个字节扩展为8个int32
                        __m128i lo_nib = _mm_cvtepu8_epi32(hi_lo);
                        __m128i hi_nib = _mm_cvtepu8_epi32(_mm_srli_si128(hi_lo, 4));
                        __m256i nibbles = _mm256_set_m128i(hi_nib, lo_nib);
                        
                        // 转换为浮点数并计算权重
                        __m256 weights = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(nibbles), scale_vec, min_vec);
                        
                        // 加载8个输入元素
                        const float* input_ptr = &inputData[i * m + l];
                        __m256 input = _mm256_loadu_ps(input_ptr);
                            
                        // 乘积累加
                        now_vec = _mm256_fmadd_ps(input, weights, now_vec);
                    }
                        
                    // 水平求和
                    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(now_vec, 1),
                                                _mm256_castps256_ps128(now_vec));
                    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
                    now += _mm_cvtss_f32(sum128);
#endif
                    // 处理剩余不足8个元素的情况
                    for (; l + 1 < gend; l += 2) {
                        uint8_t v = weightData[(j * m + l) / 2];
                        now += inputData[i * m + l] * (curMin + (v >> 4) * curScale);
                        now += inputData[i * m + l + 1] * (curMin + (v & 0x0F) * curScale);
                    }

                    for (; l < gend; l++) {
                        int id = (j * m + l) / 2;
                        float weight = 0.0f;
                        if ((j * m + l) % 2) {
                            weight = curMin + (weightData[id] & 0xF) * curScale;
                        } else {
                            weight = curMin + (weightData[id] >> 4) * curScale;
                        }
                        now += inputData[i * m + l] * weight;
                    }
                }

                outputData[i * k + j] = now;
            }
        }
    }

    struct MultiThreadBase3GroupLinearOp : MultiThreadBaseOp {
        float *inputData;
        uint8_t *weightData;
        float *biasData, *outputData;
        int n, m, k, st, end, group, groupCnt;
        uint16_t *halfScales;

        MultiThreadBase3GroupLinearOp(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                           int n, int m, int k, int st, int end, int group, int groupCnt, uint16_t *halfScales) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end), group(group), groupCnt(groupCnt), halfScales(halfScales) {}

        void Run() {
            std::vector <uint8_t> base = {1, 3, 9, 27, 81};
            int bytesPerGroup = ((groupCnt - 1) / 5) + 1;   
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    for (int g = 0; g < group; g++) {
                        uint8_t *cur = weightData + j * group * bytesPerGroup + g * bytesPerGroup;
                        float sum = 0.0;
                        int l = 0;
                        for (; l < groupCnt && g * groupCnt + l < m; l++) {
                            sum += inputData[i * m + g * groupCnt + l] * (cur[l / 5] / base[l % 5] % 3 - 1);
                        }
                        now += sum * fp16tofp32.dict[halfScales[j * group + g]];
                    }
                    outputData[i * k + j] = now;
                }
            }
        }
    };

    // float的input, int8的weight, 直接计算得到float的output
    void Int8LinearPart(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        LowBitConfig *configs, int n, int m, int k, int st, int end) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
#ifdef __aarch64__
                float32x4_t scales = vdupq_n_f32(configs[j].scale);
                uint8x8_t zeros = vdup_n_u8(configs[j].zeroPoint);
                float32x4_t sum0 = {0, 0, 0, 0};
                float32x4_t sum1 = {0, 0, 0, 0};
                for (; l + 7 < m; l += 8) {
                    uint8x8_t a = vld1_u8(weightData + j * m + l);
                    uint16x8_t result = vsubl_u8(a, zeros);
                    int16x8_t sresult = vreinterpretq_s16_u16(result);
                    int16x4_t result1 = vget_low_s16(sresult);
                    int16x4_t result2 = vget_high_s16(sresult);
                    int32x4_t result3 = vmovl_s16(result1);
                    int32x4_t result4 = vmovl_s16(result2);
                    float32x4_t f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                    float32x4_t f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));

                    sum0 = vaddq_f32(sum0, vmulq_f32(vld1q_f32(inputData + i * m + l + 0), f1));
                    sum1 = vaddq_f32(sum1, vmulq_f32(vld1q_f32(inputData + i * m + l + 4), f2));
                }
                now += sum0[0] + sum0[1] + sum0[2] + sum0[3];
                now += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#endif

                for (; l < m; l++) {
                    now += inputData[i * m + l] * configs[j].invQuantization(weightData[j * m + l]);
                }

                outputData[i * k + j] = now;
            }
        }
    }

    // float的input, int4g的weight, 直接计算得到float的output
    void Int4GroupLinearPart(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                            LowBitConfig *configs, int n, int m, int k, int st, int end, int group, int groupCnt) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                
                for (int g = 0; g < group; g++) {
                    int gst = g * groupCnt;
                    int gend = std::min((g + 1) * groupCnt, m);
                    int l = gst;
#ifdef __aarch64__
                    float32x4_t scales = vdupq_n_f32(configs[j * group + g].scale);
                    uint8x8_t zeros = vdup_n_u8(configs[j * group + g].zeroPoint);
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    float32x4_t sum0 = {0, 0, 0, 0};
                    float32x4_t sum1 = {0, 0, 0, 0};

                    for (; l + 15 < gend; l += 16) {
                        uint8x8_t ori = vld1_u8(weightData + (j * m + l) / 2);
                        float32x4x2_t in0 = vld2q_f32(inputData + i * m + l + 0);
                        float32x4x2_t in1 = vld2q_f32(inputData + i * m + l + 8);
                        uint8x8_t a = vand_u8(ori, maskLow);
                        uint16x8_t result = vsubl_u8(a, zeros);
                        int16x8_t sresult = vreinterpretq_s16_u16(result);
                        int16x4_t result1 = vget_low_s16(sresult);
                        int16x4_t result2 = vget_high_s16(sresult);
                        int32x4_t result3 = vmovl_s16(result1);
                        int32x4_t result4 = vmovl_s16(result2);
                        float32x4_t f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                        float32x4_t f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));
                        sum0 = vaddq_f32(sum0, vmulq_f32(in0.val[1], f1));
                        sum1 = vaddq_f32(sum1, vmulq_f32(in1.val[1], f2));

                        a = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        result = vsubl_u8(a, zeros);
                        sresult = vreinterpretq_s16_u16(result);
                        result1 = vget_low_s16(sresult);
                        result2 = vget_high_s16(sresult);
                        result3 = vmovl_s16(result1);
                        result4 = vmovl_s16(result2);
                        f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                        f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));

                        sum0 = vaddq_f32(sum0, vmulq_f32(in0.val[0], f1));
                        sum1 = vaddq_f32(sum1, vmulq_f32(in1.val[0], f2));
                    }
                    now += sum0[0] + sum0[1] + sum0[2] + sum0[3];
                    now += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#endif
                    for (; l < gend; l++) {
                        int id = (j * m + l) / 2;
                        float weight = 0.0f;
                        if ((j * m + l) % 2) {
                            weight = configs[j * group + g].invQuantization(weightData[id] & 0xF);
                        } else {
                            weight = configs[j * group + g].invQuantization(weightData[id] >> 4);
                        }
                        now += inputData[i * m + l] * weight;
                    }
                }

                outputData[i * k + j] = now;
            }
        }
    }

    // float的input, int4的weight, 直接计算得到float的output
    void Int4LinearPart(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        LowBitConfig *configs, int n, int m, int k, int st, int end) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
#ifdef __aarch64__X
                float32x4_t scales = vdupq_n_f32(configs[j].scale);
                uint8x8_t zeros = vdup_n_u8(configs[j].zeroPoint);
                uint8x8_t maskHigh = vdup_n_u8(0xF0);
                uint8x8_t maskLow = vdup_n_u8(0xF);
                float32x4_t sum0 = {0, 0, 0, 0};
                float32x4_t sum1 = {0, 0, 0, 0};

                for (; l + 15 < m; l += 16) {
                    uint8x8_t ori = vld1_u8(weightData + (j * m + l) / 2);
                    float32x4x2_t in0 = vld2q_f32(inputData + i * m + l + 0);
                    float32x4x2_t in1 = vld2q_f32(inputData + i * m + l + 8);
                    uint8x8_t a = vand_u8(ori, maskLow);
                    uint16x8_t result = vsubl_u8(a, zeros);
                    int16x8_t sresult = vreinterpretq_s16_u16(result);
                    int16x4_t result1 = vget_low_s16(sresult);
                    int16x4_t result2 = vget_high_s16(sresult);
                    int32x4_t result3 = vmovl_s16(result1);
                    int32x4_t result4 = vmovl_s16(result2);
                    float32x4_t f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                    float32x4_t f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));
                    sum0 = vaddq_f32(sum0, vmulq_f32(in0.val[1], f1));
                    sum1 = vaddq_f32(sum1, vmulq_f32(in1.val[1], f2));

                    a = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                    result = vsubl_u8(a, zeros);
                    sresult = vreinterpretq_s16_u16(result);
                    result1 = vget_low_s16(sresult);
                    result2 = vget_high_s16(sresult);
                    result3 = vmovl_s16(result1);
                    result4 = vmovl_s16(result2);
                    f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                    f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));

                    sum0 = vaddq_f32(sum0, vmulq_f32(in0.val[0], f1));
                    sum1 = vaddq_f32(sum1, vmulq_f32(in1.val[0], f2));
                }
                now += sum0[0] + sum0[1] + sum0[2] + sum0[3];
                now += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#endif

                for (; l < m; l++) {
                    int id = (j * m + l) / 2;
                    float weight = 0.0f;
                    if ((j * m + l) % 2) {
                        weight = configs[j].invQuantization(weightData[id] & 0xF);
                    } else {
                        weight = configs[j].invQuantization(weightData[id] >> 4);
                    }
                    now += inputData[i * m + l] * weight;
                }

                outputData[i * k + j] = now;
            }
        }
    }

    struct MultiThreadLinearInt4Op : MultiThreadBaseOp {
        uint8_t *a;
        uint8_t *b;
        int32_t *c;
        int n, m, k, kstride;
        int *weightSums, *weightZeros;
        float *scales, *bias;
        LowBitConfig *config;
        int *inputSums;

        MultiThreadLinearInt4Op(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride,
                      int *weightSums, int *weightZeros, float *scales, float *bias, LowBitConfig *config,
                      int *inputSums) : a(a), b(b), c(c), n(n), m(m), k(k), kstride(kstride), 
                      weightSums(weightSums), weightZeros(weightZeros), scales(scales), bias(bias), config(config), inputSums(inputSums) {}
        
        void Run() {
            int block = 0;
            for (; block < n; block++) {
                uint32_t inputSum = inputSums[block];
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    int value = 0;
                    uint8_t *inputWalk = inputStart;
                    int j = 0;
#ifdef __ARM_FEATURE_DOTPROD
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x2_t sum0 = {0, 0};

                    for (; j + 15 < m; j += 16) {
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8x2_t in = vld2_u8(inputWalk + j);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        sum0 = vdot_u32(sum0, va, in.val[1]);
                        sum0 = vdot_u32(sum0, vb, in.val[0]);
                    }
                    value += sum0[0] + sum0[1];
#elif defined(__aarch64__)
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x4_t sum0 = {0, 0, 0, 0};

                    for (; j + 15 < m; j += 16) {
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8x2_t in = vld2_u8(inputWalk + j);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        sum0 = vpadalq_u16(sum0, vmull_u8(va, in.val[1]));
                        sum0 = vpadalq_u16(sum0, vmull_u8(vb, in.val[0]));
                    }
                    value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#elif defined(__AVX2__)
                    value += DotU4U8(weightWalk + i * m / 2, inputWalk, m);
                    j += m;
#endif
                    for (; j + 1 < m; j += 2) {
                        int id = (i * m + j) / 2;
                        value += (weightWalk[id] >> 4) * inputWalk[j];
                        value += (weightWalk[id] & 0xF) * inputWalk[j + 1];
                    }

                    for (; j < m; j++) {
                        int id = (i * m + j) / 2;
                        if ((i * m + j) % 2) {
                            value += (weightWalk[id] & 0xF) * inputWalk[j];
                        } else {
                            value += (weightWalk[id] >> 4) * inputWalk[j];
                        }
                    }

                    value -= weightSums[i] * config[block].zeroPoint;
                    value -= inputSum * weightZeros[i];
                    value += (int)config[block].zeroPoint * weightZeros[i] * m;

                    ((float*)c)[block * kstride + i] = scales[i] * config[block].scale * value +
                                                    (bias == nullptr ? 0.0 : bias[i]);
                }
            }
        }
    };

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyInt4MultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k,
                                 int *weightSums, int *weightZeros, float *scales, float *bias, std::vector <LowBitConfig> &configs, int threadNum) {
        std::vector <int> inputSums;
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = 0; j < m; j++) {
                sum += a[i * m + j];
            }
            inputSums.push_back(sum);
        }
        auto *pool = GetAlivePool();
        threadNum = pool->threads.size();
        int per = k / threadNum;
        int cur = 0;
        if (threadNum == 1) {
            MultiThreadLinearInt4Op(a, b + cur * m / 2, c + cur, n, m, k - cur, k,
                         weightSums + cur, weightZeros + cur, scales + cur,
                         (bias == nullptr ? (float*)nullptr : bias + cur), configs.data(), inputSums.data()).Run();
        } else {
            std::vector<fastllm::MultiThreadLinearInt4Op*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
                ops.push_back(new MultiThreadLinearInt4Op(a, b + cur * m / 2, c + cur, n, m, end - cur, k,
                                               weightSums + cur, weightZeros + cur, scales + cur,
                                               (bias == nullptr ? (float *) nullptr : bias + cur), configs.data(),
                                               inputSums.data()));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyInt4GroupMultiThreadLaunch(uint8_t *a, uint8_t *b, float *c, int n, int m, int k,
                                 int *weightSums, float *weightMins, float *scales, float *bias,
                                 std::vector <float> &inputSums, std::vector <float> &iscales, std::vector <float> &izeros,
                                 std::vector <LowBitConfig> &configs, int startTid, int threadNum, int group, int groupCnt,
                                 std::vector<fastllm::MultiThreadBaseOp*> &ops, 
                                 AliveThreadPool *pool) {
        int per = k / threadNum;
        int cur = 0;
        
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
            if (group > 1) {
                ops[startTid + i] = new MultiThreadLinearInt8Int4GroupOp(a, b + cur * m / 2, c + cur, n, m, end - cur, k,
                                           weightSums + cur * group, weightMins + cur * group, scales + cur * group,
                                           (bias == nullptr ? (float *) nullptr : bias + cur), iscales.data(), izeros.data(),
                                           inputSums.data(), group, groupCnt);
            } else {
                ops[startTid + i] = new MultiThreadLinearInt4NoZeroOp(a, b + cur * m / 2, (int32_t*)c + cur, n, m, end - cur, k,
                                           weightSums + cur * group, weightMins + cur * group, scales + cur * group,
                                           (bias == nullptr ? (float *) nullptr : bias + cur), configs.data(), inputSums.data());
            }
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[startTid + i]);
        }
    }

    void GetArrayMinMax(float *a, int len, float &minValue, float &maxValue) {
        int j = 0;
        minValue = 1e100;
        maxValue = -1e100;
#ifdef __aarch64__
        float32x4_t mins = vdupq_n_f32(1e100);
        float32x4_t maxs = vdupq_n_f32(-1e100);
        for (; j + 3 < len; j += 4) {
            float32x4_t v = vld1q_f32(a + j);
            mins = vminq_f32(mins, v);
            maxs = vmaxq_f32(maxs, v);
        }
        for (int l = 0; l < 4; l++) {
            minValue = std::min(minValue, mins[l]);
            maxValue = std::max(maxValue, maxs[l]);
        }
#endif
#ifdef __AVX2__
        __m256 mins = _mm256_set1_ps(1e100);
        __m256 maxs = _mm256_set1_ps(-1e100);
        for (; j + 7 < len; j += 8) {
            __m256 v = _mm256_loadu_ps(a + j);
            mins = _mm256_min_ps(mins, v);
            maxs = _mm256_max_ps(maxs, v);
        }
        // 将 AVX2 寄存器中的最小值、最大值提取到标量
        float temp_min[8], temp_max[8];
        _mm256_storeu_ps(temp_min, mins);
        _mm256_storeu_ps(temp_max, maxs);
        for (int l = 0; l < 8; l++) {
            minValue = std::min(minValue, temp_min[l]);
            maxValue = std::max(maxValue, temp_max[l]);
        }
#endif
        for (; j < len; j++) {
            minValue = std::min(minValue, a[j]);
            maxValue = std::max(maxValue, a[j]);
        }
    }

    void QuantizationAll(float *fValue, uint8_t *uValue, int len, LowBitConfig *config) {
        float scale = config->scale;
        float zeroPoint = config->zeroPoint;
        int j = 0;
#ifdef __aarch64__
        float32x4_t scales = vdupq_n_f32(scale);
        float32x4_t zeros = vdupq_n_f32(zeroPoint + 0.5);
        int32x4_t maxds = vcombine_s32(vcreate_s32(0x000000ff000000ff), vcreate_s32(0x000000ff000000ff));
        int32x4_t minds = vcombine_s32(vcreate_s32(0x0000000000000000), vcreate_s32(0x0000000000000000));
        for (; j + 7 < len; j += 8) {
            float32x4_t fin1 = vld1q_f32(fValue + j);
            float32x4_t fin2 = vld1q_f32(fValue + j + 4);
            fin1 = vaddq_f32(vdivq_f32(fin1, scales), zeros);
            fin2 = vaddq_f32(vdivq_f32(fin2, scales), zeros);
            int32x4_t out1 = vcvtq_s32_f32(fin1);
            int32x4_t out2 = vcvtq_s32_f32(fin2);
            out1 = vmaxq_s32(out1, minds);
            out1 = vminq_s32(out1, maxds);
            out2 = vmaxq_s32(out2, minds);
            out2 = vminq_s32(out2, maxds);
            uint16x8_t out3 = vpaddq_u16(vreinterpretq_u16_s32(out1), vreinterpretq_u16_s32(out2));
            uint8x8_t out = vmovn_u16(out3);
            vst1_u8(uValue + j, out);
        }
#endif
#ifdef __AVX2__
        __m256 vScale = _mm256_set1_ps(scale);
        __m256 vZeroPoint = _mm256_set1_ps(zeroPoint);
        __m256 vZero = _mm256_setzero_ps();
        __m256 vHalf = _mm256_set1_ps(0.5f);
        __m256 vMax = _mm256_set1_ps(255.0f);
        for (; j + 7 < len; j += 8) {
            // Load 8 floats
            __m256 vValue = _mm256_loadu_ps(&fValue[j]);
            
            // fValue[j] / scale + zeroPoint + 0.5
            __m256 vScaled = _mm256_div_ps(vValue, vScale);
            __m256 vWithZP = _mm256_add_ps(vScaled, vZeroPoint);
            __m256 vWithHalf = _mm256_add_ps(vWithZP, vHalf);
            
            // max(..., 0.0)
            __m256 vClampedLow = _mm256_max_ps(vWithHalf, vZero);
            
            // min(..., 255.0)
            __m256 vClampedHigh = _mm256_min_ps(vClampedLow, vMax);
            
            // QuantizationAll adds 0.5 explicitly, so conversion must truncate
            // to match the scalar LowBitConfig::quantization implementation.
            __m256i vInt32 = _mm256_cvttps_epi32(vClampedHigh);
            
            // Pack into 16-bit integers
            __m128i vInt16 = _mm_packus_epi32(
                _mm256_extractf128_si256(vInt32, 0),
                _mm256_extractf128_si256(vInt32, 1));
            
            // Pack into 8-bit integers
            __m128i vInt8 = _mm_packus_epi16(vInt16, vInt16);
            
            // Store the lower 64 bits (8 bytes)
            _mm_storel_epi64((__m128i*)&uValue[j], vInt8);
        }
#endif
        for (; j < len; j++) {
            uValue[j] = (uint8_t) (std::min(255., (double) std::max(fValue[j] / scale + zeroPoint + 0.5, 0.0)));
        }
    }

#ifdef __AVX2__
    void Avx2InputPermute(uint8_t* output, int n, int m, int groupCnt) {
        // The packed INT4 dot-product kernels consume a matching input layout:
        // AVX512-VNNI works on 64 values at a time, while AVX2 works on 32.
        // Keep quantization-group boundaries intact when choosing the layout.
        if (groupCnt <= 0 || groupCnt % 32 != 0 || m % 32 != 0) {
            return;
        }
        if (cpuInstructInfo.hasAVX512VNNI &&
            groupCnt % 64 == 0 && m % 64 == 0) {
            uint8_t *temp = new uint8_t[64];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j + 63 < m; j += 64) {
                    memcpy(temp, output + i * m + j, 64);
                    for (int k = 0; k < 32; k++) {
                        output[i * m + j + k] = temp[k * 2 + 1];
                        output[i * m + j + k + 32] = temp[k * 2];
                    }
                }
            }
            delete[] temp;
            return;
        } 
        
        /*uint8_t *temp = new uint8_t[32];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j + 31 < m; j += 32) {
                memcpy(temp, output + i * m + j, 32);
                for (int k = 0; k < 16; k++) {
                    output[i * m + j + k] = temp[k * 2 + 1];
                    output[i * m + j + k + 16] = temp[k * 2];
                }
            }
        }
        delete[] temp;
        return;*/

        const __m256i mask_even = _mm256_setr_epi8(
            0, 2, 4, 6, 8, 10, 12, 14, 
            16, 18, 20, 22, 24, 26, 28, 30,
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30
        );
        const __m256i mask_odd = _mm256_setr_epi8(
            1, 3, 5, 7, 9, 11, 13, 15,
            17, 19, 21, 23, 25, 27, 29, 31,
            1, 3, 5, 7, 9, 11, 13, 15,
            17, 19, 21, 23, 25, 27, 29, 31
        );
        for (int i = 0; i < n; i++) {
            for (int j = 0; j + 31 < m; j += 32) {
                // 加载32字节数据
                __m256i data = _mm256_loadu_si256((__m256i*)(output + i * m + j));
                __m256i evens = _mm256_shuffle_epi8(data, mask_even);
                __m256i odds = _mm256_shuffle_epi8(data, mask_odd);
                __m128i evenLow = _mm256_castsi256_si128(evens); // a[0]~a[15]
                __m128i evenHigh = _mm256_extracti128_si256(evens, 1); // a[16]~a[31]
                __m128i low = _mm_unpacklo_epi64(evenLow, evenHigh);

                __m128i oddLow = _mm256_castsi256_si128(odds); // a[0]~a[15]
                __m128i oddHigh = _mm256_extracti128_si256(odds, 1); // a[16]~a[31]
                __m128i high = _mm_unpacklo_epi64(oddLow, oddHigh);

                // 存储结果
                _mm256_storeu_si256((__m256i*)(output + i * m + j), _mm256_set_m128i(low, high));
            }
        }
    }
#endif

    void MultiThreadFloat32ToBFloat16Op::Run() {
        Float32ToBFloat16(input, output, len);
    }

    void MultiThreadFloat32ToQ8KOp::Run() {
        AssertInFastLLM((void*)input != (void*)output, "MultiThreadFloat32ToQ8KOp's input and output should be diff.\n");
        iqk_quantize_row_q8_K (
            input, output, len, 
            ggml_type_vec_dot_type((ggml_type)ggmlType), (ggml_type)ggmlType
        );
    }

    void MultiThreadOnlineQuantizationOp::Run() {
        int realGroup = (m - 1) / groupCnt + 1;
        for (int i = 0; i < n; i++) {
            float *cur = input + i * m;
            uint8_t *u = output + i * m;
            for (int g = 0; g < realGroup; g++) {
                int st = g * groupCnt;
                int end = std::min(m, (g + 1) * groupCnt);
                float minValue = 1e9, maxValue = -1e9;
                GetArrayMinMax(input + i * m + st, end - st, minValue, maxValue);
                configs[i * group + g] = (LowBitConfig(minValue, maxValue, 8, 0));
                QuantizationAll(cur + st, u + st, end - st, &configs[i * group + g]);
            }
        }

        if (permuteType == 0) {
            // for INT8 * INT8
#ifdef __AVX2__
            for (int i = 0; i < n * m; i++) {
                output[i] = (output[i] + !output[i]);
            }
#endif
        }

        if (permuteType == 1) {
            // for INT8 * INT4
#ifdef __AVX2__
            Avx2InputPermute(output, n, m, groupCnt);
#endif
        }

        if (inputSums != nullptr) {
            for (int i = 0; i < n; i++) {
                for (int g = 0; g < realGroup; g++) {
                    iscales[i * group + g] = configs[i * group + g].scale;
                    izeros[i * group + g] = configs[i * group + g].zeroPoint;
                    int sum = 0;
                    int j = g * groupCnt;
#ifdef __AVX2__
                    const __m256i ones8 = _mm256_set1_epi8(1);
                    const __m256i ones16 = _mm256_set1_epi16(1);
                    __m256i acc = _mm256_setzero_si256();
                    for (; j + 31 < (g + 1) * groupCnt && j + 31 < m; j += 32) {
                        __m256i data = _mm256_loadu_si256((__m256i*)(output + i * m + j));
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(data, ones8), ones16));
                    }
                    sum += I32sum(acc);
#endif
                    for (; j < (g + 1) * groupCnt && j < m; j++) {
                        sum += output[i * m + j];
                    }
                    inputSums[i * group + g] = sum;
                }
            }
        }

        if (permuteType == 0) {
            // for INT8 * INT8
#ifdef __AVX2__
            for (int i = 0; i < n * m; i++) {
                output[i] ^= 128;
            }
#endif
        }
    }

    bool CpuLinearOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        return true;
    }

    void DoCpuLinear(Data &input, Data &weight, const Data &bias, Data &output) {
//auto st = std::chrono::system_clock::now();
        output.Allocate();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();
        int threadSt = GetAlivePool()->curActivateThreadInterval.first;
        int threadLen = GetAlivePool()->curActivateThreadInterval.second - GetAlivePool()->curActivateThreadInterval.first;
        auto linearTypeError = [&]() {
            fprintf(stderr, "Linear error: unsupported data types input=%s, weight=%s, output=%s.\n",
                    GetDataTypeName(input.dataType).c_str(),
                    GetDataTypeName(weight.dataType).c_str(),
                    GetDataTypeName(output.dataType).c_str());
            fflush(stderr);
            ErrorInFastLLM("Linear error: unsupported data types input=" + GetDataTypeName(input.dataType) +
                           ", weight=" + GetDataTypeName(weight.dataType) +
                           ", output=" + GetDataTypeName(output.dataType) + ".\n");
        };

        if (input.dataType == DataType::FLOAT32 && output.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32) {
                RunLinearFloat32Float32((float*)input.cpuData, (float*)weight.cpuData, (float*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::BFLOAT16) {
                RunLinearFloat32BFloat16((float*)input.cpuData, (uint16_t*)weight.cpuData, (float*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::FLOAT16) {
                RunLinearFloat32Float16((float*)input.cpuData, (uint16_t*)weight.cpuData, (float*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT8) {
                RunLinearFloat32Int8((float*)input.cpuData, weight, (float*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT4_GROUP || weight.dataType == DataType::INT4_NOZERO) {
                int group = weight.group, groupCnt = weight.groupCnt;
                if (weight.dataType == DataType::INT4_NOZERO) {
                    group = 1, groupCnt = m;
                }
                RunLinearFloat32Int4Group((float*)input.cpuData, weight, (float*)output.cpuData, 
                                        bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, group, groupCnt,
                                        GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT4_GROUP32) {
                RunLinearFloat32Int4Group32(
                    (float*)input.cpuData, weight, (float*)output.cpuData,
                    bias.dims.size() > 0 ? (float*)bias.cpuData : nullptr,
                    n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT2_GROUP) {
                int group = weight.group, groupCnt = weight.groupCnt;
                RunLinearFloat32Int2Group((float*)input.cpuData, weight, (float*)output.cpuData, 
                                        bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, group, groupCnt,
                                        GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::BASE3_GROUP) {
                std::vector <uint8_t> base = {1, 3, 9, 27, 81};
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                
                auto pool = GetAlivePool();
                int threadNum = pool->threads.size();
                int per = k / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadBase3GroupLinearOp> ops;
                ops.reserve(threadNum);
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    if (i == threadNum - 1) {
                        end = k;
                    }
                    ops.emplace_back(inputData, weightData, biasData, outputData,
                                     n, m, k, cur, end, weight.group, weight.groupCnt, weight.halfScales.data());
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(i, &ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(i);
                }
            } else if (weight.dataType == DataType::INT4) {
                // 目前已经不用这种数据类型了
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                weight.CalcWeightSum();

                std::vector<LowBitConfig> inputConfigs;
                std::vector<uint8_t> uinput;
                std::vector <float> inputSums, iscales, izeros;
                OnlineQuantization(inputData, uinput, inputConfigs, n, m, 1, m, inputSums, iscales, izeros, 1);
                MultiplyInt4MultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k,
                                            weight.weightSum.data(), weight.zeros.data(), weight.scales.data(),
                                            biasData,
                                            inputConfigs, GetThreads());
            } else if (weight.dataType == DataType::FP8_E4M3) {
                RunLinearFloat32FP8E4M3((float*)input.cpuData, weight, (float*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::NVFP4) {
                RunLinearFloat32NVFP4((float*)input.cpuData, weight, (float*)output.cpuData,
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::DATA_GGUF_FORMAT) {
                RunLinearFloat32GGUF((float*)input.cpuData, (uint8_t*)weight.cpuData, (float*)output.cpuData, bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, 
                    &weight, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else {
                linearTypeError();
            }
        } else if (input.dataType == DataType::BFLOAT16 && output.dataType == DataType::FLOAT32) {
            if (CanRunBFloat16NativeLinearWeight(weight)) {
                RunLinearBFloat16NativeToFloat32((uint16_t*)input.cpuData, weight, (float*)output.cpuData,
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::BFLOAT16) {
                RunLinearBFloat16BFloat16((uint16_t*)input.cpuData, (uint16_t*)weight.cpuData, (float*)output.cpuData,
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::FLOAT16) {
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                if ((cpuInstructInfo.hasAVX512F &&
                     LinearBFloat16Float16Decode_AVX512F_Kernel(
                        (uint16_t*)input.cpuData, (uint16_t*)weight.cpuData, biasData, (float*)output.cpuData,
                        n, m, k, 0, k)) ||
                    (cpuInstructInfo.hasAVX2 &&
                     LinearBFloat16Float16_AVX2_Kernel(
                        (uint16_t*)input.cpuData, (uint16_t*)weight.cpuData, biasData, (float*)output.cpuData,
                        n, m, k, 0, k))) {
                } else {
                    std::vector<float> floatInput;
                    floatInput.resize(n * m);
                    BFloat16ToFloat32((uint16_t*)input.cpuData, floatInput.data(), n * m);
                    RunLinearFloat32Float16(floatInput.data(), (uint16_t*)weight.cpuData, (float*)output.cpuData,
                        biasData, n, m, k, GetAlivePool(), threadSt, threadLen);
                }
            } else if (weight.dataType == DataType::FP8_E4M3) {
                RunLinearBFloat16FP8E4M3((uint16_t*)input.cpuData, weight, (float*)output.cpuData,
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::NVFP4) {
                RunLinearBFloat16NVFP4((uint16_t*)input.cpuData, weight, (float*)output.cpuData,
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else {
                linearTypeError();
            }
        } else if (input.dataType == DataType::BFLOAT16 && output.dataType == DataType::BFLOAT16) {
            std::vector<float> floatOutput;
            floatOutput.resize(n * k);
            if (CanRunBFloat16NativeLinearWeight(weight)) {
                RunLinearBFloat16NativeToFloat32((uint16_t*)input.cpuData, weight, floatOutput.data(),
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::BFLOAT16) {
                RunLinearBFloat16BFloat16((uint16_t*)input.cpuData, (uint16_t*)weight.cpuData, floatOutput.data(),
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::FLOAT16) {
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                if ((cpuInstructInfo.hasAVX512F &&
                     LinearBFloat16Float16Decode_AVX512F_Kernel(
                        (uint16_t*)input.cpuData, (uint16_t*)weight.cpuData, biasData, floatOutput.data(),
                        n, m, k, 0, k)) ||
                    (cpuInstructInfo.hasAVX2 &&
                     LinearBFloat16Float16_AVX2_Kernel(
                        (uint16_t*)input.cpuData, (uint16_t*)weight.cpuData, biasData, floatOutput.data(),
                        n, m, k, 0, k))) {
                } else {
                    std::vector<float> floatInput;
                    floatInput.resize(n * m);
                    BFloat16ToFloat32((uint16_t*)input.cpuData, floatInput.data(), n * m);
                    RunLinearFloat32Float16(floatInput.data(), (uint16_t*)weight.cpuData, floatOutput.data(),
                        biasData, n, m, k, GetAlivePool(), threadSt, threadLen);
                }
            } else if (weight.dataType == DataType::FP8_E4M3) {
                RunLinearBFloat16FP8E4M3((uint16_t*)input.cpuData, weight, floatOutput.data(),
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::NVFP4) {
                RunLinearBFloat16NVFP4((uint16_t*)input.cpuData, weight, floatOutput.data(),
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else {
                linearTypeError();
            }
            Float32ToBFloat16(floatOutput.data(), (uint16_t*)output.cpuData, n * k);
        } else if (input.dataType == DataType::FLOAT16 && output.dataType == DataType::FLOAT16) {
            if (weight.dataType == DataType::FLOAT32) {
                RunLinearFloat16Float32((uint16_t*)input.cpuData, (float*)weight.cpuData, (uint16_t*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::FLOAT16) {
                RunLinearFloat16Float16((uint16_t*)input.cpuData, (uint16_t*)weight.cpuData, (uint16_t*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT8) {
                RunLinearFloat16Int8((uint16_t*)input.cpuData, weight, (uint16_t*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT4_GROUP || weight.dataType == DataType::INT4_NOZERO) {
                int group = weight.group, groupCnt = weight.groupCnt;
                if (weight.dataType == DataType::INT4_NOZERO) {
                    group = 1, groupCnt = m;
                }
                RunLinearFloat16Int4Group((uint16_t*)input.cpuData, weight, (uint16_t*)output.cpuData, 
                                        bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, group, groupCnt,
                                        GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::FP8_E4M3) {
                RunLinearFloat16FP8E4M3((uint16_t*)input.cpuData, weight, (uint16_t*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::NVFP4) {
                RunLinearFloat16NVFP4((uint16_t*)input.cpuData, weight, (uint16_t*)output.cpuData,
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::DATA_GGUF_FORMAT) {
                RunLinearFloat16GGUF((uint16_t*)input.cpuData, (uint8_t*)weight.cpuData, (uint16_t*)output.cpuData, bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, 
                    &weight, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else {
                linearTypeError();
            }
        } else {
            linearTypeError();
        }
//float spend = GetSpan(st, std::chrono::system_clock::now());
//float gops = (float)n * m * k / spend / 1e9;
//printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }

    void CpuLinearOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);
        AssertInFastLLM(bias.dataType == DataType::FLOAT32, "Linear's bias' type should be float32.\n");
        DoCpuLinear(input, weight, bias, output);
    }

    void CpuSplitOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));
        std::vector <int> dims = input.dims;
        dims[axis] = end - start;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    struct MultiThreadSliceOp : MultiThreadBaseOp {
        uint8_t *input, *output;
        int outer, inputStride, outputStride, copyLen;

        MultiThreadSliceOp (uint8_t *output, uint8_t *input, int outer, int outputStride, int inputStride, int copyLen) : 
            output(output), input(input), outer(outer), inputStride(inputStride), outputStride(outputStride), copyLen(copyLen) {}

        void Run() {
            for (int o = 0; o < outer; o++) {
                memcpy(output + o * outputStride, input + o * inputStride, copyLen);
            }
        }
    };

    static void RunMultiThreadSlice(uint8_t *output, uint8_t *input, int outer, int inputStride, int outputStride, int copyLen, AliveThreadPool *pool) {
        if (outer == 1) {
            (MultiThreadSliceOp(output, input, outer, outputStride, inputStride, copyLen)).Run();
            return;
        }
        int threadNum = pool->threads.size();
        int per = outer / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadSliceOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
            ops.push_back(new MultiThreadSliceOp(output + cur * outputStride, input + cur * inputStride, end - cur, outputStride, inputStride, copyLen));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuSplitOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));

        int outer = input.Count(0) / input.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];
        int unitSize = input.unitSize;

        int copyLen = (end - start) * inner * unitSize;
        if (outer == 1) {
            // 整块连续拷贝，按字节分段并行，避免单线程拷几百 MB
            RunMultiThreadMemcpy(output.cpuData, input.cpuData + start * inner * unitSize,
                                 copyLen, GetAlivePool());
        } else {
            RunMultiThreadSlice(output.cpuData, input.cpuData + start * inner * unitSize, outer,
                inputStride * unitSize, outputStride * unitSize, copyLen, GetAlivePool());
        }
    }

    void CpuRepeatOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int repeatTimes = intParams.find("repeatTimes") != intParams.end() ? intParams.find("repeatTimes")->second : 1;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        std::vector <int> dims = input.dims;
        dims[axis] *= repeatTimes;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuRepeatOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int repeatTimes = intParams.find("repeatTimes") != intParams.end() ? intParams.find("repeatTimes")->second : 1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        output.Allocate();

        int outer = output.Count(0) / output.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];
        int unitSize = input.unitSize;

        if ((long long)outer * outputStride * unitSize >= 262144) {
            std::vector <MultiThreadMemcpyMultiLinesTask> tasks;
            tasks.reserve((size_t)outer * repeatTimes);
            for (int o = 0; o < outer; o++) {
                for (int t = 0; t < repeatTimes; t++) {
                    tasks.push_back(MultiThreadMemcpyMultiLinesTask(
                        output.cpuData + (long long)o * outputStride * unitSize + (long long)t * channels * inner * unitSize,
                        input.cpuData + (long long)o * inputStride * unitSize,
                        channels * inner * unitSize));
                }
            }
            RunMultiThreadMemcpyMultiLines(tasks, GetAlivePool());
        } else {
            for (int o = 0; o < outer; o++) {
                for (int t = 0; t < repeatTimes; t++) {
                    memcpy(output.cpuData + o * outputStride * unitSize + t * channels * inner * unitSize,
                        input.cpuData + (o * inputStride) * unitSize,
                        channels * inner * unitSize);
                }
            }
        }
    }

    struct CpuCopyRangeOp : MultiThreadBaseOp {
        uint8_t *output;
        uint8_t *input;
        uint64_t st, end;

        CpuCopyRangeOp(uint8_t *output, uint8_t *input, uint64_t st, uint64_t end) :
            output(output), input(input), st(st), end(end) {}

        void Run() {
            memcpy(output + st, input + st, (size_t)(end - st));
        }
    };

    void CpuCopyOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        if (&input == &output) {
            return;
        }
        output.Allocate();
        uint64_t bytes = input.GetBytes();
        if (bytes == 0) {
            return;
        }

        uint8_t *inputData = input.cpuData;
        uint8_t *outputData = output.cpuData;
        auto *pool = GetAlivePool();
        int threadNum = pool == nullptr ? 1 : (int)pool->threads.size();
        threadNum = std::min(threadNum, 8);
        if (threadNum <= 1 || bytes < 256 * 1024) {
            memcpy(outputData, inputData, (size_t)bytes);
            return;
        }

        std::vector<CpuCopyRangeOp*> ops;
        uint64_t per = (bytes + threadNum - 1) / threadNum;
        for (int i = 0; i < threadNum; i++) {
            uint64_t st = (uint64_t)i * per;
            uint64_t end = std::min(bytes, st + per);
            if (st >= end) {
                break;
            }
            ops.push_back(new CpuCopyRangeOp(outputData, inputData, st, end));
        }
        for (int i = 0; i < (int)ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < (int)ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuCatOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        if (input0.dims.size() == 0 && input1.dims.size() > 0) {
            output.Resize(input1.dims);
            return;
        }
        if (input1.dims.size() == 0 && input0.dims.size() > 0) {
            output.Resize(input0.dims);
            return;
        }

        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
                        (input0.dataType == DataType::BFLOAT16 && input1.dataType == DataType::BFLOAT16),
                        "Cat's input's type should be float32, float16 or bfloat16.\n");
        AssertInFastLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.");

        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        for (int i = 0; i < dimsLen; i++) {
            if (i != axis) {                
                AssertInFastLLM(input0.dims[i] == input1.dims[i], "Cat Error: input's shape doesn't match.");
            }
        }

        std::vector <int> dims = input0.dims;
        dims[axis] += input1.dims[axis];

        output.dataType = input0.dataType;
        output.Resize(dims);
    }

    void CpuCatOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        auto catSt = std::chrono::system_clock::now();
        output.Allocate();
        auto catAlloc = std::chrono::system_clock::now();

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        if (input0.dims.size() == 0 && input1.dims.size() > 0) {
            output.CopyFrom(input1);
            return;
        }
        if (input1.dims.size() == 0 && input0.dims.size() > 0) {
            output.CopyFrom(input0);
            return;
        }

        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        int outer = output.Count(0) / output.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);
        int outputStride = output.Count(axis);
        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        if (outer > 1 && (long long)outer * outputStride * unitSize >= 262144) {
            std::vector <MultiThreadMemcpyMultiLinesTask> tasks;
            tasks.reserve((size_t)outer * 2);
            int len0 = input0.dims[axis] * inner * unitSize;
            int len1 = input1.dims[axis] * inner * unitSize;
            for (int o = 0; o < outer; o++) {
                tasks.push_back(MultiThreadMemcpyMultiLinesTask(
                    output.cpuData + (long long)o * outputStride * unitSize,
                    input0.cpuData + (long long)o * input0Stride * unitSize, len0));
                tasks.push_back(MultiThreadMemcpyMultiLinesTask(
                    output.cpuData + (long long)o * outputStride * unitSize + len0,
                    input1.cpuData + (long long)o * input1Stride * unitSize, len1));
            }
            RunMultiThreadMemcpyMultiLines(tasks, GetAlivePool());
        } else {
            for (int o = 0; o < outer; o++) {
                memcpy(output.cpuData + o * outputStride * unitSize,
                       input0.cpuData + (o * input0Stride) * unitSize,
                       input0.dims[axis] * inner * unitSize);
                memcpy(output.cpuData + o * outputStride * unitSize + input0.dims[axis] * inner * unitSize,
                       input1.cpuData + (o * input1Stride) * unitSize,
                       input1.dims[axis] * inner * unitSize);
            }
        }

        static const bool catProfSlow = std::getenv("FASTLLM_PROFILE_SLOW_OPS") != nullptr;
        if (catProfSlow) {
            float allocSpend = GetSpan(catSt, catAlloc);
            float copySpend = GetSpan(catAlloc, std::chrono::system_clock::now());
            if (allocSpend + copySpend > 0.02f) {
                printf("[fastllm-cat-detail] alloc=%.6f copy=%.6f total=%.6f s (outer=%d)\n",
                       allocSpend, copySpend, allocSpend + copySpend, outer);
                fflush(stdout);
            }
        }
    }

    void CpuPadOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        int padSize = intParams.find("padSize") != intParams.end() ? intParams.find("padSize")->second : 0;
        AssertInFastLLM(padSize >= 0, "Pad Error: padSize should be non-negative.");

        output.dataType = input.dataType;
        if (input.dims.empty()) {
            output.Resize(input.dims);
            return;
        }

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        std::vector<int> dims = input.dims;
        dims[axis] += padSize;
        output.Resize(dims);
    }

    void CpuPadOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        int padSize = intParams.find("padSize") != intParams.end() ? intParams.find("padSize")->second : 0;
        AssertInFastLLM(padSize >= 0, "Pad Error: padSize should be non-negative.");
        if (input.dims.empty() || padSize == 0) {
            output.CopyFrom(input);
            return;
        }

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        output.Allocate();
        memset(output.cpuData, 0, output.GetBytes());

        int outer = output.Count(0) / output.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int inner = input.strides[axis];
        int unitSize = input.unitSize;

        for (int o = 0; o < outer; o++) {
            memcpy(output.cpuData + o * outputStride * unitSize,
                   input.cpuData + o * inputStride * unitSize,
                   input.dims[axis] * inner * unitSize);
        }
    }

    void DoCpuCatDirect(Data &input0, Data &input1, int axis) {
        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
                        (input0.dataType == DataType::BFLOAT16 && input1.dataType == DataType::BFLOAT16),
                        "CatDirect's input's type should be float32, float16 or bfloat16.\n");
        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "CatDirect error: inputs should use same device.\n");

        if (input0.dims.size() == 0) {
            input0.Resize(input1.dims);
            AssertInFastLLM(input0.expansionDims.size() == input1.dims.size() &&
                            input1.dims[axis] <= input0.expansionDims[axis],
                            "CatDirect Error: input0's expansion size is not enough.\n");
            int outer = input1.Count(0) / input1.Count(axis);
            int input0Stride = input0.Count(axis);
            int input1Stride = input1.Count(axis);
            int inner = input0.strides[axis];
            int unitSize = input0.unitSize;
            for (int o = 0; o < outer; o++) {
                memcpy(input0.cpuData + o * input0Stride * unitSize,
                       input1.cpuData + o * input1Stride * unitSize,
                       input1.dims[axis] * inner * unitSize);
            }

            return;
        }

        std::vector <int> dims = input0.dims;
        std::vector <int> oldDims = dims;
        dims[axis] += input1.dims[axis];
        input0.Resize(dims);
        int outer = input0.Count(0) / input0.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);

        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        for (int o = 0; o < outer; o++) {
            memcpy(input0.cpuData + o * input0Stride * unitSize + oldDims[axis] * inner * unitSize,
                   input1.cpuData + (o * input1Stride) * unitSize,
                   input1.dims[axis] * inner * unitSize);
        }
    }

    void CpuCatDirectOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        DoCpuCatDirect(input0, input1, axis);
    }

    struct MultiThreadMatMulSingleOp : MultiThreadBaseOp {
        float *input0Base, *input1Base, *outputBase;
        int input0Spatial, input1Spatial, outputSpatial;
        int input0Stride, input1Stride, n, m, k;
        float alpha;
        int st, end;

        MultiThreadMatMulSingleOp(float *input0Base, float *input1Base, float *outputBase,
                      int input0Spatial, int input1Spatial, int outputSpatial,
                      int input0Stride, int input1Stride,
                      int n, int m, int k, float alpha, int st, int end) :
                      input0Base(input0Base), input1Base(input1Base), outputBase(outputBase),
                      input0Spatial(input0Spatial), input1Spatial(input1Spatial), outputSpatial(outputSpatial),
                      input0Stride(input0Stride), input1Stride(input1Stride), 
                      n(n), m(m), k(k), alpha(alpha), st(st), end(end) {}
        
        void Run() {
            for (int b = st; b < end; b++) {
                float *input0Data = input0Base + b * input0Spatial;
                float *input1Data = input1Base + b * input1Spatial;
                float *outputData = outputBase + b * outputSpatial;
                std::fill(outputData, outputData + n * k, 0.0f);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        float now = input0Data[i * input0Stride + j] * alpha;
                        for (int l = 0; l < k; l++) {
                            outputData[i * k + l] += (now * input1Data[j * k + l]);
                        }
                    }
                }
            }
        }
    };

    struct MultiThreadMatMulFloat16SingleOp : MultiThreadBaseOp {
        uint16_t *input0Base, *input1Base, *outputBase;
        int input0Spatial, input1Spatial, outputSpatial;
        int input0Stride, input1Stride, n, m, k;
        float alpha;
        int st, end;

        MultiThreadMatMulFloat16SingleOp(uint16_t *input0Base, uint16_t *input1Base, uint16_t *outputBase,
                             int input0Spatial, int input1Spatial, int outputSpatial,
                             int input0Stride, int input1Stride,
                             int n, int m, int k, float alpha, int st, int end) :
                      input0Base(input0Base), input1Base(input1Base), outputBase(outputBase),
                      input0Spatial(input0Spatial), input1Spatial(input1Spatial), outputSpatial(outputSpatial),
                      input0Stride(input0Stride), input1Stride(input1Stride), 
                      n(n), m(m), k(k), alpha(alpha), st(st), end(end) {}

        void Run() {
            float *input0 = new float[n * m];
            float *input1 = new float[m * k];
            float *output = new float[n * k];

            for (int b = st; b < end; b++) {
                uint16_t *input0Data = input0Base + b * input0Spatial;
                uint16_t *input1Data = input1Base + b * input1Spatial;
                uint16_t *outputData = outputBase + b * outputSpatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        input0[i * m + j] = fp16tofp32.dict[input0Data[i * input0Stride + j]];
                    }
                }
                for (int j = 0; j < m; j++) {
                    for (int l = 0; l < k; l++) {
                        input1[j * k + l] = fp16tofp32.dict[input1Data[j * k + l]];
                    }
                }
                std::fill(output, output + n * k, 0.0f);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        float now = input0[i * m + j] * alpha;
                        for (int l = 0; l < k; l++) {
                            output[i * k + l] += (now * input1[j * k + l]);
                        }
                    }
                }
                for (int i = 0; i < n * k; i++) {
                    outputData[i] = float_to_half(output[i]);
                }
            }

            delete[] input0;
            delete[] input1;
            delete[] output;
        }
    };

    struct MultiThreadMatMulTransBSingleOp : MultiThreadBaseOp {
        float *input0Base, *input1Base, *outputBase;
        int input0Spatial, input1Spatial, outputSpatial;
        int input0Stride, input1Stride, n, m, k;
        float alpha;
        int st, end;

        MultiThreadMatMulTransBSingleOp(float *input0Base, float *input1Base, float *outputBase,
                      int input0Spatial, int input1Spatial, int outputSpatial,
                      int input0Stride, int input1Stride,
                      int n, int m, int k, float alpha, int st, int end) :
                      input0Base(input0Base), input1Base(input1Base), outputBase(outputBase),
                      input0Spatial(input0Spatial), input1Spatial(input1Spatial), outputSpatial(outputSpatial),
                      input0Stride(input0Stride), input1Stride(input1Stride), 
                      n(n), m(m), k(k), alpha(alpha), st(st), end(end) {}
        
        void Run() {
            for (int b = st; b < end; b++) {
                float *input0Data = input0Base + b * input0Spatial;
                float *input1Data = input1Base + b * input1Spatial;
                float *outputData = outputBase + b * outputSpatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < k; j++) {
                        float now = 0.0f;
                        int l = 0;
#ifdef __aarch64__
                        float32x4_t sum = {0, 0, 0, 0};
                        for (; l + 3 < m; l += 4) {
                            sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(input0Data + i * input0Stride + l),
                                                        vld1q_f32(input1Data + j * input1Stride + l)));
                        }
                        now += sum[0] + sum[1] + sum[2] + sum[3];
#elif defined(__AVX__)
                        __m256 vsum = _mm256_set1_ps(0.0f);
                        for (; l + 7 < m; l += 8) {
                            __m256 vx = _mm256_loadu_ps((const float *) (input0Data + i * input0Stride + l));
                            __m256 vy = _mm256_loadu_ps((const float *) (input1Data + j * input1Stride + l));
                            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
                        }
                        now += Floatsum(vsum);
#endif
                        for (; l < m; l++) {
                            now += input0Data[i * input0Stride + l] * input1Data[j * input1Stride + l];
                        }
                        outputData[i * k + j] = now * alpha;
                    }
                }
            }
        }
    };

    struct MultiThreadMatMulTransBFloat16SingleOp : MultiThreadBaseOp {
        uint16_t *input0Base, *input1Base, *outputBase;
        int input0Spatial, input1Spatial, outputSpatial;
        int input0Stride, input1Stride, n, m, k;
        float alpha;
        int st, end;

        MultiThreadMatMulTransBFloat16SingleOp(uint16_t *input0Base, uint16_t *input1Base, uint16_t *outputBase,
                      int input0Spatial, int input1Spatial, int outputSpatial,
                      int input0Stride, int input1Stride,
                      int n, int m, int k, float alpha, int st, int end) :
                      input0Base(input0Base), input1Base(input1Base), outputBase(outputBase),
                      input0Spatial(input0Spatial), input1Spatial(input1Spatial), outputSpatial(outputSpatial),
                      input0Stride(input0Stride), input1Stride(input1Stride), 
                      n(n), m(m), k(k), alpha(alpha), st(st), end(end) {}
        void Run() {
            for (int b = st; b < end; b++) {
                uint16_t *input0Data = input0Base + b * input0Spatial;
                uint16_t *input1Data = input1Base + b * input1Spatial;
                uint16_t *outputData = outputBase + b * outputSpatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < k; j++) {
                        float now = 0.0f;
                        int l = 0;
#if defined(__AVX__)
                        __m256 vsum = _mm256_set1_ps(0.0f);
                        for (; l + 7 < m; l += 8) {
                            __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (input0Data + i * input0Stride + l)));
                            __m256 vy = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (input1Data + j * input1Stride + l)));
                            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
                        }
                        now += Floatsum(vsum);
#endif
                        for (; l < m; l++) {
                            now += fp16tofp32.dict[input0Data[i * input0Stride + l]] *
                                    fp16tofp32.dict[input1Data[j * input1Stride + l]];
                        }
                        outputData[i * k + j] = float_to_half(now * alpha);
                    }
                }
            }
        }
    };

    void CpuMatMulOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "MatMul error: inputs should use same device.\n");
        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
                        (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16),
                        "MatMul's input's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims.size() >= 2 && input1.dims.size() >= 2,
                        "MatMul's input's shape's size should be >= 2.\n");
        AssertInFastLLM(input0.dims.back() == input1.dims[input1.dims.size() - 2],
                        "MatMul's shape error.\n");
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        AssertInFastLLM(batch0 == batch1 * group, "MatMul: input0.dims[1] should be equal to input1.dims[0] * group.\n");
        // AssertInFastLLM(batch0 == batch1, "MatMul's shape error.\n");

        std::vector <int> dims = input0.dims;
        dims.back() = input1.dims[input1.dims.size() - 1];

        output.dataType = input0.dataType;
        output.Resize(dims);
    }

    void CpuMatMulOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        int input0Spatial = input0.Count(input0.dims.size() - 2) * group;
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2] * group;
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 1];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2) * group;
        int threadNum = GetThreads();
#ifdef _WIN64
        threadNum = 1;
#endif
        if (batch0 * n * m * k < 64 * 4096) {
            threadNum = 1;
        }
        threadNum = std::min(threadNum, 4);
        // TODO: 汇编优化
        int per = batch0 / threadNum;
        int cur = 0;
        if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) {
            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulSingleOp*> ops;
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulSingleOp(
                    (float *) input0.cpuData, (float *) input1.cpuData, (float *) output.cpuData,
                    input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                    n, m, k, alpha, o, o + 1
                ));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        } else if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16) {
            std::vector <uint16_t> fp16InputData;
            std::vector <uint16_t> fp16OutputData;
            fp16InputData.resize(input0.Count(0));
            fp16OutputData.resize(output.Count(0));
            Float32ToFloat16((float*)input0.cpuData, fp16InputData.data(), input0.Count(0));

            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulFloat16SingleOp*> ops;
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulFloat16SingleOp(
                    (uint16_t *) fp16InputData.data(), (uint16_t *) input1.cpuData, (uint16_t *) fp16OutputData.data(),
                    input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                    n, m, k, alpha, o, o + 1
                ));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
            Float16ToFloat32(fp16OutputData.data(), (float *) output.cpuData, output.Count(0));
        } else if (input0.dataType == DataType::FLOAT16) {
            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulFloat16SingleOp*> ops;
            if (batch0 == 1) {
                int partn = std::max(1, n / threads);
                for (int o = 0; o < n; o += partn) {
                    int len = std::min(partn, n - o);
                    ops.push_back(new MultiThreadMatMulFloat16SingleOp(
                        ((uint16_t *) input0.cpuData) + o * m, 
                        (uint16_t *) input1.cpuData, 
                        ((uint16_t *) output.cpuData) + o * k,
                        input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                        len, m, k, alpha, 0, 1
                    ));
                }
            } else {
                for (int o = 0; o < batch0; o++) {
                    ops.push_back(new MultiThreadMatMulFloat16SingleOp(
                        (uint16_t *) input0.cpuData, (uint16_t *) input1.cpuData, (uint16_t *) output.cpuData,
                        input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                        n, m, k, alpha, o, o + 1
                    ));
                }
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        }
    }

    void CpuMatMulTransBOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "MatMulTransB error: inputs should use same device.\n");
        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
                        (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16),
                        "MatMulTransB's input's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims.size() >= 2 && input1.dims.size() >= 2,
                        "MatMulTransB's input's shape's size should be >= 2.\n");
        AssertInFastLLM(input0.dims.back() == input1.dims.back(),
                        "MatMulTransB's shape error.\n");
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        AssertInFastLLM(batch0 == batch1 * group, "MatMulTransB: input0.dims[0] should be equal to input1.dims[0] * group.\n");
        // AssertInFastLLM(batch0 == batch1, "MatMulTransB's shape error.\n");

        std::vector <int> dims = input0.dims;
        dims.back() = input1.dims[input1.dims.size() - 2];
        output.dataType = input0.dataType;
        output.Resize(dims);
    }

    void CpuMatMulTransBOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        int input0Spatial = input0.Count(input0.dims.size() - 2) * group;
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2] * group;
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 2];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2) * group;
        int threadNum = GetThreads();
#ifdef _WIN64
        threadNum = 1;
#endif
        if (batch0 * n * m * k < 64 * 4096) {
            threadNum = 1;
        }
        threadNum = std::min(threadNum, 4);
        int per = batch0 / threadNum;
        int cur = 0;
        if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) {
            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulTransBSingleOp*> ops;
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulTransBSingleOp(
                    (float *) input0.cpuData, (float *) input1.cpuData, (float *) output.cpuData,
                    input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                    n, m, k, alpha, o, o + 1
                ));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        } else if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16) {
            std::vector <uint16_t> fp16InputData, fp16OutputData;
            fp16InputData.resize(input0.Count(0));
            fp16OutputData.resize(output.Count(0));
            Float32ToFloat16((float*)input0.cpuData, fp16InputData.data(), input0.Count(0));

            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulTransBFloat16SingleOp*> ops;
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulTransBFloat16SingleOp(
                    (uint16_t *) fp16InputData.data(), (uint16_t *) input1.cpuData, (uint16_t *) fp16OutputData.data(),
                    input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                    n, m, k, alpha, o, o + 1
                ));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
            Float16ToFloat32(fp16OutputData.data(), (float *) output.cpuData, output.Count(0));
        } else {
            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulTransBFloat16SingleOp*> ops;
            if (batch0 == 1) {
                int partn = std::max(1, n / threads);
                for (int o = 0; o < n; o += partn) {
                    int len = std::min(partn, n - o);
                    ops.push_back(new MultiThreadMatMulTransBFloat16SingleOp(
                        ((uint16_t *) input0.cpuData) + o * m, 
                        (uint16_t *) input1.cpuData, 
                        ((uint16_t *) output.cpuData) + o * k,
                        input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                        len, m, k, alpha, 0, 1
                    ));
                }
            } else {
                for (int o = 0; o < batch0; o++) {
                    ops.push_back(new MultiThreadMatMulTransBFloat16SingleOp(
                        (uint16_t *) input0.cpuData, (uint16_t *) input1.cpuData, (uint16_t *) output.cpuData,
                        input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                        n, m, k, alpha, o, o + 1
                    ));
                }
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        }
    }

    void CpuNormalizeOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Normalize error: Data's type should be float32 or float16.\n");

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData = new float[len];
            outputData = new float[len];
            for (int i = 0; i < len; i++) {
                inputData[i] = fp16tofp32.dict[((uint16_t *) input.cpuData)[i]];
            }
        }
        if (inner == 1) {
            for (int i = 0; i < outer; i++) {
                float sum = 0;
                for (int j = 0; j < channels; j++) {
                    sum += inputData[j];
                }
                for (int j = 0; j < channels; j++) {
                    inputData[j] /= sum;
                }
                inputData += channels;
                outputData += channels;
            }
        } else {
            /*for (int i = 0; i < outer; i++) {
                std::vector<float> maxValue(inner, -FLT_MAX);
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        maxValue[k] = std::max(maxValue[k], inputData[j * inner + k]);
                    }
                }
                std::vector<float> sum(inner, 0.0);
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        outputData[j * inner + k] = std::exp(inputData[j * inner + k] - maxValue[k]);
                        sum[k] += outputData[j * inner + k];
                    }
                }

                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        outputData[j * inner + k] /= sum[k];
                    }
                }

                inputData += channels * inner;
                outputData += channels * inner;
            }*/
        }

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData -= len;
            outputData -= len;
            for (int i = 0; i < len; i++) {
                ((uint16_t *) output.cpuData)[i] = float_to_half(outputData[i]);
            }

            delete[] inputData;
            delete[] outputData;
        }
    }

    void CpuSoftMaxOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Softmax error: Data's type should be float32.\n");

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData = new float[len];
            outputData = new float[len];
            for (int i = 0; i < len; i++) {
                inputData[i] = fp16tofp32.dict[((uint16_t *) input.cpuData)[i]];
            }
        }

        if (inner == 1) {
            for (int i = 0; i < outer; i++) {
                float maxValue = 0;
                int j = 0;
#ifdef __aarch64__
                float32x4_t vmax = vdupq_n_f32(-1e9);
                for (; j + 3 < channels; j += 4) {
                    vmax = vmaxq_f32(vmax, vld1q_f32(inputData + j));
                }
                for (int k = 0; k < 4; k++) {
                    maxValue = std::max(maxValue, vmax[k]);
                }
#endif
                for (; j < channels; j++) {
                    maxValue = std::max(maxValue, inputData[j]);
                }

                j = 0;
#ifdef __aarch64__
                vmax = vdupq_n_f32(maxValue);
                for (; j + 3 < channels; j += 4) {
                    vst1q_f32(outputData + j, exp_ps(vsubq_f32(vld1q_f32(inputData + j), vmax)));
                }
#endif
                for (; j < channels; j++) {
                    outputData[j] = exp(inputData[j] - maxValue);
                }
                float sum = 0.0;
                j = 0;
                for (; j < channels; j++) {
                    sum += outputData[j];
                }
                if (fabs(sum) < 1e-9) {
                    sum = 0.1;
                }
                j = 0;
#ifdef __aarch64__
                float32x4_t fsum = vdupq_n_f32(sum);
                for (j = 0; j + 3 < channels; j += 4) {
                    vst1q_f32(outputData + j, vdivq_f32(vld1q_f32(outputData + j), fsum));
                }
#endif
                for (; j < channels; j++) {
                    outputData[j] = outputData[j] / sum;
                }
                inputData += channels;
                outputData += channels;
            }
        } else {
            for (int i = 0; i < outer; i++) {
                std::vector<float> maxValue(inner, -FLT_MAX);
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        maxValue[k] = std::max(maxValue[k], inputData[j * inner + k]);
                    }
                }
                std::vector<float> sum(inner, 0.0);
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        outputData[j * inner + k] = std::exp(inputData[j * inner + k] - maxValue[k]);
                        sum[k] += outputData[j * inner + k];
                    }
                }

                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        outputData[j * inner + k] /= sum[k];
                    }
                }

                inputData += channels * inner;
                outputData += channels * inner;
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData -= len;
            outputData -= len;
            for (int i = 0; i < len; i++) {
                ((uint16_t *) output.cpuData)[i] = float_to_half(outputData[i]);
            }

            delete[] inputData;
            delete[] outputData;
        }
    }

    struct FP16SiluManager {
        uint16_t dict[65536];

        FP16SiluManager() {
            for (int i = 0; i < 65536; i++) {
                float x = half_to_float(i);
                float y = x / (1.0 + expf(-x));
                dict[i] = float_to_half(y);
            }
        }
    } fp16SiluManager;

    struct FP16SigmoidManager {
        uint16_t dict[65536];

        FP16SigmoidManager() {
            for (int i = 0; i < 65536; i++) {
                float x = half_to_float(i);
                float y = 1.0 / (1.0 + expf(-x));
                dict[i] = float_to_half(y);
            }
        }
    } fp16SigmoidManager;

    void CpuSiluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Silu error: Data's type should be float32 or float16.\n");
        int len = input.Count(0);

        if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t*)input.cpuData;
            uint16_t *outputData = (uint16_t*)output.cpuData;
            for (int i = 0; i < len; i++) {
                outputData[i] = fp16SiluManager.dict[inputData[i]];
            }
        } else if (len < 65536) {
            float *inputData = (float*)input.cpuData;
            float *outputData = (float*)output.cpuData;
            int i = 0;
    #ifdef __aarch64__
            float32x4_t c1 = vdupq_n_f32(1.0f);
            for (; i + 3 < len; i += 4) {
                float32x4_t vx = vld1q_f32(inputData + i);
                float32x4_t vdiv = vaddq_f32(c1, exp_ps(vnegq_f32(vx)));
                vx = vdivq_f32(vx, vdiv);
                vst1q_f32(outputData + i, vx);
            }
    #endif
            for (; i < len; i++) {
                float x = inputData[i];
                outputData[i] = x / (1.0 + expf(-x));
            }
        } else {
            auto *pool = GetAlivePool();
            int threadNum = pool->threads.size();
            int per = len / threadNum;
            std::vector<fastllm::MultiThreadSiluOp*> ops;
            int cur = 0;
            for (int i = 0; i < threadNum; i++) {
                int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
                ops.push_back(new MultiThreadSiluOp((float*)input.cpuData + cur, end - cur,
                                                    (float*)output.cpuData + cur, 1, 0, 0));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }
    }

    void CpuTanHOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);                    
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "GeluNew error: Data's type should be float32.\n");

        float temp = sqrt(2.0f / M_PI), factor = 0.044715;
        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
        for (; i < len; i++) {
            outputData[i] = tanhf(inputData[i]);
        }
    }

    float erf(float a)
    {
        float r, s, t, u;

        t = fabsf(a);
        s = a * a;
        if (t > 0.927734375f)
        {   // 475/512
            // maximum error 0.99527 ulp
            r = fmaf(-1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
            u = fmaf(-3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
            r = fmaf(r, s, u);
            r = fmaf(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
            r = fmaf(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
            r = fmaf(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
            r = fmaf(r, t, -t);
            r = 1.0f - expf(r);
            r = copysignf(r, a);
        }
        else
        {
            // maximum error 0.98929 ulp
            r = -5.96761703e-4f;             // -0x1.38e000p-11
            r = fmaf(r, s, 4.99119423e-3f);  //  0x1.471a58p-8
            r = fmaf(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
            r = fmaf(r, s, 1.12819925e-1f);  //  0x1.ce1c44p-4
            r = fmaf(r, s, -3.76125336e-1f); // -0x1.812700p-2
            r = fmaf(r, s, 1.28379166e-1f);  //  0x1.06eba8p-3
            r = fmaf(r, a, a);
        }
        return r;
    }

    void CpuReluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Relu error: Data's type should be float32.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = x > 0 ? x : 0;
        }
    }

    void CpuSigmoidOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16 ||
                        input.dataType == DataType::BFLOAT16,
                        "Sigmoid error: Data's type should be float32, float16 or bfloat16.\n");

        int len = input.Count(0);
        if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t*)input.cpuData;
            uint16_t *outputData = (uint16_t*)output.cpuData;
            for (int i = 0; i < len; i++) {
                outputData[i] = fp16SigmoidManager.dict[inputData[i]];
            }
        } else if (input.dataType == DataType::BFLOAT16) {
            uint16_t *inputData = (uint16_t*)input.cpuData;
            uint16_t *outputData = (uint16_t*)output.cpuData;
            for (int i = 0; i < len; i++) {
                float value = BFloat16BitsToFloat32(inputData[i]);
                outputData[i] = Float32ToBFloat16RNEBits(
                    1.0f / (1.0f + std::exp(-value)));
            }
        } else {
            float *inputData = (float*)input.cpuData;
            float *outputData = (float*)output.cpuData;
            auto *pool = GetAlivePool();
            int threadNum = pool->threads.size();
            if (len < 65536) {
                MultiThreadSigmoidOp(inputData, len, outputData).Run();
            } else {
                int per = len / threadNum;
                std::vector<fastllm::MultiThreadSigmoidOp*> ops;
                int cur = 0;
                for (int i = 0; i < threadNum; i++) {
                    int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
                    ops.push_back(new MultiThreadSigmoidOp(inputData + cur, end - cur, outputData + cur));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            }
        }
    }

    void CpuExpOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, "Exp error: Data's type should be float32 or float16.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        std::vector <float> floatInputVector, floatOutputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            floatOutputVector.resize(output.Count(0));
            inputData = (float*)floatInputVector.data();
            outputData = (float*)floatOutputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        int len = input.Count(0);
        int i = 0;
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = exp(x);
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(outputData, (uint16_t*)output.cpuData, (int)floatOutputVector.size());
        }
    }

    void CpuGeluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16,
                        "Gelu error: Data's type should be float32 or float16.\n");

        std::vector<float> floatInputVector, floatOutputVector;
        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            floatOutputVector.resize(output.Count(0));
            inputData = floatInputVector.data();
            outputData = floatOutputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        int len = input.Count(0);
        int i = 0;
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = x * 0.5f * (1.0f + erf(x / sqrt(2.0)));
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(outputData, (uint16_t*)output.cpuData, (int)floatOutputVector.size());
        }
    }

    void CpuGeluNewOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "GeluNew error: Data's type should be float32.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
#ifdef __aarch64__
        float32x4_t c0 = vdupq_n_f32(0.044715f);
        float32x4_t c1 = vdupq_n_f32(1.0f);
        float32x4_t c2 = vdupq_n_f32(0.7978845608028654f);
        float32x4_t c3 = vdupq_n_f32(0.5f);

        for (; i + 3 < len; i += 4) {
            float32x4_t vx = vld1q_f32(inputData + i);
            float32x4_t v1 = vaddq_f32(c1, vmulq_f32(vmulq_f32(c0, vx), vx));
            float32x4_t v2 = vmulq_f32(vmulq_f32(c2, vx), v1);
            float32x4_t vex = exp_ps(v2);
            float32x4_t venegx = exp_ps(vnegq_f32(v2));
            float32x4_t vtan = vdivq_f32(vsubq_f32(vex, venegx), vaddq_f32(vex, venegx));
            float32x4_t vout = vmulq_f32(vmulq_f32(c3, vx), vaddq_f32(c1, vtan));
            vst1q_f32(outputData + i, vout);
        }
#endif
#ifdef __AVX2__
        auto var1 = _mm256_set1_ps(0.044715f);
        auto var2 = _mm256_set1_ps(0.7978845608028654f);
        auto var3 = _mm256_set1_ps(378.f);
        auto var4 = _mm256_set1_ps(17325.f);
        auto var5 = _mm256_set1_ps(135135.f);
        auto var6 = _mm256_set1_ps(28.f);
        auto var7 = _mm256_set1_ps(3150.f);
        auto var8 = _mm256_set1_ps(62370.f);
        auto var9 = _mm256_set1_ps(135135.f);
        auto var10 = _mm256_set1_ps(0.5);
        auto varOne = _mm256_set1_ps(1.f);
        auto varNegOne = _mm256_set1_ps(-1.f);

        for (; i < len - 7; i+=8) {
            auto x = _mm256_loadu_ps(inputData + i);  
            // sqrt(2 / PI) * (0.044715 * x^3 + x)
            auto y = _mm256_mul_ps(x, x);
            y = _mm256_mul_ps(y, x);
            y = _mm256_mul_ps(y, var1);
            y = _mm256_add_ps(y, x);
            y = _mm256_mul_ps(y, var2);

            // y = tanh(y)
            {
            auto y2 = _mm256_mul_ps(y, y);
            auto w = _mm256_add_ps(y2, var3);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var4);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var5);
            w = _mm256_mul_ps(w, y);
            auto z = _mm256_mul_ps(y2, var6);
            z = _mm256_add_ps(z, var7);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var8);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var9);
            z = _mm256_div_ps(w, z);
            z = _mm256_max_ps(z, varNegOne);
            y = _mm256_min_ps(z, varOne);
            }

            y = _mm256_add_ps(y, varOne);
            y = _mm256_mul_ps(y, x);
            y = _mm256_mul_ps(y, var10);
            _mm256_storeu_ps(outputData + i, y);
        }
#endif
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
        }
    }

    void DoCpuSwigluReshape(Data &input, Data &output) {
        std::vector <int> dims = input.dims;
        dims[dims.size() - 1] /= 2;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void DoCpuGegluReshape(Data &input, Data &output) {
        DoCpuSwigluReshape(input, output);
    }

    void CpuGegluOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        DoCpuGegluReshape(input, output);
    }

    void DoCpuGeglu(Data &input, Data &output) {
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16 ||
                        input.dataType == DataType::BFLOAT16,
                        "Geglu error: Data's type should be float32, float16 or bfloat16.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
        int outer = input.Count(0) / spatial;

        if (input.dataType == DataType::FLOAT32) {
            (MultiThreadGegluOp((float*)inputData, spatial / 2, spatial / 2, (float*)outputData, outer, spatial, spatial / 2)).Run();
        } else if (input.dataType == DataType::FLOAT16) {
            (MultiThreadGegluFloat16Op((uint16_t*)inputData, spatial / 2, spatial / 2, (uint16_t*)outputData, outer, spatial, spatial / 2)).Run();
        } else if (input.dataType == DataType::BFLOAT16) {
            (MultiThreadGegluBFloat16Op((uint16_t*)inputData, spatial / 2, spatial / 2, (uint16_t*)outputData, outer, spatial, spatial / 2)).Run();
        } else {
            printf("Unsupport geglu type.");
        }
    }

    void CpuGegluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16 ||
                        input.dataType == DataType::BFLOAT16,
                        "Geglu error: Data's type should be float32, float16 or bfloat16.\n");

        int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
        int outer = input.Count(0) / spatial;

        if (input.dataType == DataType::FLOAT32) {
            GegluMultiThread((float*)input.cpuData, mid, mid, (float*)output.cpuData, outer, spatial, mid, GetAlivePool());
        } else if (input.dataType == DataType::FLOAT16) {
            GegluMultiThreadFloat16((uint16_t*)input.cpuData, mid, mid, (uint16_t*)output.cpuData, outer, spatial, mid, GetAlivePool());
        } else if (input.dataType == DataType::BFLOAT16) {
            GegluMultiThreadBFloat16((uint16_t*)input.cpuData, mid, mid, (uint16_t*)output.cpuData, outer, spatial, mid, GetAlivePool());
        } else {
            printf("Unsupport geglu type.");
        }
    }

    void CpuSwigluOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        std::vector <int> dims = input.dims;
        dims[dims.size() - 1] /= 2;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void DoCpuSwiglu(Data &input, Data &output) {
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16 ||
                        input.dataType == DataType::BFLOAT16,
                        "Swiglu error: Data's type should be float32, float16 or bfloat16.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
        int outer = input.Count(0) / spatial;

        if (input.dataType == DataType::FLOAT32) {
            (MultiThreadSwigluOp((float*)inputData, spatial / 2, spatial / 2, (float*)outputData, outer, spatial, spatial / 2)).Run();
        } else if (input.dataType == DataType::FLOAT16) {
            (MultiThreadSwigluFloat16Op((uint16_t*)inputData, spatial / 2, spatial / 2, (uint16_t*)outputData, outer, spatial, spatial / 2)).Run();
        } else if (input.dataType == DataType::BFLOAT16) {
            (MultiThreadSwigluBFloat16Op((uint16_t*)inputData, spatial / 2, spatial / 2, (uint16_t*)outputData, outer, spatial, spatial / 2)).Run();
        } else {
            printf("Unsupport swiglu type.");
        }
    }

    void CpuSwigluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16 ||
                        input.dataType == DataType::BFLOAT16,
                        "Swiglu error: Data's type should be float32, float16 or bfloat16.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
        int outer = input.Count(0) / spatial;

        if (input.dataType == DataType::FLOAT32) {
            (SwigluMultiThread((float*)inputData, spatial / 2, spatial / 2, (float*)outputData, outer, spatial, spatial / 2, GetAlivePool()));
        } else if (input.dataType == DataType::FLOAT16) {
            (SwigluMultiThreadFloat16((uint16_t*)inputData, spatial / 2, spatial / 2, (uint16_t*)outputData, outer, spatial, spatial / 2, GetAlivePool()));
        } else if (input.dataType == DataType::BFLOAT16) {
            (SwigluMultiThreadBFloat16((uint16_t*)inputData, spatial / 2, spatial / 2, (uint16_t*)outputData, outer, spatial, spatial / 2, GetAlivePool()));
        } else {
            printf("Unsupport swiglu type.");
        }
        return;
    }

    void CpuCrossSwigluOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        std::vector <int> dims = input.dims;
        dims[dims.size() - 1] /= 2;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuCrossSwigluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32,
                        "CrossSwiglu error: Data's type should be float32.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
        int outer = input.Count(0) / spatial;

        // CrossSwiglu: 交替存储格式, y[i] = x[i*2+1] * silu(x[i*2])
        CrossSwigluMultiThread((float*)inputData, mid, mid, (float*)outputData, outer, spatial, mid, GetAlivePool());

        return;
    }

    void CpuSwigluGptOssOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        std::vector <int> dims = input.dims;
        dims[dims.size() - 1] /= 2;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuSwigluGptOssOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16 ||
                        input.dataType == DataType::BFLOAT16,
                        "Swiglu error: Data's type should be float32, float16 or bfloat16.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
        int outer = input.Count(0) / spatial;

        if (input.dataType == DataType::FLOAT32) {
            (SwigluGptOssMultiThread((float*)inputData, spatial / 2, spatial / 2, (float*)outputData, outer, spatial, spatial / 2, GetAlivePool()));
        } else if (input.dataType == DataType::FLOAT16) {
            ErrorInFastLLM("Error: Gpt oss swiglu, data type should be f32.");
            // (SwigluMultiThreadFloat16((uint16_t*)inputData, spatial / 2, spatial / 2, (uint16_t*)outputData, outer, spatial, spatial / 2, GetAlivePool()));
        } else {
            printf("Unsupport swiglu type.");
        }
        return;
    }

    inline float softplus(float x) {
        return  x > 20.0f ? x : std::log1p(std::exp(x));
    }

    void CpuMambaSoftplusOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &aLogData = *(datas.find("aLog")->second);
        Data &dtBiasData = *(datas.find("dtBias")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "CpuMambaSoftplusOp error: Data's type should be float32 or float16.\n");
        AssertInFastLLM(aLogData.dataType == DataType::FLOAT32 && dtBiasData.dataType == DataType::FLOAT32,
                        "CpuMambaSoftplusOp error: alog's type and dtbias's type should be float32.\n");

        int dimsLen = input.dims.size();
        int outer = input.Count(0) / input.Count(dimsLen - 1);
        int channels = input.dims[dimsLen - 1];

        float *aLog = (float*)aLogData.cpuData;
        float *dtBias = (float*)dtBiasData.cpuData;

        // g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if (input.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *outputData = (float *) output.cpuData;
            for (int o = 0; o < outer; o++) {
                for (int i = 0; i < channels; i++) {
                    outputData[o * channels + i] = -exp(aLog[i]) * softplus(inputData[o * channels + i] + dtBias[i]);
                }
            }
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t *) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;

            for (int o = 0; o < outer; o++) {
                for (int i = 0; i < channels; i++) {
                    outputData[o * channels + i] = float_to_half(-exp(aLog[i]) * softplus(
                        fp16tofp32.dict[inputData[o * channels + i]] + dtBias[i]));
                }
            }
        }
    }

    struct MultiThreadMulScalarOp : MultiThreadBaseOp {
        void *input, *output;
        float v;
        long long st, end;
        bool isFloat16;

        MultiThreadMulScalarOp (void *input, void *output, float v, long long st, long long end, bool isFloat16) :
            input(input), output(output), v(v), st(st), end(end), isFloat16(isFloat16) {}

        void Run() {
            if (!isFloat16) {
                float *inputData = (float*)input + st;
                float *outputData = (float*)output + st;
                for (long long i = 0; i < end - st; i++) {
                    outputData[i] = inputData[i] * v;
                }
            } else {
                uint16_t *inputData = (uint16_t*)input + st;
                uint16_t *outputData = (uint16_t*)output + st;
                for (long long i = 0; i < end - st; i++) {
                    outputData[i] = float_to_half(fp16tofp32.dict[inputData[i]] * v);
                }
            }
        }
    };

    static void RunMultiThreadMulScalar(void *input, void *output, float v, long long len,
                                        bool isFloat16, AliveThreadPool *pool) {
        if (len < 65536) {
            MultiThreadMulScalarOp(input, output, v, 0, len, isFloat16).Run();
            return;
        }
        int threadNum = pool->threads.size();
        long long per = len / threadNum;
        std::vector<fastllm::MultiThreadMulScalarOp*> ops;
        long long cur = 0;
        for (int i = 0; i < threadNum; i++) {
            long long end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new MultiThreadMulScalarOp(input, output, v, cur, end, isFloat16));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuMulOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        auto mulSt = std::chrono::system_clock::now();
        output.Allocate();
        auto mulAlloc = std::chrono::system_clock::now();

        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Mul error: Data's type should be float32 or float16.\n");

        int len = input.Count(0);

        if (input.dataType == DataType::FLOAT32) {
            RunMultiThreadMulScalar(input.cpuData, output.cpuData, v, len, false, GetAlivePool());
        } else if (input.dataType == DataType::FLOAT16) {
            RunMultiThreadMulScalar(input.cpuData, output.cpuData, v, len, true, GetAlivePool());
        }

        static const bool mulProfSlow = std::getenv("FASTLLM_PROFILE_SLOW_OPS") != nullptr;
        if (mulProfSlow) {
            float allocSpend = GetSpan(mulSt, mulAlloc);
            float copySpend = GetSpan(mulAlloc, std::chrono::system_clock::now());
            if (allocSpend + copySpend > 0.02f) {
                printf("[fastllm-mul-detail] alloc=%.6f copy=%.6f total=%.6f s (len=%d)\n",
                       allocSpend, copySpend, allocSpend + copySpend, len);
                fflush(stdout);
            }
        }
    }

    void CpuAddOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Add error: Data's type should be float32 or float16.\n");

        int len = input.Count(0);

        if (input.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *outputData = (float *) output.cpuData;
            for (int i = 0; i < len; i++) {
                outputData[i] = inputData[i] + v;
            }
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t *) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;
            for (int i = 0; i < len; i++) {
                outputData[i] = float_to_half(fp16tofp32.dict[inputData[i]] + v);
            }
        }
    }

    struct MultiThreadMulToOp : MultiThreadBaseOp {
        uint8_t *input0Data, *input1Data;
        long long st, end;
        int dataType;          // 0: float32, 1: float16, 2: bfloat16
        long long channelLen;  // > 0: input1 按 i / channelLen 广播
        bool scalar;           // input1 只有 1 个元素

        MultiThreadMulToOp (uint8_t *input0Data, uint8_t *input1Data, long long st, long long end,
                            int dataType, long long channelLen, bool scalar) :
            input0Data(input0Data), input1Data(input1Data), st(st), end(end),
            dataType(dataType), channelLen(channelLen), scalar(scalar) {}

        void Run() {
            if (dataType == 0) {
                float *a = (float*)input0Data + st;
                float *b = (float*)input1Data;
                if (scalar) {
                    float v = b[0];
                    for (long long i = 0; i < end - st; i++) {
                        a[i] *= v;
                    }
                } else if (channelLen > 0) {
                    for (long long i = 0; i < end - st; i++) {
                        a[i] *= b[(st + i) / channelLen];
                    }
                } else {
                    float *bb = b + st;
                    for (long long i = 0; i < end - st; i++) {
                        a[i] *= bb[i];
                    }
                }
            } else if (dataType == 1) {
                uint16_t *a = (uint16_t*)input0Data + st;
                uint16_t *b = (uint16_t*)input1Data;
                if (scalar) {
                    float v = fp16tofp32.dict[b[0]];
                    for (long long i = 0; i < end - st; i++) {
                        a[i] = float_to_half(fp16tofp32.dict[a[i]] * v);
                    }
                } else if (channelLen > 0) {
                    for (long long i = 0; i < end - st; i++) {
                        a[i] = float_to_half(fp16tofp32.dict[a[i]] * fp16tofp32.dict[b[(st + i) / channelLen]]);
                    }
                } else {
                    uint16_t *bb = b + st;
                    for (long long i = 0; i < end - st; i++) {
                        a[i] = float_to_half(fp16tofp32.dict[a[i]] * fp16tofp32.dict[bb[i]]);
                    }
                }
            } else {
                uint16_t *a = (uint16_t*)input0Data + st;
                uint16_t *b = (uint16_t*)input1Data;
                if (scalar) {
                    float v = BFloat16BitsToFloat32(b[0]);
                    for (long long i = 0; i < end - st; i++) {
                        a[i] = Float32ToBFloat16RNEBits(BFloat16BitsToFloat32(a[i]) * v);
                    }
                } else if (channelLen > 0) {
                    for (long long i = 0; i < end - st; i++) {
                        a[i] = Float32ToBFloat16RNEBits(
                            BFloat16BitsToFloat32(a[i]) * BFloat16BitsToFloat32(b[(st + i) / channelLen]));
                    }
                } else {
                    uint16_t *bb = b + st;
                    for (long long i = 0; i < end - st; i++) {
                        a[i] = Float32ToBFloat16RNEBits(
                            BFloat16BitsToFloat32(a[i]) * BFloat16BitsToFloat32(bb[i]));
                    }
                }
            }
        }
    };

    static void RunMultiThreadMulTo(uint8_t *input0Data, uint8_t *input1Data, long long len,
                                    int dataType, long long channelLen, bool scalar,
                                    AliveThreadPool *pool) {
        if (len < 65536) {
            MultiThreadMulToOp(input0Data, input1Data, 0, len, dataType, channelLen, scalar).Run();
            return;
        }
        int threadNum = pool->threads.size();
        long long per = len / threadNum;
        std::vector<fastllm::MultiThreadMulToOp*> ops;
        long long cur = 0;
        for (int i = 0; i < threadNum; i++) {
            long long end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new MultiThreadMulToOp(input0Data, input1Data, cur, end, dataType, channelLen, scalar));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuMulToOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        int input0Len = input0.Count(0);
        int input1Len = input1.Count(0);
        AssertInFastLLM(input0.dataType == input1.dataType &&
                        (input0.dataType == DataType::FLOAT32 ||
                         input0.dataType == DataType::FLOAT16 ||
                         input0.dataType == DataType::BFLOAT16),
                        "MulTo error: inputs should have the same float32, float16 or bfloat16 type.\n");
        AssertInFastLLM(input0.dims == input1.dims || input1Len == 1 || input0Len % input1Len == 0, "MulTo error: input's shape should be same.\n");

        int len = input0.Count(0);
        int inner = input1.Count(0);
        AssertInFastLLM(len % inner == 0, "MulTo error: Data`s shape can`t perform MulTo operation.\n");
        int round = (len / inner);

        int dataType = (input0.dataType == DataType::FLOAT16 ? 1 :
                        (input0.dataType == DataType::BFLOAT16 ? 2 : 0));
        if (input1Len == 1) {
            RunMultiThreadMulTo(input0.cpuData, input1.cpuData, len, dataType, 0, true, GetAlivePool());
        } else if (input0Len == input1Len) {
            RunMultiThreadMulTo(input0.cpuData, input1.cpuData, len, dataType, 0, false, GetAlivePool());
        } else {
            int channelLen = input0Len / input1Len;
            RunMultiThreadMulTo(input0.cpuData, input1.cpuData, len, dataType, channelLen, false, GetAlivePool());
        }
    }

    struct MultiThreadAddToFloatOp : MultiThreadBaseOp {
        float *input, *output;
        int len;
        float alpha;

        MultiThreadAddToFloatOp (float *output, float *input, float alpha, int len) : input(input), output(output), alpha(alpha), len(len) {}

        void Run() {
            for (int i = 0; i < len; i++) {
                output[i] += input[i] * alpha;
            }
        }
    };

    static void RunMultiThreadAddToFloat(float *output, float *input, float alpha, int len, AliveThreadPool *pool) {
        if (len < 256 * 1024) {
            (MultiThreadAddToFloatOp(output, input, alpha, len)).Run();
            return;
        }
        int threadNum = pool->threads.size();
        int per = len / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadAddToFloatOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new MultiThreadAddToFloatOp(output + cur, input + cur, alpha, end - cur));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuAddToOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;

        if (!((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
              (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
              (input0.dataType == DataType::BFLOAT16 && input1.dataType == DataType::BFLOAT16))) {
            std::ostringstream oss;
            oss << "AddTo error: Data's type should be float32, float16 or bfloat16, and both inputs should use the same type."
                << " input0.name=" << input0.name
                << " input1.name=" << input1.name
                << " input0.type=" << (int) input0.dataType
                << " input1.type=" << (int) input1.dataType
                << " input0.device=" << (int) input0.dataDevice
                << " input1.device=" << (int) input1.dataDevice
                << " input0.dims=[";
            for (int i = 0; i < input0.dims.size(); i++) {
                if (i > 0) oss << ",";
                oss << input0.dims[i];
            }
            oss << "] input1.dims=[";
            for (int i = 0; i < input1.dims.size(); i++) {
                if (i > 0) oss << ",";
                oss << input1.dims[i];
            }
            oss << "] alpha=" << alpha;
            ErrorInFastLLM(oss.str());
        }
        AssertInFastLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");

        int len = input0.Count(0);

        if (input0.dataType == DataType::FLOAT32) {
            float *input0Data = (float *) input0.cpuData;
            float *input1Data = (float *) input1.cpuData;
            RunMultiThreadAddToFloat(input0Data, input1Data, alpha, len, GetAlivePool());
        } else if (input0.dataType == DataType::FLOAT16) {
            uint16_t *input0Data = (uint16_t *) input0.cpuData;
            uint16_t *input1Data = (uint16_t *) input1.cpuData;
            for (int i = 0; i < len; i++) {
                input0Data[i] = float_to_half(fp16tofp32.dict[input0Data[i]] + fp16tofp32.dict[input1Data[i]] * alpha);
            }
        } else if (input0.dataType == DataType::BFLOAT16) {
            uint16_t *input0Data = (uint16_t *) input0.cpuData;
            uint16_t *input1Data = (uint16_t *) input1.cpuData;
            for (int i = 0; i < len; i++) {
                float val = bf16tofp32.dict[input0Data[i]] + bf16tofp32.dict[input1Data[i]] * alpha;
                input0Data[i] = Float32ToBFloat16RNEBits(val);
            }
        }
    }

    void CpuRecurrentGatedDeltaRuleOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &last_recurrent_state = *(datas.find("last_recurrent_state")->second);
        Data &core_attn_out = *(datas.find("core_attn_out")->second);
        
        std::vector <int> dims = last_recurrent_state.dims;
        core_attn_out.dataType = last_recurrent_state.dataType;
        core_attn_out.Resize({dims[0], dims[1], 1, dims[3]});
    }

    struct MultiThreadRecurrentGatedDeltaRuleOp : MultiThreadBaseOp {
        int n0, n1, n2, n3, group;
        float *flast, *fgt, *fkt, *fvt, *fbt, *fqt, *fatv;
        int st, end;

        MultiThreadRecurrentGatedDeltaRuleOp(
            int n0, int n1, int n2, int n3, int group, 
            float *flast, float *fgt, float *fkt, float *fvt, 
            float *fbt, float *fqt, float *fatv,
            int st, int end
        ) : n0(n0), n1(n1), n2(n2), n3(n3), group(group),
            flast(flast), fgt(fgt), fkt(fkt), fvt(fvt),
            fbt(fbt), fqt(fqt), fatv(fatv),
            st(st), end(end) {}
        
        void Run() {
            std::vector <float> fkv_mem, temp;
            fkv_mem.resize(n3);
            temp.resize(n3);
            for (int i = st; i < end; i++) {
                float v = exp(fgt[i]);
                for (int j = 0; j < n2 * n3; j++) {
                    flast[i * n2 * n3 + j] *= v;
                }

                std::fill(fkv_mem.begin(), fkv_mem.end(), 0.0f);
                for (int j = 0; j < n2; j++) {
                    float curfkt = fkt[i / group * n2 + j];
                    for (int k = 0; k < n3; k++) {
                        fkv_mem[k] += flast[i * n2 * n3 + j * n3 + k] * curfkt;
                    }
                }

                float curfbt = fbt[i];
                for (int k = 0; k < n3; k++) {
                    temp[k] = ((fvt[i * n3 + k] - fkv_mem[k]) * curfbt);
                }

                for (int j = 0; j < n2; j++) {
                    for (int k = 0; k < n3; k++) {
                        flast[i * n2 * n3 + j * n3 + k] += fkt[i / group * n2 + j] * temp[k];
                    }
                }

                for (int j = 0; j < n2; j++) {
                    float curfqt = fqt[i / group * n2 + j];
                    for (int k = 0; k < n3; k++) {
                        fatv[i * n3 + k] += flast[i * n2 * n3 + j * n3 + k] * curfqt;
                    }
                }
            }
        }
    };

    void CpuRecurrentGatedDeltaRuleOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &g = *(datas.find("g")->second);
        Data &b = *(datas.find("b")->second);
        Data &last_recurrent_state = *(datas.find("last_recurrent_state")->second);
        Data &core_attn_out = *(datas.find("core_attn_out")->second);
        core_attn_out.Allocate(0.0f);
        
        Data &q_t = q, &k_t = k, &v_t = v, &g_t = g, &b_t = b;

        // last_recurrent_state = last_recurrent_state * g_t
        // kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        // delta = (v_t - kv_mem) * beta_t
        // last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        // core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
        int n0 = last_recurrent_state.dims[0], n1 = last_recurrent_state.dims[1], n2 = last_recurrent_state.dims[2], n3 = last_recurrent_state.dims[3];
        float *flast = (float*)last_recurrent_state.cpuData;
        float *fgt = (float*)g_t.cpuData;
        float *fkt = (float*)k_t.cpuData;
        float *fvt = (float*)v_t.cpuData;
        float *fbt = (float*)b_t.cpuData;
        float *fqt = (float*)q_t.cpuData;
        float *fatv = (float*)core_attn_out.cpuData;

        int group = v.dims[1] / q.dims[1];
        std::vector <float> lastVector, gtVector, ktVector, vtVector, btVector, qtVector, atvVector;
        if (q.dataType == DataType::FLOAT16) {
            lastVector.resize(last_recurrent_state.Count(0));
            gtVector.resize(g_t.Count(0));
            ktVector.resize(k_t.Count(0));
            vtVector.resize(v_t.Count(0));
            btVector.resize(b_t.Count(0));
            qtVector.resize(q_t.Count(0));
            atvVector.resize(core_attn_out.Count(0));

            flast = (float*)lastVector.data();
            fgt = (float*)gtVector.data();
            fkt = (float*)ktVector.data();
            fvt = (float*)vtVector.data();
            fbt = (float*)btVector.data();
            fqt = (float*)qtVector.data();
            fatv = (float*)atvVector.data();

            Float16ToFloat32((uint16_t*)last_recurrent_state.cpuData, flast, (int)lastVector.size());
            Float16ToFloat32((uint16_t*)g_t.cpuData, fgt, (int)gtVector.size());
            Float16ToFloat32((uint16_t*)k_t.cpuData, fkt, (int)ktVector.size());
            Float16ToFloat32((uint16_t*)v_t.cpuData, fvt, (int)vtVector.size());
            Float16ToFloat32((uint16_t*)b_t.cpuData, fbt, (int)btVector.size());
            Float16ToFloat32((uint16_t*)q_t.cpuData, fqt, (int)qtVector.size());
        }

        int n = n0 * n1;
        auto pool = GetAlivePool();
        int threadNum = pool->threads.size();
        int per = n / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadRecurrentGatedDeltaRuleOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
            ops.push_back(new MultiThreadRecurrentGatedDeltaRuleOp(n0, n1, n2, n3, group, flast, fgt, fkt, fvt, fbt, fqt, fatv, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }

        if (q.dataType == DataType::FLOAT16) {
            Float32ToFloat16(fatv, (uint16_t*)core_attn_out.cpuData, (int)atvVector.size());
        }
    }

    struct MultiThreadTransferAttnOp : MultiThreadBaseOp {
        float *inputData;
        int n, m, st, end;

        MultiThreadTransferAttnOp (float *inputData, int n, int m, int st, int end) :
            inputData(inputData), n(n), m(m), st(st), end(end) {}

        void Run() {
            std::vector <float> tempRow(m);
            std::vector <float> tempSub((long long)m * m);
            for (int o = st; o < end; o++) {
                float *batchData = inputData + (long long)o * n * m;

                for (int i = 1; i < n; i++) {
                    // 复制第 i 行的前 i 个元素到临时数组
                    std::memcpy(tempRow.data(), batchData + i * m, i * sizeof(float));

                    // 复制子矩阵到临时数组
                    for (int k = 0; k < i; k++) {
                        std::memcpy(tempSub.data() + k * m, batchData + k * m, i * sizeof(float));
                    }

                    // 更新第 i 行的前 i 个元素
                    for (int j = 0; j < i; j++) {
                        float sum = tempRow[j];
                        for (int k = 0; k < i; k++) {
                            sum += tempRow[k] * tempSub[k * m + j];
                        }
                        batchData[i * m + j] = sum;
                    }
                }

                // attn = attn + torch.eye(chunk_size, ...)
                for (int i = 0; i < n; i++) {
                    inputData[(long long)o * n * n + i * m + i] += 1.0f;
                }
            }
        }
    };

    void CpuTransferAttnOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        float *inputData = (float*)input.cpuData;
        int dimsLen = input.dims.size();
        int n = input.dims[dimsLen - 2], m = input.dims[dimsLen - 1], outer = input.Count(0) / input.Count(dimsLen - 2);

        std::vector <float> floatInputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            inputData = (float*)floatInputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        if ((long long)outer * n * m < 65536 || outer <= 1) {
            MultiThreadTransferAttnOp(inputData, n, m, 0, outer).Run();
        } else {
            auto *pool = GetAlivePool();
            int threadNum = std::min((int)pool->threads.size(), outer);
            int per = outer / threadNum;
            std::vector<fastllm::MultiThreadTransferAttnOp*> ops;
            int cur = 0;
            for (int i = 0; i < threadNum; i++) {
                int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
                ops.push_back(new MultiThreadTransferAttnOp(inputData, n, m, cur, end));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(inputData, (uint16_t*)input.cpuData, (int)floatInputVector.size());
        }
    }

    void CpuApplyChunkDecayByLastLogGOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &g = *(datas.find("g")->second);
        AssertInFastLLM((input.dataType == DataType::FLOAT32 && g.dataType == DataType::FLOAT32) ||
                        (input.dataType == DataType::FLOAT16 && g.dataType == DataType::FLOAT16) ||
                        (input.dataType == DataType::BFLOAT16 && g.dataType == DataType::BFLOAT16),
                        "ApplyChunkDecayByLastLogG's input's type should be float32, float16 or bfloat16.\n");
        AssertInFastLLM(input.dims.size() >= 2 && g.dims.size() >= 1,
                        "ApplyChunkDecayByLastLogG error: invalid dims.\n");
        int dim = input.dims[input.dims.size() - 2];
        int channels = input.dims.back();
        AssertInFastLLM(g.dims.back() == dim && input.Count(0) == g.Count(0) * channels,
                        "ApplyChunkDecayByLastLogG error: input and g shape mismatch.\n");

        int outer = g.Count(0) / dim;
        std::vector<float> inputFp32, gFp32;
        float *inputData = nullptr, *gData = nullptr;

        if (input.dataType == DataType::FLOAT32) {
            inputData = (float*)input.cpuData;
            gData = (float*)g.cpuData;
        } else if (input.dataType == DataType::FLOAT16) {
            inputFp32.resize(input.Count(0));
            gFp32.resize(g.Count(0));
            inputData = inputFp32.data();
            gData = gFp32.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, input.Count(0));
            Float16ToFloat32((uint16_t*)g.cpuData, gData, g.Count(0));
        } else {
            inputFp32.resize(input.Count(0));
            gFp32.resize(g.Count(0));
            inputData = inputFp32.data();
            gData = gFp32.data();
            BFloat16ToFloat32((uint16_t*)input.cpuData, inputData, input.Count(0));
            BFloat16ToFloat32((uint16_t*)g.cpuData, gData, g.Count(0));
        }

        for (int o = 0; o < outer; o++) {
            float last = gData[o * dim + dim - 1];
            for (int i = 0; i < dim; i++) {
                float scale = std::exp(last - gData[o * dim + i]);
                float *cur = inputData + ((size_t)o * dim + i) * channels;
                for (int c = 0; c < channels; c++) {
                    cur[c] *= scale;
                }
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(inputData, (uint16_t*)input.cpuData, input.Count(0));
        } else if (input.dataType == DataType::BFLOAT16) {
            Float32ToBFloat16(inputData, (uint16_t*)input.cpuData, input.Count(0));
        }
    }

    void CpuCausalMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        int base = intParams.find("base") != intParams.end() ? intParams.find("base")->second : 0;
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;

        float *inputData = (float*)input.cpuData;

        std::vector <float> floatInputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            inputData = (float*)floatInputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        int dimsLen = input.dims.size();
        int n = input.dims[dimsLen - 2], m = input.dims[dimsLen - 1], outer = input.Count(0) / input.Count(dimsLen - 2);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < n; i++) {
                for (int j = i + base; j < m; j++) {
                    inputData[o * n * m + i * m + j] = maskValue;
                }
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(inputData, (uint16_t*)input.cpuData, (int)floatInputVector.size());
        }
    }

    void CpuAttentionMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];

        AssertInFastLLM(mask.dataType == DataType::FLOAT32 || mask.dataType == input.dataType, "AttentionMask: mask's datatype should be float32.");
        if (input.dataType == DataType::FLOAT32) {
            float *maskData = (float *) mask.cpuData;
            float *attnData = (float *) input.cpuData;
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        if (maskData[on * spatial + i] > 0.99) {
                            attnData[o * spatial + i] = maskValue;
                        }
                    }
                }
            }
        } else if (input.dataType == DataType::FLOAT16 && mask.dataType == DataType::FLOAT32) {
            float *maskData = (float *) mask.cpuData;
            uint16_t *attnData = (uint16_t *) input.cpuData;
            uint16_t hMaskValue = float_to_half(maskValue);
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        if (maskData[on * spatial + i] > 0.99) {
                            attnData[o * spatial + i] = hMaskValue;
                        }
                    }
                }
            }
        } else if (input.dataType == DataType::FLOAT16 && mask.dataType == DataType::FLOAT16) {
            std::vector <float> floatMaskData;
            floatMaskData.resize(mask.Count(0));
            Float16ToFloat32((uint16_t*)mask.cpuData, floatMaskData.data(), mask.Count(0));
            float *maskData = floatMaskData.data();
            uint16_t *attnData = (uint16_t *) input.cpuData;
            uint16_t hMaskValue = float_to_half(maskValue);
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        if (maskData[on * spatial + i] > 0.99) {
                            attnData[o * spatial + i] = hMaskValue;
                        }
                    }
                }
            }
        } else {
            ErrorInFastLLM("AttentionMask error: unsupport input's dataType.\n");
        }
    }

    void CpuAttentionExtendedMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        int spatial = input.dims[3], n = input.dims[0], m = input.dims[1] * input.dims[2];

        AssertInFastLLM(mask.dataType == DataType::FLOAT32, "AttentionExtendedMask: mask's datatype should be float32.");
        if (input.dataType == DataType::FLOAT32) {
            float *maskData = (float *) mask.cpuData;
            float *attnData = (float *) input.cpuData;
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        attnData[o * spatial + i] += maskData[on * spatial + i];
                    }
                }
            }
        } else {
            ErrorInFastLLM("AttentionExtendedMask error: unsupport input's dataType.\n");
        }
    }

    void CpuAlibiMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        float *maskData = (float *) mask.cpuData;
        float *attnData = (float *) input.cpuData;
        int n = input.dims[0], m = input.dims[1];
        int spn = input.dims[2], spm = input.dims[3];
        int spatial = input.Count(2);
        for (int on = 0; on < n; on++) {
            for (int om = 0; om < m; om++) {
                float now = maskData[om];
                int o = on * m + om;
                float *inputNow = attnData + o * spatial;
                for (int i = 0; i < spn; i++) {
                    int mid = (spm - spn + i);
                    for (int j = 0; j <= mid; j++) {
                        inputNow[i * spm + j] += now * j;
                    }
                    for (int j = mid + 1; j < spm; j++) {
                        inputNow[i * spm + j] = maskValue;
                    }
                }
            }
        }
    }

    void CpuTopKOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;

        AssertInFastLLM(input.dataType == DataType::FLOAT32, "TopK error: Data's type should be float32.\n");

        int dimsLen = input.dims.size();
        std::vector<int> dims = input.dims;
        dims[dimsLen - 1] = topk * 2;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuTopKOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : -1;
        int dimsLen = input.dims.size();
        int outer = input.Count(0) / input.Count(dimsLen - 1);
        int channels = input.dims[dimsLen - 1];

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        if (topk == 1) {
            for (int o = 0; o < outer; o++) {
                float maxValue = -1e100, idx = -1;
                for (int j = 0; j < channels; j++) {
                    if (inputData[j] > maxValue) {
                        maxValue = inputData[j];
                        idx = j;
                    }
                }
                outputData[0] = idx;
                outputData[1] = maxValue;
                inputData += channels;
                outputData += 2;
            }
        } else {
            for (int o = 0; o < outer; o++) {
                std::set <std::pair <float, int> > ans;
                for (int j = 0; j < channels; j++) {
                    if (ans.size() == topk) {
                        if (ans.begin()->first < inputData[j]) {
                            ans.erase(ans.begin());
                            ans.insert(std::make_pair(inputData[j], j));
                        }
                    } else {
                        ans.insert(std::make_pair(inputData[j], j));
                    }
                }

                int j = topk - 1;
                for (auto &it : ans) {
                    outputData[j * 2] = it.second;
                    outputData[j * 2 + 1] = it.first;
                    j--;
                }

                inputData += channels;
                outputData += 2 * topk;
            }
            return;
/*
            for (int o = 0; o < outer; o++) {
                std::vector <std::pair <float, int> > v;
                for (int j = 0; j < channels; j++) {
                    v.push_back(std::make_pair(-inputData[j], j));
                }
                sort(v.begin(), v.end());
                for (int j = 0; j < topk; j++) {
                    outputData[j * 2] = v[j].second;
                    outputData[j * 2 + 1] = -v[j].first;
                }

                inputData += channels;
                outputData += 2 * topk;
            }
*/
        }
    }

    void CpuSelectExpertOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &logits = *(datas.find("logits")->second);
        Data &index = *(datas.find("index")->second);
        Data &score = *(datas.find("score")->second);
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;

        AssertInFastLLM(logits.dataType == DataType::FLOAT32, "SelectExpert error: logits's type should be float32.\n");
        
        int dimsLen = logits.dims.size();
        int n = logits.Count(0) / logits.dims[dimsLen - 1]; // number of tokens
        int numExperts = logits.dims[dimsLen - 1]; // number of experts

        // Index output: [n, topk]
        index.dataType = DataType::INT32;
        index.Resize({n, topk});
        
        // Score output: [n, topk]
        score.dataType = DataType::FLOAT32;
        score.Resize({n, topk});
    }

    void CpuSelectExpertOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &logits = *(datas.find("logits")->second);
        Data &index = *(datas.find("index")->second);
        Data &score = *(datas.find("score")->second);
        Data *gateBias = datas.find("gateBias") != datas.end() ? datas.find("gateBias")->second : nullptr;
        
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        bool needNorm = intParams.find("needNorm") != intParams.end() ? (intParams.find("needNorm")->second != 0) : false;
        float routeScale = floatParams.find("routeScale") != floatParams.end() ? floatParams.find("routeScale")->second : 1.0f;
        
        index.Allocate();
        score.Allocate();
        
        // 确保logits在CPU上且为FLOAT32
        ToDataType(logits, DataType::FLOAT32);
        logits.ToDevice(DataDevice::CPU);
        
        int dimsLen = logits.dims.size();
        int n = logits.Count(0) / logits.dims[dimsLen - 1]; // number of tokens
        int numExperts = logits.dims[dimsLen - 1]; // number of experts
        
        float *logitsData = (float*)logits.cpuData;
        int32_t *indexData = (int32_t*)index.cpuData;
        float *scoreData = (float*)score.cpuData;
        
        float *biasData = nullptr;
        if (gateBias != nullptr && gateBias->dims.size() > 0) {
            ToDataType(*gateBias, DataType::FLOAT32);
            gateBias->ToDevice(DataDevice::CPU);
            biasData = (float*)gateBias->cpuData;
        }
        
        for (int i = 0; i < n; i++) {
            float *curLogits = logitsData + i * numExperts;
            
            // 创建pair数组用于排序: (负值, 索引)
            std::vector<std::pair<float, int>> v;
            v.resize(numExperts);
            for (int j = 0; j < numExperts; j++) {
                v[j].first = -curLogits[j];
                v[j].second = j;
            }
            
            // 如果有bias，减去bias
            if (biasData != nullptr) {
                for (int j = 0; j < numExperts; j++) {
                    v[j].first -= biasData[j];
                }
            }
            
            // 使用partial_sort找到topk
            std::partial_sort(v.begin(), v.begin() + topk, v.end());
            
            // 计算归一化sum
            float sum = 1.0f;
            if (needNorm) {
                sum = 0.0f;
                for (int j = 0; j < topk; j++) {
                    sum += curLogits[v[j].second];
                }
            }
            
            // 填充输出
            for (int j = 0; j < topk; j++) {
                int expertIdx = v[j].second;
                indexData[i * topk + j] = expertIdx;
                scoreData[i * topk + j] = curLogits[expertIdx] / sum * routeScale;
            }
        }
    }

    void CpuPermuteOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16 ||
                        input.dataType == DataType::BFLOAT16, "Permute error: datatype should be float32, float16 or bfloat16.");
        AssertInFastLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");
        std::vector<int> new_dims;
        for (int i = 0; i < axis.size(); i++) {
            new_dims.push_back(input.dims[axis[i]]);
        }

        output.dataType = input.dataType;
        output.Resize(new_dims);
    }

    void Transpose4x4(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
        if (n < 4 || m < 4) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    pDst[j * dstStride + i] = pSrc[i * srcStride + j];
                }
            }

            return;
        }

#ifdef __aarch64__
        float32x4x2_t q01 = vtrnq_f32(vld1q_f32(pSrc), vld1q_f32(pSrc + srcStride));
        float32x4x2_t q23 = vtrnq_f32(vld1q_f32(pSrc + 2 * srcStride), vld1q_f32(pSrc + 3 * srcStride));

        float32x4_t qq0 = q01.val[0];
        float32x2_t d00 = vget_low_f32(qq0);
        float32x2_t d01 = vget_high_f32(qq0);

        float32x4_t qq1 = q01.val[1];
        float32x2_t d10 = vget_low_f32(qq1);
        float32x2_t d11 = vget_high_f32(qq1);

        float32x4_t qq2 = q23.val[0];
        float32x2_t d20 = vget_low_f32(qq2);
        float32x2_t d21 = vget_high_f32(qq2);

        float32x4_t qq3 = q23.val[1];
        float32x2_t d30 = vget_low_f32(qq3);
        float32x2_t d31 = vget_high_f32(qq3);

        vst1q_f32(pDst, vcombine_f32(d00, d20));
        vst1q_f32(pDst + 1 * dstStride, vcombine_f32(d10, d30));
        vst1q_f32(pDst + 2 * dstStride, vcombine_f32(d01, d21));
        vst1q_f32(pDst + 3 * dstStride, vcombine_f32(d11, d31));
#else
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pDst[j * dstStride + i] = pSrc[i * srcStride + j];
            }
        }
#endif
    }

    void Transpose(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
        int per = 4;
        for (int i = 0; i < n; i += per) {
            for (int j = 0; j < m; j += per) {
                Transpose4x4(pDst + j * dstStride + i,
                             pSrc + i * srcStride + j,
                             dstStride, srcStride,
                             std::min(per, n - i),
                             std::min(per, m - j));
            }
        }
    }

    struct MultiThreadTransposeOp : MultiThreadBaseOp {
        float *pDst, *pSrc;
        int dstStride, srcStride, n, m;

        MultiThreadTransposeOp(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) :
            pDst(pDst), pSrc(pSrc), dstStride(dstStride), srcStride(srcStride), n(n), m(m) {}
        
        void Run() {
            Transpose(pDst, pSrc, dstStride, srcStride, n, m);
        }
    };

    static void RunMultiThreadTransposeMatrix(float *pDst, float *pSrc, int dstStride, int srcStride,
                                              int n, int m, AliveThreadPool *pool) {
        if ((long long)n * m < 65536 || n <= 1) {
            Transpose(pDst, pSrc, dstStride, srcStride, n, m);
            return;
        }
        int threadNum = std::min((int)pool->threads.size(), n);
        int per = n / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadTransposeOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
            ops.push_back(new MultiThreadTransposeOp(pDst + cur, pSrc + (long long)cur * srcStride,
                                                     dstStride, srcStride, end - cur, m));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    struct MultiThreadPermuteCopyOp : MultiThreadBaseOp {
        uint8_t *tmpData, *curData;
        std::vector <int> axis, oldSteps, newSteps;
        int unitSize, st, end;

        MultiThreadPermuteCopyOp (uint8_t *tmpData, uint8_t *curData, std::vector <int> axis,
                                  std::vector <int> oldSteps, std::vector <int> newSteps,
                                  int unitSize, int st, int end) :
            tmpData(tmpData), curData(curData), axis(axis), oldSteps(oldSteps),
            newSteps(newSteps), unitSize(unitSize), st(st), end(end) {}

        int MapPos(int i) {
            int old = 0, idx = i;
            for (int j = 0; j < (int)axis.size(); ++j) {
                old += (idx / newSteps[j]) * oldSteps[axis[j]];
                idx %= newSteps[j];
            }
            return old;
        }

        void Run() {
            if (unitSize == 4) {
                for (int i = st; i < end; i++) {
                    ((float*)tmpData)[i] = ((float*)curData)[MapPos(i)];
                }
            } else if (unitSize == 2) {
                for (int i = st; i < end; i++) {
                    ((uint16_t*)tmpData)[i] = ((uint16_t*)curData)[MapPos(i)];
                }
            } else if (unitSize == 1) {
                for (int i = st; i < end; i++) {
                    ((uint8_t*)tmpData)[i] = ((uint8_t*)curData)[MapPos(i)];
                }
            }
        }
    };

    void CpuPermuteOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

        output.Allocate();
        uint8_t *tmpData = (uint8_t *) output.cpuData;
        uint8_t *curData = (uint8_t *) input.cpuData;

        if (axis == std::vector <int> {1, 2, 0} && input.dataType == DataType::FLOAT32) {
            int n = input.dims[0];
            int m = input.Count(1);
            RunMultiThreadTransposeMatrix((float*)tmpData, (float*)curData, n, m, n, m, GetAlivePool());
        } else if (axis == std::vector <int> {1, 0, 2}) {
            int n = input.dims[0];
            int m = input.dims[1];
            int k = input.dims[2];
            int unitSize = input.unitSize;
            RunMultiThreadTransposeByLine(tmpData, curData, n, m, k * unitSize, GetAlivePool());
        } else if (axis == std::vector <int> {2, 0, 1, 3}) {
            int n = input.dims[0] * input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            int unitSize = input.unitSize;
            RunMultiThreadTransposeByLine(tmpData, curData, n, m, k * unitSize, GetAlivePool());
        } else if (axis == std::vector<int> {0, 2, 1, 3}) {
            int b = input.dims[0];
            int n = input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            int unitSize = input.unitSize;
            for (int o = 0; o < b; o++) {
                RunMultiThreadTransposeByLine(tmpData, curData, n, m, k * unitSize, GetAlivePool());
                tmpData += output.Count(1) * unitSize;
                curData += input.Count(1) * unitSize;
            }
        } else {
            std::vector<int> oldSteps;
            std::vector<int> newSteps;
            int count = input.Count(0);
            for (int i = 0; i < axis.size(); i++) {
                oldSteps.push_back(input.Count(i + 1));
                newSteps.push_back(output.Count(i + 1));
            }

            if (count < 65536) {
                MultiThreadPermuteCopyOp(tmpData, curData, axis, oldSteps, newSteps,
                                         input.unitSize, 0, count).Run();
            } else {
                auto *pool = GetAlivePool();
                int threadNum = pool->threads.size();
                int per = count / threadNum;
                std::vector<fastllm::MultiThreadPermuteCopyOp*> ops;
                int cur = 0;
                for (int i = 0; i < threadNum; i++) {
                    int end = (i == threadNum - 1 ? count : cur + per + (cur + per * (threadNum - i) < count));
                    ops.push_back(new MultiThreadPermuteCopyOp(tmpData, curData, axis, oldSteps,
                                                               newSteps, input.unitSize, cur, end));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            }
        }
    }

    static std::vector <uint8_t> vold;

    struct MultiThreadPermuteSelfSwapOp : MultiThreadBaseOp {
        float *input;
        int n, m, st, end;
        std::vector <float> temp;

        MultiThreadPermuteSelfSwapOp (float *input, int n, int m, int st, int end) :
            input(input), n(n), m(m), st(st), end(end) {}

        void Run() {
            if ((int)temp.size() < n * m) {
                temp.resize(n * m);
            }
            for (int o = st; o < end; o++) {
                float *cur = input + (long long)o * n * m;
                memcpy(temp.data(), cur, (long long)n * m * sizeof(float));
                Transpose(cur, temp.data(), n, m, n, m);
            }
        }
    };

    void DoCpuPermuteSelf(Data &input, const std::vector <int> &axis) {
        bool same = false;
        same |= ((axis == std::vector <int>{1, 2, 0} || axis == std::vector <int>{1, 0, 2}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[2] == 1);
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[0] == 1 && input.dims[1] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1, 3}) && (input.dims[1] == 1 || input.dims[2] == 1));
        same |= ((axis == std::vector <int>{1, 0, 2, 3}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{1, 2, 0, 3}) && input.dims[1] == 1 && input.dims[2] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1}) && (input.dims[1] == 1 || input.dims[2] == 1));
        if (same) {
            std::vector<int> new_dims;
            for (int i = 0; i < axis.size(); i++) {
                new_dims.push_back(input.dims[axis[i]]);
            }
            input.Resize(new_dims);
            return;
        }

        std::vector<int> new_dims;
        for (int i = 0; i < axis.size(); i++) {
            new_dims.push_back(input.dims[axis[i]]);
        }

        bool swapLastTwoDims = false;
        if (input.dims.size() >= 2 && input.dims.size() == new_dims.size()) {
            std::vector <int> dims = input.dims;
            std::swap(dims[dims.size() - 2], dims[dims.size() - 1]);
            swapLastTwoDims = (dims == new_dims);
        }

        if (swapLastTwoDims && input.dataType == DataType::FLOAT32) {
            int dl = input.dims.size();
            int outer = input.Count(0) / input.Count(dl - 2);
            int n = input.dims[dl - 2], m = input.dims[dl - 1];
            float *finput = (float*)input.cpuData;
            if ((long long)outer * n * m < 65536) {
                float *temp = new float[n * m];
                for (int i = 0; i < outer; i++) {
                    memcpy(temp, finput + (long long)i * n * m, (long long)n * m * sizeof(float));
                    Transpose(finput + (long long)i * n * m, temp, n, m, n, m);
                }
                delete[] temp;
            } else if (outer == 1) {
                float *temp = new float[n * m];
                uint64_t copyBytes = (uint64_t)n * m * sizeof(float);
                if (copyBytes < 2147483648ULL) {
                    RunMultiThreadMemcpy((uint8_t*)temp, input.cpuData, (int)copyBytes, GetAlivePool());
                } else {
                    memcpy(temp, input.cpuData, copyBytes);
                }
                RunMultiThreadTransposeMatrix(finput, temp, n, m, n, m, GetAlivePool());
                delete[] temp;
            } else {
                auto *pool = GetAlivePool();
                int threadNum = std::min((int)pool->threads.size(), outer);
                int per = outer / threadNum;
                std::vector<fastllm::MultiThreadPermuteSelfSwapOp*> ops;
                int cur = 0;
                for (int i = 0; i < threadNum; i++) {
                    int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
                    ops.push_back(new MultiThreadPermuteSelfSwapOp(finput, n, m, cur, end));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            }
            input.Resize(new_dims);
        } else if (axis == std::vector<int> {0, 2, 1, 3}) {
            if (vold.size() < input.GetBytes()) {
                vold.resize(input.GetBytes());
            }
            RunMultiThreadMemcpy(vold.data(), input.cpuData, input.GetBytes(), GetAlivePool());
            uint8_t *oldData = vold.data();
            uint8_t *newData = (uint8_t *) input.cpuData;
            int b = input.dims[0];
            int n = input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            int unitSize = input.unitSize;
            for (int o = 0; o < b; o++) {
                RunMultiThreadTransposeByLine(newData, oldData, n, m, k * unitSize, GetAlivePool());
                oldData += input.Count(1) * unitSize;
                newData += input.Count(1) * unitSize;
            }
            input.Resize(new_dims);
        } else if (axis == std::vector <int> {1, 0, 2}) {
            if (vold.size() < input.GetBytes()) {
                vold.resize(input.GetBytes());
            }
            RunMultiThreadMemcpy(vold.data(), input.cpuData, input.GetBytes(), GetAlivePool());
            uint8_t *oldData = vold.data();
            uint8_t *newData = (uint8_t *) input.cpuData;
            int n = input.dims[0];
            int m = input.dims[1];
            int k = input.dims[2];
            int unitSize = input.unitSize;
            RunMultiThreadTransposeByLine(newData, oldData, n, m, k * unitSize, GetAlivePool());
            input.Resize(new_dims);
        } else {
            auto tmp = new Data();
            fastllm::Permute(input, axis, *tmp);

            uint64_t copyBytes = (uint64_t)input.unitSize * input.Count(0);
            if (copyBytes < 2147483648ULL) {
                RunMultiThreadMemcpy(input.cpuData, tmp->cpuData, (int)copyBytes, GetAlivePool());
            } else {
                memcpy(input.cpuData, tmp->cpuData, copyBytes);
            }
            input.Resize(tmp->dims);
            delete tmp;
        }
    }

    void CpuPermuteSelfOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16 ||
                        input.dataType == DataType::BFLOAT16,
                        "Permute error: datatype should be float32, float16 or bfloat16.");
        AssertInFastLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");
        DoCpuPermuteSelf(input, axis);
    }

    void CpuRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 64;

        int len = data.dims[0], bs = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        int stride = (int)sinData.dims[1];
        for (int l = 0; l < len; l++) {
            for (int b = 0; b < bs; b++) {
                for (int part = 0; part < 2; part++) {
                    int index = (int) ((float *) positionIds.cpuData)[(b * 2 + part) * positionIds.dims.back() + l];
                    float *sin = ((float*)sinData.cpuData) + stride * index;
                    float *cos = ((float*)cosData.cpuData) + stride * index;
                    float *d = (float *) data.cpuData + (l * bs + b) * spatial + part * m / 2;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < rotaryDim && j < m / 4; j++) {
                            float a = d[j], b = d[j + m / 4];
                            d[j] = a * cos[j] - b * sin[j];
                            d[j + m / 4] = a * sin[j] + b * cos[j];
                        }

                        d += m;
                    }
                }
            }
        }
    }

    void CpuNearlyRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 64;
        int positionStride = intParams.find("positionStride") != intParams.end() ? intParams.find("positionStride")->second : 1;

        int len = data.dims[0], bs = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        int stride = (int)sinData.dims[1];
        positionStride *= positionIds.dims.back();
        for (int l = 0; l < len; l++) {
            for (int b = 0; b < bs; b++) {
                if (data.dataType == DataType::FLOAT32) {
                    int index = (int) ((float *) positionIds.cpuData)[b * positionStride + l];
                    float *sin = ((float*)sinData.cpuData) + stride * index;
                    float *cos = ((float*)cosData.cpuData) + stride * index;

                    float *d = (float *) data.cpuData + (l * bs + b) * spatial;
                    for (int i = 0; i < n; i++) {
                        int j = 0;
                        for (; j < rotaryDim; j += 2) {
                            float a = d[j], b = d[j + 1];
                            d[j] = a * cos[j / 2] - b * sin[j / 2];
                            d[j + 1] = a * sin[j / 2] + b * cos[j / 2];
                        }
                        d += m;
                    }
                } else if (data.dataType == DataType::FLOAT16) {
                    int index = (int) ((float *) positionIds.cpuData)[b * positionStride + l];
                    float *sin = ((float*)sinData.cpuData) + stride * index;
                    float *cos = ((float*)cosData.cpuData) + stride * index;

                    uint16_t *d = (uint16_t *) data.cpuData + (l * bs + b) * spatial;
                    for (int i = 0; i < n; i++) {
                        int j = 0;
                        for (; j < rotaryDim; j += 2) {
                            float a = fp16tofp32.dict[d[j]], b = fp16tofp32.dict[d[j + 1]];
                            d[j] = float_to_half(a * cos[j / 2] - b * sin[j / 2]);
                            d[j + 1] = float_to_half(a * sin[j / 2] + b * cos[j / 2]);
                        }
                        d += m;
                    }
                }
            }
        }
    }

    struct MultiThreadLlamaRotatePosition2DFloatOp : MultiThreadBaseOp {
        DataType dataType;
        float *data, *positionIds, *sinData, *cosData;
        int bs, len, n, m, stride, spatial, posDim, rotaryDim;      
        int st, end;

        MultiThreadLlamaRotatePosition2DFloatOp 
            (DataType dataType, float *data, float *positionIds, float *sinData, float *cosData, 
            int bs, int len, int n, int m, int stride, int spatial, int posDim, int rotaryDim, 
            int st, int end) : 
            dataType(dataType), data(data), positionIds(positionIds), sinData(sinData), cosData(cosData), 
            bs(bs), len(len), n(n), m(m), stride(stride), spatial(spatial), posDim(posDim), rotaryDim(rotaryDim), 
            st(st), end(end) {}

        void Run() {
            if (dataType == DataType::FLOAT32) {
                for (int idx = st; idx < end; idx++) {
                    int b = idx / len;
                    int l = idx % len;
                    int index = (int) ((float *) positionIds)[b * posDim + l];
                    float *sin = ((float *) sinData) + stride * index;
                    float *cos = ((float *) cosData) + stride * index;
                    float *d = (float *) data + (b * len + l) * spatial;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < rotaryDim && j < m / 2; j++) {
                            float a = d[j], b = d[j + m / 2];
                            d[j] = a * cos[j] - b * sin[j];
                            d[j + m / 2] = a * sin[j] + b * cos[j];
                        }
                        d += m;
                    }
                }
            } else {
                for (int idx = st; idx < end; idx++) {
                    int b = idx / len;
                    int l = idx % len;
                    int index = (int) ((float *) positionIds)[b * posDim + l];
                    float *sin = ((float *) sinData) + stride * index;
                    float *cos = ((float *) cosData) + stride * index;
                    uint16_t *d = (uint16_t *) data + (b * len + l) * spatial;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < rotaryDim && j < m / 2; j++) {
                            float a = fp16tofp32.dict[d[j]], b = fp16tofp32.dict[d[j + m / 2]];
                            d[j] = float_to_half(a * cos[j] - b * sin[j]);
                            d[j + m / 2] = float_to_half(a * sin[j] + b * cos[j]);
                        }
                        d += m;
                    }
                }
            }
        }
    };

    struct MultiThreadLlamaRotatePosition2DPartFloatOp : MultiThreadBaseOp {
        DataType dataType;
        float *data, *positionIds, *sinData, *cosData;
        int bs, len, n, m, stride, spatial, posDim, rotaryDim, part;      
        int st, end;

        MultiThreadLlamaRotatePosition2DPartFloatOp 
            (DataType dataType, float *data, float *positionIds, float *sinData, float *cosData, 
            int bs, int len, int n, int m, int stride, int spatial, int posDim, int rotaryDim, int part,
            int st, int end) : 
            dataType(dataType), data(data), positionIds(positionIds), sinData(sinData), cosData(cosData), 
            bs(bs), len(len), n(n), m(m), stride(stride), spatial(spatial), posDim(posDim), rotaryDim(rotaryDim), part(part),
            st(st), end(end) {}

        void Run() {
            if (dataType == DataType::FLOAT32) {
                for (int idx = st; idx < end; idx++) {
                    int b = idx / len;
                    int l = idx % len;
                    int index = (int) ((float *) positionIds)[b * posDim + l];
                    float *sin = ((float *) sinData) + stride * index;
                    float *cos = ((float *) cosData) + stride * index;
                    float *d = (float *) data + (b * len + l) * spatial;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < rotaryDim && j < m / 2 && j < part / 2; j++) {
                            float a = d[j], b = d[j + part / 2];
                            d[j] = a * cos[j] - b * sin[j];
                            d[j + part / 2] = a * sin[j] + b * cos[j];
                        }
                        d += m;
                    }
                }
            } else {
                for (int idx = st; idx < end; idx++) {
                    int b = idx / len;
                    int l = idx % len;
                    int index = (int) ((float *) positionIds)[b * posDim + l];
                    float *sin = ((float *) sinData) + stride * index;
                    float *cos = ((float *) cosData) + stride * index;
                    uint16_t *d = (uint16_t *) data + (b * len + l) * spatial;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < rotaryDim && j < m / 2 && j < part / 2; j++) {
                            float a = fp16tofp32.dict[d[j]], b = fp16tofp32.dict[d[j + part / 2]];
                            d[j] = float_to_half(a * cos[j] - b * sin[j]);
                            d[j + part / 2] = float_to_half(a * sin[j] + b * cos[j]);
                        }
                        d += m;
                    }
                }
            }
        }
    };

    static void RunMultiThreadLlamaRotatePosition2DFloat(DataType dataType, float *data, float *positionIds, float *sinData, float *cosData, 
            int bs, int len, int n, int m, int stride, int spatial, int posDim, int rotaryDim, AliveThreadPool *pool) {
        if (bs * len == 1) {
            (MultiThreadLlamaRotatePosition2DFloatOp(dataType, data, positionIds, sinData, cosData, bs, len, n, m, stride, spatial, posDim, rotaryDim, 0, bs * len)).Run();
            return;
        }

        int threadNum = pool->threads.size();
        int per = (bs * len) / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadLlamaRotatePosition2DFloatOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? (bs * len) : cur + per + (cur + per * (threadNum - i) < (bs * len)));
            ops.push_back(new MultiThreadLlamaRotatePosition2DFloatOp(
                dataType, data, positionIds, sinData, cosData, bs, len, n, m, stride, spatial, posDim, rotaryDim, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuLlamaRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;

        int bs = data.dims[0], len = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        int stride = (int)sinData.dims[1];
        RunMultiThreadLlamaRotatePosition2DFloat(data.dataType, (float*)data.cpuData, (float*)positionIds.cpuData, 
            (float*)sinData.cpuData, (float*)cosData.cpuData, bs, len, n, m, stride, spatial, 
            positionIds.dims.back(), rotaryDim, GetAlivePool());
    }

    static void RunMultiThreadLlamaRotatePosition2DPartFloat(DataType dataType, float *data, float *positionIds, float *sinData, float *cosData, 
            int bs, int len, int n, int m, int stride, int spatial, int posDim, int rotaryDim, int part, AliveThreadPool *pool) {
        if (bs * len == 1) {
            (MultiThreadLlamaRotatePosition2DPartFloatOp(dataType, data, positionIds, sinData, cosData, bs, len, n, m, stride, spatial, posDim, rotaryDim, part, 0, bs * len)).Run();
            return;
        }

        int threadNum = pool->threads.size();
        int per = (bs * len) / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadLlamaRotatePosition2DPartFloatOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? (bs * len) : cur + per + (cur + per * (threadNum - i) < (bs * len)));
            ops.push_back(new MultiThreadLlamaRotatePosition2DPartFloatOp(
                dataType, data, positionIds, sinData, cosData, bs, len, n, m, stride, spatial, posDim, rotaryDim, part, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuLlamaRotatePosition2DPartOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        int part = intParams.find("part") != intParams.end() ? intParams.find("part")->second : 128;

        int bs = data.dims[0], len = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        int stride = (int)sinData.dims[1];
        RunMultiThreadLlamaRotatePosition2DPartFloat(data.dataType, (float*)data.cpuData, (float*)positionIds.cpuData, 
            (float*)sinData.cpuData, (float*)cosData.cpuData, bs, len, n, m, stride, spatial, 
            positionIds.dims.back(), rotaryDim, part, GetAlivePool());
    }

    struct MultiThreadRopeEncodingFloatOp : MultiThreadBaseOp {
        DataType dataType;
        float *data, *positionIds;
        int bs, len, n, m, spatial, posDim, rotaryDim;
        float ropeTheta, ropeScale;
        int st, end;

        MultiThreadRopeEncodingFloatOp
            (DataType dataType, float *data, float *positionIds,
            int bs, int len, int n, int m, int spatial, int posDim, int rotaryDim,
            float ropeTheta, float ropeScale,
            int st, int end) :
            dataType(dataType), data(data), positionIds(positionIds),
            bs(bs), len(len), n(n), m(m), spatial(spatial), posDim(posDim), rotaryDim(rotaryDim),
            ropeTheta(ropeTheta), ropeScale(ropeScale),
            st(st), end(end) {}

        void Run() {
            int half = rotaryDim / 2;
            if (dataType == DataType::FLOAT32) {
                for (int idx = st; idx < end; idx++) {
                    int b = idx / len;
                    int l = idx % len;
                    int index = (int) ((float *) positionIds)[b * posDim + l];
                    float position = (float)index / ropeScale;
                    float *d = (float *) data + (b * len + l) * spatial;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < half; j++) {
                            float freq = position / pow(ropeTheta, (float)(2 * j) / rotaryDim);
                            float curSin = sin(freq);
                            float curCos = cos(freq);
                            float a = d[j], b = d[j + half];
                            d[j] = a * curCos - b * curSin;
                            d[j + half] = a * curSin + b * curCos;
                        }
                        d += m;
                    }
                }
            } else {
                for (int idx = st; idx < end; idx++) {
                    int b = idx / len;
                    int l = idx % len;
                    int index = (int) ((float *) positionIds)[b * posDim + l];
                    float position = (float)index / ropeScale;
                    uint16_t *d = (uint16_t *) data + (b * len + l) * spatial;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < half; j++) {
                            float freq = position / pow(ropeTheta, (float)(2 * j) / rotaryDim);
                            float curSin = sin(freq);
                            float curCos = cos(freq);
                            float a = fp16tofp32.dict[d[j]], b = fp16tofp32.dict[d[j + half]];
                            d[j] = float_to_half(a * curCos - b * curSin);
                            d[j + half] = float_to_half(a * curSin + b * curCos);
                        }
                        d += m;
                    }
                }
            }
        }
    };

    static void RunMultiThreadRopeEncodingFloat(DataType dataType, float *data, float *positionIds,
            int bs, int len, int n, int m, int spatial, int posDim, int rotaryDim,
            float ropeTheta, float ropeScale, AliveThreadPool *pool) {
        if (bs * len == 1) {
            (MultiThreadRopeEncodingFloatOp(dataType, data, positionIds, bs, len, n, m, spatial, posDim, rotaryDim, ropeTheta, ropeScale, 0, bs * len)).Run();
            return;
        }

        int threadNum = pool->threads.size();
        int per = (bs * len) / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadRopeEncodingFloatOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? (bs * len) : cur + per + (cur + per * (threadNum - i) < (bs * len)));
            ops.push_back(new MultiThreadRopeEncodingFloatOp(
                dataType, data, positionIds, bs, len, n, m, spatial, posDim, rotaryDim, ropeTheta, ropeScale, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuRopeEncodingOp::Run(const std::string &opType, const fastllm::DataDict &datas,
	                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        float ropeTheta = floatParams.find("ropeTheta") != floatParams.end() ? floatParams.find("ropeTheta")->second : 10000.0f;
        float ropeScale = floatParams.find("ropeScale") != floatParams.end() ? floatParams.find("ropeScale")->second : 1.0f;

        int bs = data.dims[0], len = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        RunMultiThreadRopeEncodingFloat(data.dataType, (float*)data.cpuData, (float*)positionIds.cpuData,
            bs, len, n, m, spatial,
            positionIds.dims.back(), rotaryDim, ropeTheta, ropeScale, GetAlivePool());
    }

    static inline float CpuLlama3InvFreq(float invFreq, float factor, float originalMaxPosition,
                                         float lowFreqFactor, float highFreqFactor) {
        float wavelen = 2.0f * (float)M_PI / invFreq;
        float lowWavelen = originalMaxPosition / lowFreqFactor;
        float highWavelen = originalMaxPosition / highFreqFactor;
        float invLlama = wavelen > lowWavelen ? invFreq / factor : invFreq;
        if (!(wavelen < highWavelen) && !(wavelen > lowWavelen)) {
            float smooth = (originalMaxPosition / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
            invLlama = (1.0f - smooth) * invFreq / factor + smooth * invFreq;
        }
        return invLlama;
    }

    static inline float CpuRopeRead(const Data &data, int index) {
        if (data.dataType == DataType::FLOAT32) {
            return ((float*)data.cpuData)[index];
        }
        uint16_t v = ((uint16_t*)data.cpuData)[index];
        return data.dataType == DataType::FLOAT16 ? fp16tofp32.dict[v] : bf16tofp32.dict[v];
    }

    static inline void CpuRopeWrite(Data &data, int index, float value) {
        if (data.dataType == DataType::FLOAT32) {
            ((float*)data.cpuData)[index] = value;
        } else if (data.dataType == DataType::FLOAT16) {
            ((uint16_t*)data.cpuData)[index] = float_to_half(value);
        } else {
            Float32ToBFloat16(&value, ((uint16_t*)data.cpuData) + index, 1);
        }
    }

    void CpuYarnRopeEncodingOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        float ropeTheta = floatParams.find("ropeTheta") != floatParams.end() ? floatParams.find("ropeTheta")->second : 10000.0f;
        float factor = floatParams.find("factor") != floatParams.end() ? floatParams.find("factor")->second : 1.0f;
        float attentionFactor = floatParams.find("attentionFactor") != floatParams.end() ? floatParams.find("attentionFactor")->second : 1.0f;
        float correctionLow = floatParams.find("correctionLow") != floatParams.end() ? floatParams.find("correctionLow")->second : 0.0f;
        float correctionHigh = floatParams.find("correctionHigh") != floatParams.end() ? floatParams.find("correctionHigh")->second : 1.0f;

        AssertInFastLLM(data.dims.size() == 4,
                        "YaRN RoPE expects [batch, seq, heads, dim] input.");
        AssertInFastLLM(positionIds.dataType == DataType::FLOAT32,
                        "YaRN RoPE expects FLOAT32 position ids.");
        AssertInFastLLM(data.dataType == DataType::FLOAT32 ||
                        data.dataType == DataType::FLOAT16 ||
                        data.dataType == DataType::BFLOAT16,
                        "YaRN RoPE supports FLOAT32, FLOAT16 and BFLOAT16 input.");
        AssertInFastLLM(rotaryDim <= data.dims[3],
                        "YaRN rotary_dim exceeds the input head dimension.");

        int bs = data.dims[0], len = data.dims[1], n = data.dims[2], m = data.dims[3];
        int spatial = data.Count(2), half = rotaryDim / 2, posStride = positionIds.dims.back();
        for (int batch = 0; batch < bs; batch++) {
            for (int token = 0; token < len; token++) {
                float position = (float)(int)((float*)positionIds.cpuData)[batch * posStride + token];
                int tokenOffset = (batch * len + token) * spatial;
                for (int j = 0; j < half; j++) {
                    float posFreq = powf(ropeTheta, (float)(2 * j) / rotaryDim);
                    float extrapolation = 1.0f / posFreq;
                    float interpolation = 1.0f / (factor * posFreq);
                    float ramp = std::max(0.0f, std::min(1.0f,
                        (j - correctionLow) / (correctionHigh - correctionLow)));
                    float extrapolationFactor = 1.0f - ramp;
                    float invFreq = interpolation * (1.0f - extrapolationFactor) +
                                    extrapolation * extrapolationFactor;
                    float angle = position * invFreq;
                    float curSin = sinf(angle) * attentionFactor;
                    float curCos = cosf(angle) * attentionFactor;
                    for (int h = 0; h < n; h++) {
                        int headOffset = tokenOffset + h * m;
                        float a = CpuRopeRead(data, headOffset + j);
                        float b = CpuRopeRead(data, headOffset + j + half);
                        CpuRopeWrite(data, headOffset + j, a * curCos - b * curSin);
                        CpuRopeWrite(data, headOffset + j + half, a * curSin + b * curCos);
                    }
                }
            }
        }
    }

    void CpuLlama3RopeEncodingOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        float ropeTheta = floatParams.find("ropeTheta") != floatParams.end() ? floatParams.find("ropeTheta")->second : 10000.0f;
        float factor = floatParams.find("factor") != floatParams.end() ? floatParams.find("factor")->second : 1.0f;
        float originalMaxPosition = floatParams.find("originalMaxPosition") != floatParams.end() ? floatParams.find("originalMaxPosition")->second : 131072.0f;
        float lowFreqFactor = floatParams.find("lowFreqFactor") != floatParams.end() ? floatParams.find("lowFreqFactor")->second : 1.0f;
        float highFreqFactor = floatParams.find("highFreqFactor") != floatParams.end() ? floatParams.find("highFreqFactor")->second : 32.0f;

        int bs = data.dims[0], len = data.dims[1], n = data.dims[2], m = data.dims[3];
        int spatial = data.Count(2), half = rotaryDim / 2, posStride = positionIds.dims.back();
        for (int b = 0; b < bs; b++) {
            for (int l = 0; l < len; l++) {
                float position = ((float*)positionIds.cpuData)[b * posStride + l];
                for (int h = 0; h < n; h++) {
                    int headOffset = (b * len + l) * spatial + h * m;
                    for (int j = 0; j < half; j++) {
                        float invFreq = 1.0f / powf(ropeTheta, (float)(2 * j) / rotaryDim);
                        invFreq = CpuLlama3InvFreq(invFreq, factor, originalMaxPosition, lowFreqFactor, highFreqFactor);
                        float freq = position * invFreq;
                        float curSin = sinf(freq), curCos = cosf(freq);
                        float a = CpuRopeRead(data, headOffset + j);
                        float bval = CpuRopeRead(data, headOffset + j + half);
                        CpuRopeWrite(data, headOffset + j, a * curCos - bval * curSin);
                        CpuRopeWrite(data, headOffset + j + half, a * curSin + bval * curCos);
                    }
                }
            }
        }
    }

    static inline int ResolveQwen35InterleavedMRopeIndex(int dim, int sectionH, int sectionW) {
        if (dim % 3 == 1 && dim < sectionH * 3) {
            return 1;
        }
        if (dim % 3 == 2 && dim < sectionW * 3) {
            return 2;
        }
        return 0;
    }

    struct MultiThreadQwen35InterleavedRopeFloatOp : MultiThreadBaseOp {
        DataType dataType;
        float *data, *positionIds;
        int bs, len, n, m, spatial, positionStride, rotaryDim, sectionH, sectionW;
        float ropeTheta, ropeScale;
        int st, end;

        MultiThreadQwen35InterleavedRopeFloatOp(
            DataType dataType, float *data, float *positionIds,
            int bs, int len, int n, int m, int spatial, int positionStride,
            int rotaryDim, int sectionH, int sectionW,
            float ropeTheta, float ropeScale,
            int st, int end) :
            dataType(dataType), data(data), positionIds(positionIds),
            bs(bs), len(len), n(n), m(m), spatial(spatial), positionStride(positionStride),
            rotaryDim(rotaryDim), sectionH(sectionH), sectionW(sectionW),
            ropeTheta(ropeTheta), ropeScale(ropeScale), st(st), end(end) {}

        void Run() {
            int half = rotaryDim / 2;
            if (dataType == DataType::FLOAT32) {
                for (int idx = st; idx < end; idx++) {
                    int b = idx / len;
                    int l = idx % len;
                    float *d = (float *) data + (b * len + l) * spatial;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < half; j++) {
                            int row = ResolveQwen35InterleavedMRopeIndex(j, sectionH, sectionW);
                            float position = positionIds[row * positionStride + l] / ropeScale;
                            float freq = position / pow(ropeTheta, (float)(2 * j) / rotaryDim);
                            float curSin = sin(freq);
                            float curCos = cos(freq);
                            float a = d[j], b = d[j + half];
                            d[j] = a * curCos - b * curSin;
                            d[j + half] = a * curSin + b * curCos;
                        }
                        d += m;
                    }
                }
            } else {
                for (int idx = st; idx < end; idx++) {
                    int b = idx / len;
                    int l = idx % len;
                    uint16_t *d = (uint16_t *) data + (b * len + l) * spatial;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < half; j++) {
                            int row = ResolveQwen35InterleavedMRopeIndex(j, sectionH, sectionW);
                            float position = positionIds[row * positionStride + l] / ropeScale;
                            float freq = position / pow(ropeTheta, (float)(2 * j) / rotaryDim);
                            float curSin = sin(freq);
                            float curCos = cos(freq);
                            float a = fp16tofp32.dict[d[j]], b = fp16tofp32.dict[d[j + half]];
                            d[j] = float_to_half(a * curCos - b * curSin);
                            d[j + half] = float_to_half(a * curSin + b * curCos);
                        }
                        d += m;
                    }
                }
            }
        }
    };

    static void RunMultiThreadQwen35InterleavedRopeFloat(
        DataType dataType, float *data, float *positionIds,
        int bs, int len, int n, int m, int spatial, int positionStride,
        int rotaryDim, int sectionH, int sectionW,
        float ropeTheta, float ropeScale, AliveThreadPool *pool) {
        if (bs * len == 1) {
            (MultiThreadQwen35InterleavedRopeFloatOp(
                dataType, data, positionIds, bs, len, n, m, spatial, positionStride,
                rotaryDim, sectionH, sectionW, ropeTheta, ropeScale, 0, bs * len)).Run();
            return;
        }

        int threadNum = pool->threads.size();
        int per = (bs * len) / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadQwen35InterleavedRopeFloatOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? (bs * len) : cur + per + (cur + per * (threadNum - i) < (bs * len)));
            ops.push_back(new MultiThreadQwen35InterleavedRopeFloatOp(
                dataType, data, positionIds, bs, len, n, m, spatial, positionStride,
                rotaryDim, sectionH, sectionW, ropeTheta, ropeScale, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuQwen35InterleavedRopeOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        int sectionT = intParams.find("sectionT") != intParams.end() ? intParams.find("sectionT")->second : 0;
        int sectionH = intParams.find("sectionH") != intParams.end() ? intParams.find("sectionH")->second : 0;
        int sectionW = intParams.find("sectionW") != intParams.end() ? intParams.find("sectionW")->second : 0;
        float ropeTheta = floatParams.find("ropeTheta") != floatParams.end() ? floatParams.find("ropeTheta")->second : 10000.0f;
        float ropeScale = floatParams.find("ropeScale") != floatParams.end() ? floatParams.find("ropeScale")->second : 1.0f;

        AssertInFastLLM(data.dims.size() == 4, "Qwen3.5 interleaved RoPE expects [batch, seq, heads, dim] input.");
        AssertInFastLLM(positionIds.dims.size() == 2 && positionIds.dims[0] == 3,
                        "Qwen3.5 interleaved RoPE expects position ids with shape [3, seq].");
        AssertInFastLLM(data.dims[0] == 1, "Qwen3.5 interleaved RoPE currently supports batch size 1 only.");
        AssertInFastLLM(sectionT + sectionH + sectionW == rotaryDim / 2,
                        "Qwen3.5 interleaved RoPE section sizes must sum to rotary_dim / 2.");

        int bs = data.dims[0], len = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        RunMultiThreadQwen35InterleavedRopeFloat(
            data.dataType, (float*) data.cpuData, (float*) positionIds.cpuData,
            bs, len, n, m, spatial, positionIds.dims.back(),
            rotaryDim, sectionH, sectionW, ropeTheta, ropeScale, GetAlivePool());
    }

    void CpuQKVRMSNormRopeOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        // CPU fallback: 使用原有的 RMSNorm + RopeEncoding 分步实现
        Data &qkv = *(datas.find("qkv")->second);
        Data &qNormWeight = *(datas.find("qNormWeight")->second);
        Data &kNormWeight = *(datas.find("kNormWeight")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        int q_heads = intParams.find("q_heads")->second;
        int k_heads = intParams.find("k_heads")->second;
        int head_dim = intParams.find("head_dim")->second;
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        float eps = floatParams.find("eps")->second;
        float ropeTheta = floatParams.find("ropeTheta") != floatParams.end() ? floatParams.find("ropeTheta")->second : 10000.0f;
        float ropeScale = floatParams.find("ropeScale") != floatParams.end() ? floatParams.find("ropeScale")->second : 1.0f;

        int qdim = q_heads * head_dim;
        int per = k_heads * head_dim;

        // 拆出 q, k，做 RMSNorm + RoPE，然后写回
        Data q, k;
        Split(qkv, -1, 0, qdim, q);
        Split(qkv, -1, qdim, qdim + per, k);

        q.Reshape({q.dims[0], q.dims[1], q_heads, head_dim});
        k.Reshape({k.dims[0], k.dims[1], k_heads, head_dim});

        RMSNorm(q, qNormWeight, eps, q);
        RopeEncoding(q, positionIds, rotaryDim, ropeTheta, ropeScale);

        RMSNorm(k, kNormWeight, eps, k);
        RopeEncoding(k, positionIds, rotaryDim, ropeTheta, ropeScale);

        // 将结果写回 qkv
        q.Reshape({q.dims[0], q.dims[1], qdim});
        k.Reshape({k.dims[0], k.dims[1], per});

        int bs = qkv.dims[0], seqlen = qkv.dims[1], total_dim = qkv.dims[2];
        int unitSize = qkv.unitSize;
        for (int b = 0; b < bs; b++) {
            for (int s = 0; s < seqlen; s++) {
                int offset = (b * seqlen + s) * total_dim;
                memcpy((uint8_t*)qkv.cpuData + offset * unitSize,
                       (uint8_t*)q.cpuData + (b * seqlen + s) * qdim * unitSize,
                       qdim * unitSize);
                memcpy((uint8_t*)qkv.cpuData + (offset + qdim) * unitSize,
                       (uint8_t*)k.cpuData + (b * seqlen + s) * per * unitSize,
                       per * unitSize);
            }
        }
    }

    void CpuQKVRMSNormRopeSplitAppendPagedCacheOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qkv = *(datas.find("qkv")->second);
        Data &qNormWeight = *(datas.find("qNormWeight")->second);
        Data &kNormWeight = *(datas.find("kNormWeight")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &qOutput = *(datas.find("qOutput")->second);
        Data &pagedKCacheData = *(datas.find("pagedKCacheData")->second);
        Data &pagedVCacheData = *(datas.find("pagedVCacheData")->second);
        Data &insertIndexs = *(datas.find("insertIndexs")->second);
        Data &insertPositions = *(datas.find("insertPositions")->second);

        int q_heads = intParams.find("q_heads")->second;
        int k_heads = intParams.find("k_heads")->second;
        int v_heads = k_heads;
        int head_dim = intParams.find("head_dim")->second;
        int rotateDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        int pageLen = intParams.find("pageLen")->second;
        int batch = intParams.find("batch")->second;
        float eps = floatParams.find("eps")->second;
        float ropeTheta = floatParams.find("ropeTheta") != floatParams.end() ? floatParams.find("ropeTheta")->second : 10000.0f;
        float ropeScale = floatParams.find("ropeScale") != floatParams.end() ? floatParams.find("ropeScale")->second : 1.0f;
        int doQKNorm = intParams.find("doQKNorm") != intParams.end() ? intParams.find("doQKNorm")->second : 1;

        int qdim = q_heads * head_dim;
        int kdim = k_heads * head_dim;
        int vdim = v_heads * head_dim;

        Data q, k, v;
        Split(qkv, -1, 0, qdim, q);
        Split(qkv, -1, qdim, qdim + kdim, k);
        Split(qkv, -1, qdim + kdim, qdim + kdim + vdim, v);

        q.Reshape({q.dims[0], q.dims[1], q_heads, head_dim});
        k.Reshape({k.dims[0], k.dims[1], k_heads, head_dim});

        if (doQKNorm) {
            RMSNorm(q, qNormWeight, eps, q);
            RMSNorm(k, kNormWeight, eps, k);
        }

        RopeEncoding(q, positionIds, rotateDim, ropeTheta, ropeScale);
        RopeEncoding(k, positionIds, rotateDim, ropeTheta, ropeScale);

        int bs = qkv.dims[0];
        int seqlen = qkv.dims[1];
        int unitSize = qkv.unitSize;

        qOutput.Allocate();

        for (int b = 0; b < bs; b++) {
            for (int s = 0; s < seqlen; s++) {
                for (int h = 0; h < q_heads; h++) {
                    uint8_t *dst = (uint8_t*)qOutput.cpuData +
                        ((b * q_heads + h) * seqlen + s) * head_dim * unitSize;
                    uint8_t *src = (uint8_t*)q.cpuData +
                        ((b * seqlen + s) * q_heads + h) * head_dim * unitSize;
                    memcpy(dst, src, head_dim * unitSize);
                }
            }
        }

        uint8_t *pagedKData = (uint8_t*)pagedKCacheData.cpuData;
        uint8_t *pagedVData = (uint8_t*)pagedVCacheData.cpuData;
        int32_t *idxData = (int32_t*)insertIndexs.cpuData;
        int32_t *posData = (int32_t*)insertPositions.cpuData;

        k.Reshape({k.dims[0] * k.dims[1], k_heads, head_dim});
        v.Reshape({v.dims[0] * v.dims[1], v_heads, head_dim});

        for (int b = 0; b < batch; b++) {
            int pageIdx = idxData[b];
            int pageOffset = posData[b];
            for (int h = 0; h < k_heads; h++) {
                uint8_t *dst = pagedKData +
                    ((size_t)pageIdx * pageLen * k_heads * head_dim + pageOffset * k_heads * head_dim + h * head_dim) * unitSize;
                uint8_t *src = (uint8_t*)k.cpuData +
                    (b * k_heads * head_dim + h * head_dim) * unitSize;
                memcpy(dst, src, head_dim * unitSize);
            }
            for (int h = 0; h < v_heads; h++) {
                uint8_t *dst = pagedVData +
                    ((size_t)pageIdx * pageLen * v_heads * head_dim + pageOffset * v_heads * head_dim + h * head_dim) * unitSize;
                uint8_t *src = (uint8_t*)v.cpuData +
                    (b * v_heads * head_dim + h * head_dim) * unitSize;
                memcpy(dst, src, head_dim * unitSize);
            }
        }
    }

    void CpuRepeatPenaltyOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &penalty = *(datas.find("penalty")->second);
        Data &penaltyScale = *(datas.find("penaltyScale")->second);
        AssertInFastLLM(input.dataType == DataType::FLOAT32 && penalty.dataType == DataType::FLOAT32 && penaltyScale.dataType == DataType::FLOAT32,
                        "Repeat Penalty error: Data's type should be float32.\n");
        float *inputData = (float*)input.cpuData;
        float *penaltyData = (float*)penalty.cpuData;
        float *penaltyScaleData = (float*)penaltyScale.cpuData;

        int batch = penalty.dims[0], tokens = penalty.dims[1];
        int vocabs = input.dims.back();
        for (int b = 0; b < batch; b++) {
            float scale = penaltyScaleData[b];
            for (int i = 0; i < tokens; i++) {
                int token = (int)(penaltyData[b * tokens + i] + 1e-6);
                if (token >= 0) {
                    int id = b * vocabs + token;
                    inputData[id] = inputData[id] < 0 ? inputData[id] * scale : inputData[id] / scale;
                }
            }
        }
    }

    void CpuApplyLognAttnOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &lognAttn = *(datas.find("lognAttn")->second);
        Data &positionIds = *(datas.find("positionIds")->second);

        float *inputData = (float *) input.cpuData;
        float *lognData = (float *) lognAttn.cpuData;

        int batch = input.dims[0];
        int seqLen = input.dims[1];
        int spatial = input.Count(2);
        int curPos = (int) ((float *) positionIds.cpuData) [0];
        for (int b = 0; b < batch; b++) {
            float *curInput = inputData + b * seqLen * spatial;
            for (int i = 0; i < seqLen; i++) {
                float logn = lognData[i + curPos];
                for (int s = 0; s < spatial; s++) {
                    curInput[s] *= logn;
                }
                curInput += spatial;
            }
        }
    }

    void CpuCumSumLastDimOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);

        float *inputData = (float *) input.cpuData;

        std::vector <float> floatInputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            inputData = (float*)floatInputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        int dim = input.dims.back();
        int outer = input.Count(0) / dim;
        for (int o = 0; o < outer; o++) {
            for (int j = 1; j < dim; j++) {
                inputData[o * dim + j] += inputData[o * dim + j - 1];
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(inputData, (uint16_t*)input.cpuData, (int)floatInputVector.size());
        }
    }

    void CpuMakeDecayMaskOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "CpuMakeDecayMaskOp's input's type should be float32 or float16.\n");

        std::vector <int> dims = input.dims;
        dims.push_back(dims.back());
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    struct MultiThreadMakeDecayMaskOp : MultiThreadBaseOp {
        float *inputData, *outputData;
        int dim, st, end;

        MultiThreadMakeDecayMaskOp (float *inputData, float *outputData, int dim, int st, int end) :
            inputData(inputData), outputData(outputData), dim(dim), st(st), end(end) {}

        void Run() {
            // decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
            for (int o = st; o < end; o++) {
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j <= i && j < dim; j++) {
                        outputData[(long long)o * dim * dim + i * dim + j] =
                            std::exp(inputData[(long long)o * dim + i] - inputData[(long long)o * dim + j]);
                    }
                    for (int j = i + 1; j < dim; j++) {
                        outputData[(long long)o * dim * dim + i * dim + j] = 0.0f;
                    }
                }
            }
        }
    };

    void CpuMakeDecayMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        int dim = input.dims.back();
        int outer = input.Count(0) / dim;

        float *inputData = (float *) input.cpuData;
        float *outputData = (float *) output.cpuData;

        std::vector <float> floatInputVector, floatOutputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            floatOutputVector.resize(output.Count(0));
            inputData = (float*)floatInputVector.data();
            outputData = (float*)floatOutputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        if ((long long)outer * dim * dim < 65536 || outer <= 1) {
            MultiThreadMakeDecayMaskOp(inputData, outputData, dim, 0, outer).Run();
        } else {
            auto *pool = GetAlivePool();
            int threadNum = std::min((int)pool->threads.size(), outer);
            int per = outer / threadNum;
            std::vector<fastllm::MultiThreadMakeDecayMaskOp*> ops;
            int cur = 0;
            for (int i = 0; i < threadNum; i++) {
                int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
                ops.push_back(new MultiThreadMakeDecayMaskOp(inputData, outputData, dim, cur, end));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(outputData, (uint16_t*)output.cpuData, (int)floatOutputVector.size());
        }
    }

    void SiluMultiThread(float *input, int len, float *output,
                         int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSiluOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadSiluOp(input + cur, end - cur, output + cur,
                                                         n, inputStride, outputStride));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void GeluMultiThread(float *input, int len, float *output,
                         int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadGeluOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadGeluOp(input + cur, end - cur, output + cur,
                                                           n, inputStride, outputStride));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void GegluMultiThread(float *input, int mid, int len, float *output,
                          int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadGegluOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadGegluOp(input + cur, mid, end - cur, output + cur,
                                                          n, inputStride, outputStride));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void SwigluGptOssMultiThread(float *input, int mid, int len, float *output,
                           int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSwigluGptOssOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadSwigluGptOssOp(input + cur, mid, end - cur, output + cur,
                                                           n, inputStride, outputStride));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void SwigluMultiThread(float *input, int mid, int len, float *output,
                           int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSwigluOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadSwigluOp(input + cur, mid, end - cur, output + cur,
                                                           n, inputStride, outputStride));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CrossSwigluMultiThread(float *input, int mid, int len, float *output,
                           int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadCrossSwigluOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadCrossSwigluOp(input + cur * 2, mid, end - cur, output + cur,
                                                           n, inputStride, outputStride));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void SwigluMultiThreadFloat16(uint16_t *input, int mid, int len, uint16_t *output,
                           int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSwigluFloat16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadSwigluFloat16Op(input + cur, mid, end - cur, output + cur,
                                                           n, inputStride, outputStride));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void GegluMultiThreadFloat16(uint16_t *input, int mid, int len, uint16_t *output,
                           int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadGegluFloat16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadGegluFloat16Op(input + cur, mid, end - cur, output + cur,
                                                                 n, inputStride, outputStride));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void SwigluMultiThreadBFloat16(uint16_t *input, int mid, int len, uint16_t *output,
                           int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSwigluBFloat16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadSwigluBFloat16Op(input + cur, mid, end - cur, output + cur,
                                                           n, inputStride, outputStride));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void GegluMultiThreadBFloat16(uint16_t *input, int mid, int len, uint16_t *output,
                           int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadGegluBFloat16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadGegluBFloat16Op(input + cur, mid, end - cur, output + cur,
                                                                  n, inputStride, outputStride));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void SoftmaxMultiThread(float *input, int n, int m, int lastlen, AliveThreadPool *pool) {
        if (n == 1) {
            (MultiThreadSoftmaxOp(input, n, m, lastlen)).Run();
            return;
        }
        int threadNum = pool->threads.size();
        int per = n / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSoftmaxOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
            ops.push_back(new fastllm::MultiThreadSoftmaxOp(input + cur * m, end - cur, m, lastlen + cur));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void MultiThreadSoftmaxOp::Run() {
        for (int i = 0; i < n; i++) {
            float maxValue = -1e100;
            for (int j = 0; j < m; j++) {
                if (lastlen + i < j) {
                    value[i * m + j] = -10000;
                }
                maxValue = std::max(maxValue, value[i * m + j]);
            }
            float sum = 0.0;
            for (int j = 0; j < m; j++) {
                value[i * m + j] = expf(value[i * m + j] - maxValue);
                sum += value[i * m + j];
            }
            for (int j = 0; j < m; j++) {
                value[i * m + j] /= sum;
            }
        }
    }

    void MultiThreadSiluOp::Run() {
        for (int o = 0; o < n; o++) {
            float *cur = (float *) input + o * inputStride;
            float *out = (float *) output + o * outputStride;

            int i = 0;
#ifdef __aarch64__
            float32x4_t c1 = vdupq_n_f32(1.0f);
            for (; i + 3 < len; i += 4) {
                float32x4_t vx = vld1q_f32(cur + i);
                float32x4_t vdiv = vaddq_f32(c1, exp_ps(vnegq_f32(vx)));
                vx = vdivq_f32(vx, vdiv);
                vst1q_f32(out + i, vx);
            }
#endif
            for (; i < len; i++) {
                float x = cur[i];
                out[i] = x / (1.0 + expf(-x));
            }
        }
    }

    void MultiThreadGeluOp::Run() {
        for (int o = 0; o < n; o++) {
            float *cur = (float *) input + o * inputStride;
            float *out = (float *) output + o * outputStride;
            for (int i = 0; i < len; i++) {
                out[i] = gelu(cur[i]);
            }
        }
    }

    void MultiThreadSigmoidOp::Run() {
        int i = 0;
#ifdef __AVX2__
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 negOne = _mm256_set1_ps(-1.0f);
        const __m256 lo = _mm256_set1_ps(-87.0f);
        const __m256 hi = _mm256_set1_ps(87.0f);
        const __m256 loOut = _mm256_set1_ps(0.0f);
        const __m256 hiOut = _mm256_set1_ps(1.0f);
        for (; i + 7 < len; i += 8) {
            __m256 x = _mm256_loadu_ps(input + i);
            __m256 clipped = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
            __m256 e = exp256_ps(_mm256_mul_ps(negOne, clipped));
            __m256 r = _mm256_div_ps(one, _mm256_add_ps(one, e));
            __m256 below = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);
            __m256 above = _mm256_cmp_ps(x, hi, _CMP_GT_OQ);
            r = _mm256_blendv_ps(r, loOut, below);
            r = _mm256_blendv_ps(r, hiOut, above);
            _mm256_storeu_ps(output + i, r);
        }
#endif
        for (; i < len; i++) {
            float x = input[i];
            output[i] = 1.0f / (1.0f + std::exp(-x));
        }
    }
    
    void MultiThreadSingleAttentionCausalOp::Run() {
        float *qk = new float[klen];
        float *temp = new float[klen];
        for (int i = 0; i < qlen; i++) {
            float maxValue = -10000, sum = 0.0;
            for (int j = 0; j < klen; j++) {
                if (lastlen + i < j) {
                    qk[j] = -10000;
                    continue;
                }

                float now = 0.0f;
                int l = 0;
#ifdef __aarch64__
                float32x4_t sum = {0, 0, 0, 0};
                for (; l + 3 < qdim; l += 4) {
                    sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(qd + i * qdim + l),
                                                   vld1q_f32(kd + j * qdim + l)));
                }
                now += sum[0] + sum[1] + sum[2] + sum[3];
#endif
                for (; l < qdim; l++) {
                    now += qd[i * qdim + l] * kd[j * qdim + l];
                }
                qk[j] = now * scale;
                maxValue = std::max(maxValue, now * scale);
            }

            int j = 0;
#ifdef __aarch64__
            float32x4_t vmax = vdupq_n_f32(maxValue);
            for (; j + 3 < klen; j += 4) {
                vst1q_f32(temp + j, exp_ps(vsubq_f32(vld1q_f32(qk + j), vmax)));
            }
#endif
            for (; j < klen; j++) {
                temp[j] = expf(qk[j] - maxValue);
            }

            sum = 0.0f;
            for (j = 0; j < klen; j++) {
                sum += temp[j];
            }
            sum = std::max(sum, 0.1f);
            for (j = 0; j < klen; j++) {
                qk[j] = temp[j] / sum;
            }
            for (j = 0; j < klen; j++) {
                for (int l = 0; l < vdim; l++) {
                    od[i * vdim + l] += qk[j] * vd[j * vdim + l];
                }
            }
        }
        delete[] qk;
        delete[] temp;
    }

    void CpuAppendPagedCacheOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &cache = *(datas.find("cache")->second);
        Data &input = *(datas.find("input")->second);
        PagedCacheManager &pagedCacheManager = *((PagedCacheManager*)datas.find("pagedCacheManager")->second);

        AssertInFastLLM(cache.dataType == DataType::FLOAT32 ||
                        cache.dataType == DataType::FLOAT16 ||
                        cache.dataType == DataType::BFLOAT16, 
                        "CpuAppendPagedCacheOp's cache's type should be float32, float16 or bfloat16.\n");
        
        AssertInFastLLM(input.dims.size() == 3, 
                        "CpuAppendPagedCacheOp's input should have 3 dimensions [numHeads, seqLen, headDim].\n");
        
        AssertInFastLLM(cache.dataType != DataType::FP8_E4M3,
                        "CpuAppendPagedCacheOp doesn't support fp8_e4m3 KV cache.\n");
        AssertInFastLLM(input.dataType == cache.dataType,
                        "CpuAppendPagedCacheOp's input and cache should have the same data type.\n");
        
        cache.isPagedKVCache = true;
        
        // 获取输入的形状信息
        int numHeads = input.dims[0];
        int seqLen = input.dims[1];
        int headDim = input.dims[2];
        
        if (cache.pagedKVCacheData == nullptr) {
            cache.pagedKVCacheData = &pagedCacheManager;
        }
        cache.pageLen = pagedCacheManager.pageLen;
            
        // 检查 pagedKVCacheData 的形状是否匹配
        AssertInFastLLM(cache.pagedKVCacheData->dims.size() == 4,
                            "CpuAppendPagedCacheOp's pagedKVCacheData should have 4 dimensions.\n");
        AssertInFastLLM(cache.pagedKVCacheData->dims[1] == cache.pageLen,
                            "CpuAppendPagedCacheOp's pagedKVCacheData pageLen mismatch.\n");
        AssertInFastLLM(cache.pagedKVCacheData->dims[2] == numHeads,
                            "CpuAppendPagedCacheOp's pagedKVCacheData numHeads mismatch.\n");
        AssertInFastLLM(cache.pagedKVCacheData->dims[3] == headDim,
                            "CpuAppendPagedCacheOp's pagedKVCacheData headDim mismatch.\n");
        
        // 计算需要多少个新的 pages
        int currentUsedTokens = 0;
        if (cache.pageIndex.size() > 0) {
            currentUsedTokens = (cache.pageIndex.size() - 1) * cache.pageLen + cache.lastPageLen;
        }
        int totalNeededTokens = currentUsedTokens + seqLen;
        int totalNeededPages = (totalNeededTokens + cache.pageLen - 1) / cache.pageLen;
        int currentPages = (int)cache.pageIndex.size();
        int newPagesNeeded = totalNeededPages - currentPages;
        
        // 检查是否有足够的 pages
        int maxPages = cache.pagedKVCacheData->dims[0];
        if (totalNeededPages > maxPages) {
            ErrorInFastLLM("CpuAppendPagedCacheOp: No more pages available. Need to resize pagedKVCacheData. "
                           "seqLen = " + std::to_string(seqLen) +
                           ", currentUsedTokens = " + std::to_string(currentUsedTokens) +
                           ", pageLen = " + std::to_string(cache.pageLen) +
                           ", currentPages = " + std::to_string((int)cache.pageIndex.size()) +
                           ", totalNeededPages = " + std::to_string(totalNeededPages) +
                           ", maxPages = " + std::to_string(maxPages) + ".\n");
        }

        if (cache.dims.size() == 0) {
            cache.Resize(input.dims);
        } else {
            cache.Resize({cache.dims[0], cache.dims[1] + input.dims[1], cache.dims[2]});
        }
    }

    void CpuAppendPagedCacheOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &cache = *(datas.find("cache")->second);
        Data &input = *(datas.find("input")->second);
        
        // 获取输入的形状信息
        int numHeads = input.dims[0];
        int seqLen = input.dims[1];
        int headDim = input.dims[2];
        
        // 计算需要追加的 token 数量
        int remainingInCurrentPage = cache.pageLen - cache.lastPageLen;
        int tokensToAppend = seqLen;
        int inputOffset = 0;
        
        // 获取 pagedKVCacheData 的数据指针
        uint8_t *pagedData = cache.pagedKVCacheData->cpuData;
        uint8_t *inputData = input.cpuData;
        int unitSize = cache.unitSize;

        // {
        //     static std::once_flag debugOnce;
        //     std::call_once(debugOnce, []() {
        //         printf("[fastllm-debug-append] CpuAppendPagedCacheOp::Run (new checks active)\n");
        //         fflush(stdout);
        //     });
        // }
        // printf("[fastllm-debug-append] numHeads=%d seqLen=%d headDim=%d unitSize=%d pageLen=%d lastPageLen=%d maxPages=%d pagedData=%p\n",
        //        numHeads, seqLen, headDim, unitSize, cache.pageLen, cache.lastPageLen,
        //        (int)cache.pagedKVCacheData->dims[0], (void*)pagedData);
        // fflush(stdout);

        if (cache.pagedKVCacheData == nullptr || pagedData == nullptr) {
            ErrorInFastLLM("CpuAppendPagedCacheOp: pagedKVCacheData cpuData is null.\n");
        }
        if (cache.pagedKVCacheData->dataType != cache.dataType) {
            ErrorInFastLLM("CpuAppendPagedCacheOp: dataType mismatch. pagedKVCacheData = " +
                           GetDataTypeName(cache.pagedKVCacheData->dataType) +
                           ", cache = " + GetDataTypeName(cache.dataType) + ".\n");
        }
        if ((int)cache.pagedKVCacheData->dims.size() != 4 ||
            cache.pagedKVCacheData->dims[1] != cache.pageLen ||
            cache.pagedKVCacheData->dims[2] != numHeads ||
            cache.pagedKVCacheData->dims[3] != headDim) {
            ErrorInFastLLM("CpuAppendPagedCacheOp: dims mismatch. pagedKVCacheData dims = [" +
                           std::to_string(cache.pagedKVCacheData->dims.size() > 0 ? cache.pagedKVCacheData->dims[0] : -1) + ", " +
                           std::to_string(cache.pagedKVCacheData->dims.size() > 1 ? cache.pagedKVCacheData->dims[1] : -1) + ", " +
                           std::to_string(cache.pagedKVCacheData->dims.size() > 2 ? cache.pagedKVCacheData->dims[2] : -1) + ", " +
                           std::to_string(cache.pagedKVCacheData->dims.size() > 3 ? cache.pagedKVCacheData->dims[3] : -1) +
                           "], cache.pageLen = " + std::to_string(cache.pageLen) +
                           ", input dims = [" + std::to_string(numHeads) + ", " +
                           std::to_string(seqLen) + ", " + std::to_string(headDim) + "].\n");
        }
        
        // 先填充当前 page 的剩余空间
        if (cache.pageIndex.size() > 0 && remainingInCurrentPage > 0) {
            int currentPageIdx = cache.pageIndex.back();
            int copyLen = std::min(remainingInCurrentPage, tokensToAppend);
            
            // 复制数据：对于每个 token，每个 head，复制 headDim 个元素
            // pagedKVCacheData shape: [maxPages, pageLen, numHeads, headDim]
            // input shape: [numHeads, seqLen, headDim]
            for (int t = 0; t < copyLen; t++) {
                for (int h = 0; h < numHeads; h++) {
                    uint8_t *dst = pagedData + 
                        ((size_t)currentPageIdx * cache.pageLen * numHeads * headDim +
                         (cache.lastPageLen + t) * numHeads * headDim +
                         h * headDim) * unitSize;
                    uint8_t *src = inputData + 
                        (h * seqLen * headDim + (inputOffset + t) * headDim) * unitSize;
                    memcpy(dst, src, headDim * unitSize);
                }
            }
            
            cache.lastPageLen += copyLen;
            tokensToAppend -= copyLen;
            inputOffset += copyLen;
        }

        // 如果还有剩余的数据，需要分配新的 pages
        while (tokensToAppend > 0) {
            // 分配新的 page
            int newPageIdx = cache.pagedKVCacheData->GetUnusedPageIndex(true);

            if (newPageIdx < 0 || newPageIdx >= cache.pagedKVCacheData->dims[0]) {
                ErrorInFastLLM("CpuAppendPagedCacheOp: newPageIdx out of range. newPageIdx = " +
                               std::to_string(newPageIdx) + ", maxPages = " +
                               std::to_string(cache.pagedKVCacheData->dims[0]) + ".\n");
            }

            // printf("[fastllm-debug-append] newPageIdx=%d tokensToAppend=%d offsetBytes=%lld totalBytes=%lld\n",
            //        newPageIdx, tokensToAppend,
            //        (long long)(newPageIdx * cache.pageLen * numHeads * headDim) * unitSize,
            //        (long long)(cache.pagedKVCacheData->dims[0]) * cache.pageLen * numHeads * headDim * unitSize);
            // fflush(stdout);

            cache.pageIndex.push_back(newPageIdx);
            
            // 计算这个 page 可以存储多少 token
            int copyLen = std::min(cache.pageLen, tokensToAppend);
            
            // 复制数据：对于每个 token，每个 head，复制 headDim 个元素
            // pagedKVCacheData shape: [maxPages, pageLen, numHeads, headDim]
            // input shape: [numHeads, seqLen, headDim]
            for (int t = 0; t < copyLen; t++) {
                for (int h = 0; h < numHeads; h++) {
                    uint8_t *dst = pagedData + 
                        ((size_t)newPageIdx * cache.pageLen * numHeads * headDim +
                         t * numHeads * headDim +
                         h * headDim) * unitSize;
                    uint8_t *src = inputData + 
                        (h * seqLen * headDim + (inputOffset + t) * headDim) * unitSize;
                    memcpy(dst, src, headDim * unitSize);
                }
            }
            
            cache.lastPageLen = copyLen;
            tokensToAppend -= copyLen;
            inputOffset += copyLen;
        }
    }

    struct MultiThreadPagedAttentionFloat32Op : MultiThreadBaseOp {
        float *qHead, *oHead, *maskHead;
        float scale;
        int q1, q2, k1, v2, group, o;
        int qLo, qHi; // 本任务负责的 query 行区间（绝对行号）
        int pageLen, kNumHeads, kHeadDim, kUnitSize;
        int vPageLen, vNumHeads, vHeadDim, vUnitSize;
        uint8_t *kPagedData, *vPagedData;
        const std::vector<int> *kPageIndex, *vPageIndex;
        int kLastPageLen, vLastPageLen;
        const float *kF32Base, *vF32Base; // 预转换的连续 fp32 K/V，布局 [head][token][dim]

        MultiThreadPagedAttentionFloat32Op(
            float *qHead, float *oHead, float *maskHead, float scale,
            int q1, int q2, int k1, int v2, int group, int o,
            int qLo, int qHi,
            int pageLen, int kNumHeads, int kHeadDim, int kUnitSize,
            int vPageLen, int vNumHeads, int vHeadDim, int vUnitSize,
            uint8_t *kPagedData, uint8_t *vPagedData,
            const std::vector<int> *kPageIndex, const std::vector<int> *vPageIndex,
            int kLastPageLen, int vLastPageLen,
            const float *kF32Base, const float *vF32Base) :
            qHead(qHead), oHead(oHead), maskHead(maskHead), scale(scale),
            q1(q1), q2(q2), k1(k1), v2(v2), group(group), o(o),
            qLo(qLo), qHi(qHi),
            pageLen(pageLen), kNumHeads(kNumHeads), kHeadDim(kHeadDim), kUnitSize(kUnitSize),
            vPageLen(vPageLen), vNumHeads(vNumHeads), vHeadDim(vHeadDim), vUnitSize(vUnitSize),
            kPagedData(kPagedData), vPagedData(vPagedData),
            kPageIndex(kPageIndex), vPageIndex(vPageIndex),
            kLastPageLen(kLastPageLen), vLastPageLen(vLastPageLen),
            kF32Base(kF32Base), vF32Base(vF32Base) {}

        void Run() {
            const int kvHeadIdx = o / group;
            const float *kF32 = kF32Base + (size_t)kvHeadIdx * k1 * kHeadDim;
            const float *vF32 = vF32Base + (size_t)kvHeadIdx * k1 * vHeadDim;

            const int rows = qHi - qLo;
            const int base = k1 - q1;

            // online softmax（flash）：K/V 按 kvBlock 分块，scores 块缓冲
            // 常驻 L2，不再物化 [rows × k1] 的中间分数矩阵。
            // 临时 buffer 提为线程局部并复用 capacity：本类 task 数量可达
            // 数百个（heads × query 块），每 task 重新 malloc 百 KB 会高频
            // 触发 mmap/munmap + TLB shootdown，成为 prefill 长序列的瓶颈。
            const int kvBlock = 128;
            static thread_local std::vector<float> scratchAcc;
            static thread_local std::vector<float> scratchMVal;
            static thread_local std::vector<float> scratchLVal;
            static thread_local std::vector<float> scratchS;
            scratchAcc.assign((size_t)rows * v2, 0.0f);
            scratchMVal.assign((size_t)rows, -1e30f);
            scratchLVal.assign((size_t)rows, 0.0f);
            scratchS.assign((size_t)rows * kvBlock, 0.0f);
            float *acc = scratchAcc.data();
            float *mVal = scratchMVal.data();
            float *lVal = scratchLVal.data();
            float *s = scratchS.data();

            for (int jb = 0; jb < k1; jb += kvBlock) {
                const int jEnd = std::min(jb + kvBlock, k1);
                const int blockLen = jEnd - jb;

                // 整块对本 op 的所有行都不可见（因果）→ 直接跳过
                if (!maskHead && jb > base + qHi - 1) {
                    continue;
                }
                // 本块对所有行完全可见 → 免逐元素遮罩判断
                const bool allVisible = !maskHead && (jEnd - 1) <= base + qLo;

                // A: 本块 scores
                for (int j = jb; j < jEnd; j++) {
                    const float *kToken = kF32 + (size_t)j * kHeadDim;
                    for (int t = 0; t < rows; t++) {
                        const int i = qLo + t;
                        float *dst = s + (size_t)t * blockLen + (j - jb);
                        if (!allVisible) {
                            if (maskHead) {
                                if (maskHead[i * k1 + j] > 0.99) {
                                    *dst = -10000.0f;
                                    continue;
                                }
                            } else if (j > base + i) {
                                *dst = -10000.0f;
                                continue;
                            }
                        }

                        const float *qRow = qHead + (size_t)i * q2;
                        float dotProduct = 0.0f;
                        int l = 0;
#ifdef __aarch64__
                        float32x4_t sum = {0, 0, 0, 0};
                        for (; l + 3 < q2; l += 4) {
                            sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(qRow + l),
                                                            vld1q_f32(kToken + l)));
                        }
                        dotProduct += sum[0] + sum[1] + sum[2] + sum[3];
#elif defined(__AVX__)
                        __m256 vsum0 = _mm256_setzero_ps();
                        __m256 vsum1 = _mm256_setzero_ps();
                        __m256 vsum2 = _mm256_setzero_ps();
                        __m256 vsum3 = _mm256_setzero_ps();
                        for (; l + 31 < q2; l += 32) {
                            vsum0 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l)),
                                                    _mm256_loadu_ps((const float *) (kToken + l)), vsum0);
                            vsum1 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 8)),
                                                    _mm256_loadu_ps((const float *) (kToken + l + 8)), vsum1);
                            vsum2 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 16)),
                                                    _mm256_loadu_ps((const float *) (kToken + l + 16)), vsum2);
                            vsum3 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 24)),
                                                    _mm256_loadu_ps((const float *) (kToken + l + 24)), vsum3);
                        }
                        __m256 vsum = _mm256_add_ps(_mm256_add_ps(vsum0, vsum1),
                                                    _mm256_add_ps(vsum2, vsum3));
                        for (; l + 7 < q2; l += 8) {
                            vsum = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l)),
                                                   _mm256_loadu_ps((const float *) (kToken + l)), vsum);
                        }
                        dotProduct += Floatsum(vsum);
#endif
                        for (; l < q2; l++) {
                            dotProduct += qRow[l] * kToken[l];
                        }
                        *dst = dotProduct * scale;
                    }
                }

                // B: online softmax 状态更新（行最大值 / 归一化因子 / acc 重标定）
                for (int t = 0; t < rows; t++) {
                    float *sRow = s + (size_t)t * blockLen;
                    float blockMax = -1e30f;
                    for (int jj = 0; jj < blockLen; jj++) {
                        blockMax = std::max(blockMax, sRow[jj]);
                    }
                    float mNew = std::max(mVal[t], blockMax);
                    if (mNew <= -1e29f) {
                        continue; // 到目前为止整行仍无可见 token
                    }
                    float correction = (mVal[t] <= -1e29f) ? 0.0f : expf(mVal[t] - mNew);
                    float lt = 0.0f;
                    for (int jj = 0; jj < blockLen; jj++) {
                        float e = expf(sRow[jj] - mNew);
                        sRow[jj] = e;
                        lt += e;
                    }
                    lVal[t] = lVal[t] * correction + lt;
                    float *accRow = acc + (size_t)t * v2;
                    if (correction != 1.0f) {
                        int d = 0;
#ifdef __aarch64__
                        float32x4_t cv = vdupq_n_f32(correction);
                        for (; d + 3 < v2; d += 4) {
                            vst1q_f32(accRow + d, vmulq_f32(cv, vld1q_f32(accRow + d)));
                        }
#elif defined(__AVX__)
                        __m256 cv = _mm256_set1_ps(correction);
                        for (; d + 7 < v2; d += 8) {
                            _mm256_storeu_ps(accRow + d, _mm256_mul_ps(cv, _mm256_loadu_ps(accRow + d)));
                        }
#endif
                        for (; d < v2; d++) {
                            accRow[d] *= correction;
                        }
                    }
                    mVal[t] = mNew;
                }

                // C: V 块累加（s 已是本块的未归一化概率）
                for (int j = jb; j < jEnd; j++) {
                    const float *vToken = vF32 + (size_t)j * vHeadDim;
                    for (int t = 0; t < rows; t++) {
                        float w = s[(size_t)t * blockLen + (j - jb)];
                        if (w == 0.0f) {
                            continue;
                        }
                        float *oRow = acc + (size_t)t * v2;
                        int l = 0;
#ifdef __aarch64__
                        float32x4_t wv = vdupq_n_f32(w);
                        for (; l + 3 < v2; l += 4) {
                            vst1q_f32(oRow + l,
                                      vaddq_f32(vld1q_f32(oRow + l),
                                                vmulq_f32(wv, vld1q_f32(vToken + l))));
                        }
#elif defined(__AVX__)
                        __m256 wv = _mm256_set1_ps(w);
                        for (; l + 31 < v2; l += 32) {
                            _mm256_storeu_ps(oRow + l,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l),
                                                             _mm256_loadu_ps(oRow + l)));
                            _mm256_storeu_ps(oRow + l + 8,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 8),
                                                             _mm256_loadu_ps(oRow + l + 8)));
                            _mm256_storeu_ps(oRow + l + 16,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 16),
                                                             _mm256_loadu_ps(oRow + l + 16)));
                            _mm256_storeu_ps(oRow + l + 24,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 24),
                                                             _mm256_loadu_ps(oRow + l + 24)));
                        }
                        for (; l + 7 < v2; l += 8) {
                            _mm256_storeu_ps(oRow + l,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l),
                                                             _mm256_loadu_ps(oRow + l)));
                        }
#endif
                        for (; l < v2; l++) {
                            oRow[l] += w * vToken[l];
                        }
                    }
                }
            }

            // 最终归一化并写出
            for (int t = 0; t < rows; t++) {
                float denom = std::max(lVal[t], 0.1f);
                float inv = 1.0f / denom;
                const float *accRow = acc + (size_t)t * v2;
                float *oRow = oHead + (size_t)(qLo + t) * v2;
                int d = 0;
#ifdef __aarch64__
                float32x4_t iv = vdupq_n_f32(inv);
                for (; d + 3 < v2; d += 4) {
                    vst1q_f32(oRow + d, vmulq_f32(iv, vld1q_f32(accRow + d)));
                }
#elif defined(__AVX__)
                __m256 iv = _mm256_set1_ps(inv);
                for (; d + 7 < v2; d += 8) {
                    _mm256_storeu_ps(oRow + d, _mm256_mul_ps(iv, _mm256_loadu_ps(accRow + d)));
                }
#endif
                for (; d < v2; d++) {
                    oRow[d] = accRow[d] * inv;
                }
            }
        }
    };

    // GQA 融合版：一个 task 负责 (同一 kv 头的多个连续 q 头) × 一个 query 行块。
    // K/V 块在外层只流经内存一次，被组内多个 q 头复用。group > 1 时，
    // 旧实现每个 q 头独立流式读一遍完整 K/V（12MB/头），group=8 即 8× 冗余
    // DRAM 流量。这里把 kv block 提到最外层，8 个 q 头轮流消费同一块，K/V
    // 流量降为 1/group。
    struct MultiThreadPagedAttentionGqaFloat32Op : MultiThreadBaseOp {
        float *qHeadBase, *oHeadBase, *maskHeadBase;
        int qStride, oStride, maskStride;
        int maskHeadDenom; // q0 / batch：该 batch 有多少 q 头共用同一 mask
        float scale;
        int q1, q2, k1, v2, group;
        int kHeadDim, vHeadDim;
        int qHeadLo, qHeadHi; // 本 task 负责的 q 头区间（共享同一 kv 头）
        int qLo, qHi;         // query 行区间
        const float *kF32Base, *vF32Base;
        int kvHeadIdx;

        MultiThreadPagedAttentionGqaFloat32Op(
            float *qHeadBase, float *oHeadBase, float *maskHeadBase,
            int qStride, int oStride, int maskStride, int maskHeadDenom,
            float scale, int q1, int q2, int k1, int v2, int group,
            int kHeadDim, int vHeadDim,
            int qHeadLo, int qHeadHi, int qLo, int qHi,
            const float *kF32Base, const float *vF32Base, int kvHeadIdx) :
            qHeadBase(qHeadBase), oHeadBase(oHeadBase), maskHeadBase(maskHeadBase),
            qStride(qStride), oStride(oStride), maskStride(maskStride),
            maskHeadDenom(maskHeadDenom),
            scale(scale), q1(q1), q2(q2), k1(k1), v2(v2), group(group),
            kHeadDim(kHeadDim), vHeadDim(vHeadDim),
            qHeadLo(qHeadLo), qHeadHi(qHeadHi), qLo(qLo), qHi(qHi),
            kF32Base(kF32Base), vF32Base(vF32Base), kvHeadIdx(kvHeadIdx) {}

        void Run() {
            const int rows = qHi - qLo;
            const int base = k1 - q1;
            const int nHead = qHeadHi - qHeadLo;
            const float *kF32 = kF32Base + (size_t)kvHeadIdx * k1 * kHeadDim;
            const float *vF32 = vF32Base + (size_t)kvHeadIdx * k1 * vHeadDim;

            const int kvBlock = 128;
            static thread_local std::vector<float> scratchS;
            static thread_local std::vector<float> scratchAccH;
            static thread_local std::vector<float> scratchMValH;
            static thread_local std::vector<float> scratchLValH;
            scratchS.assign((size_t)nHead * rows * kvBlock, 0.0f);
            scratchAccH.assign((size_t)nHead * rows * v2, 0.0f);
            scratchMValH.assign((size_t)nHead * rows, -1e30f);
            scratchLValH.assign((size_t)nHead * rows, 0.0f);
            float *s = scratchS.data();
            float *accH = scratchAccH.data();
            float *mValH = scratchMValH.data();
            float *lValH = scratchLValH.data();

            // token 外层、head 内层：每个 K/V token 只从内存流经一次，
            // 组内全部 q 头复用。decode(q1=1) 时 K/V DRAM 流量从 nHead×
            // 降为 1×，是关键提速点；prefill 亦减少 K 的重复加载。
            for (int jb = 0; jb < k1; jb += kvBlock) {
                const int jEnd = std::min(jb + kvBlock, k1);
                const int blockLen = jEnd - jb;
                const bool blockFullyVisible = !maskHeadBase && (jEnd - 1) <= base + qLo;
                const bool blockFullyInvisible = !maskHeadBase && jb > base + qHi - 1;
                if (blockFullyInvisible) {
                    continue;
                }

                // A: 本块 scores（token 外层，head 内层，复用 kToken）
                for (int j = jb; j < jEnd; j++) {
                    const float *kToken = kF32 + (size_t)j * kHeadDim;
                    for (int ho = 0; ho < nHead; ho++) {
                        const int o = qHeadLo + ho;
                        const float *qHead = qHeadBase + (size_t)o * qStride;
                        const float *maskHead = maskHeadBase ? (maskHeadBase + (size_t)(o / maskHeadDenom) * maskStride) : nullptr;
                        float *sHead = s + (size_t)ho * rows * blockLen;
                        int t = 0;
                        for (; t + 1 < rows; t += 2) {
                            const int i0 = qLo + t;
                            const int i1 = qLo + t + 1;
                            float *dst0 = sHead + (size_t)t * blockLen + (j - jb);
                            float *dst1 = sHead + (size_t)(t + 1) * blockLen + (j - jb);
                            const float *qRow0 = qHead + (size_t)i0 * q2;
                            const float *qRow1 = qHead + (size_t)i1 * q2;
                            float dot0 = 0.0f, dot1 = 0.0f;
                            int l = 0;
#ifdef __AVX__
                            __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps(), a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
                            __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps(), a12 = _mm256_setzero_ps(), a13 = _mm256_setzero_ps();
                            for (; l + 31 < kHeadDim; l += 32) {
                                __m256 k0 = _mm256_loadu_ps((const float *)(kToken + l));
                                __m256 k1 = _mm256_loadu_ps((const float *)(kToken + l + 8));
                                __m256 k2 = _mm256_loadu_ps((const float *)(kToken + l + 16));
                                __m256 k3 = _mm256_loadu_ps((const float *)(kToken + l + 24));
                                a00 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *)(qRow0 + l)), k0, a00);
                                a01 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *)(qRow0 + l + 8)), k1, a01);
                                a02 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *)(qRow0 + l + 16)), k2, a02);
                                a03 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *)(qRow0 + l + 24)), k3, a03);
                                a10 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *)(qRow1 + l)), k0, a10);
                                a11 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *)(qRow1 + l + 8)), k1, a11);
                                a12 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *)(qRow1 + l + 16)), k2, a12);
                                a13 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *)(qRow1 + l + 24)), k3, a13);
                            }
                            __m256 vsum0 = _mm256_add_ps(_mm256_add_ps(a00, a01), _mm256_add_ps(a02, a03));
                            __m256 vsum1 = _mm256_add_ps(_mm256_add_ps(a10, a11), _mm256_add_ps(a12, a13));
                            dot0 = Floatsum(vsum0);
                            dot1 = Floatsum(vsum1);
                            for (; l + 7 < kHeadDim; l += 8) {
                                __m256 kv = _mm256_loadu_ps((const float *)(kToken + l));
                                dot0 += Floatsum(_mm256_mul_ps(_mm256_loadu_ps((const float *)(qRow0 + l)), kv));
                                dot1 += Floatsum(_mm256_mul_ps(_mm256_loadu_ps((const float *)(qRow1 + l)), kv));
                            }
#endif
                            for (; l < kHeadDim; l++) {
                                dot0 += qRow0[l] * kToken[l];
                                dot1 += qRow1[l] * kToken[l];
                            }
                            if (blockFullyVisible) {
                                *dst0 = dot0 * scale;
                                *dst1 = dot1 * scale;
                            } else if (maskHead) {
                                *dst0 = (maskHead[i0 * k1 + j] > 0.99) ? -10000.0f : dot0 * scale;
                                *dst1 = (maskHead[i1 * k1 + j] > 0.99) ? -10000.0f : dot1 * scale;
                            } else {
                                *dst0 = (j > base + i0) ? -10000.0f : dot0 * scale;
                                *dst1 = (j > base + i1) ? -10000.0f : dot1 * scale;
                            }
                        }
                        for (; t < rows; t++) {
                            const int i = qLo + t;
                            float *dst = sHead + (size_t)t * blockLen + (j - jb);
                            const float *qRow = qHead + (size_t)i * q2;
                            float dotProduct = 0.0f;
                            int l = 0;
#ifdef __AVX__
                            __m256 vsum0 = _mm256_setzero_ps();
                            __m256 vsum1 = _mm256_setzero_ps();
                            __m256 vsum2 = _mm256_setzero_ps();
                            __m256 vsum3 = _mm256_setzero_ps();
                            for (; l + 31 < kHeadDim; l += 32) {
                                vsum0 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l)),
                                                        _mm256_loadu_ps((const float *) (kToken + l)), vsum0);
                                vsum1 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 8)),
                                                        _mm256_loadu_ps((const float *) (kToken + l + 8)), vsum1);
                                vsum2 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 16)),
                                                        _mm256_loadu_ps((const float *) (kToken + l + 16)), vsum2);
                                vsum3 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 24)),
                                                        _mm256_loadu_ps((const float *) (kToken + l + 24)), vsum3);
                            }
                            __m256 vsum = _mm256_add_ps(_mm256_add_ps(vsum0, vsum1),
                                                        _mm256_add_ps(vsum2, vsum3));
                            for (; l + 7 < kHeadDim; l += 8) {
                                vsum = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l)),
                                                       _mm256_loadu_ps((const float *) (kToken + l)), vsum);
                            }
                            dotProduct += Floatsum(vsum);
#endif
                            for (; l < kHeadDim; l++) {
                                dotProduct += qRow[l] * kToken[l];
                            }
                            if (blockFullyVisible) {
                                *dst = dotProduct * scale;
                            } else if (maskHead) {
                                *dst = (maskHead[i * k1 + j] > 0.99) ? -10000.0f : dotProduct * scale;
                            } else {
                                *dst = (j > base + i) ? -10000.0f : dotProduct * scale;
                            }
                        }
                    }
                }

                // B: online softmax 状态更新（按头分片）
                for (int ho = 0; ho < nHead; ho++) {
                    float *mValLocal = mValH + (size_t)ho * rows;
                    float *lValLocal = lValH + (size_t)ho * rows;
                    float *accLocal = accH + (size_t)ho * rows * v2;
                    float *sHead = s + (size_t)ho * rows * blockLen;
                    for (int t = 0; t < rows; t++) {
                        float *sRow = sHead + (size_t)t * blockLen;
                        float blockMax = -1e30f;
                        for (int jj = 0; jj < blockLen; jj++) {
                            blockMax = std::max(blockMax, sRow[jj]);
                        }
                        float mNew = std::max(mValLocal[t], blockMax);
                        if (mNew <= -1e29f) {
                            continue;
                        }
                        float correction = (mValLocal[t] <= -1e29f) ? 0.0f : expf(mValLocal[t] - mNew);
                        float lt = 0.0f;
                        for (int jj = 0; jj < blockLen; jj++) {
                            float e = expf(sRow[jj] - mNew);
                            sRow[jj] = e;
                            lt += e;
                        }
                        lValLocal[t] = lValLocal[t] * correction + lt;
                        float *accRow = accLocal + (size_t)t * v2;
                        if (correction != 1.0f) {
                            int d = 0;
#ifdef __AVX__
                            __m256 cv = _mm256_set1_ps(correction);
                            for (; d + 7 < v2; d += 8) {
                                _mm256_storeu_ps(accRow + d, _mm256_mul_ps(cv, _mm256_loadu_ps(accRow + d)));
                            }
#endif
                            for (; d < v2; d++) {
                                accRow[d] *= correction;
                            }
                        }
                        mValLocal[t] = mNew;
                    }
                }

                // C: V 块累加（token 外层，head 内层，复用 vToken）
                for (int j = jb; j < jEnd; j++) {
                    const float *vToken = vF32 + (size_t)j * vHeadDim;
                    for (int ho = 0; ho < nHead; ho++) {
                        float *accLocal = accH + (size_t)ho * rows * v2;
                        float *sHead = s + (size_t)ho * rows * blockLen;
                        for (int t = 0; t < rows; t++) {
                            float w = sHead[(size_t)t * blockLen + (j - jb)];
                            if (w == 0.0f) {
                                continue;
                            }
                            float *oRow = accLocal + (size_t)t * v2;
                            int l = 0;
#ifdef __AVX__
                            __m256 wv = _mm256_set1_ps(w);
                            for (; l + 31 < v2; l += 32) {
                                _mm256_storeu_ps(oRow + l,
                                                 _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l),
                                                                 _mm256_loadu_ps(oRow + l)));
                                _mm256_storeu_ps(oRow + l + 8,
                                                 _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 8),
                                                                 _mm256_loadu_ps(oRow + l + 8)));
                                _mm256_storeu_ps(oRow + l + 16,
                                                 _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 16),
                                                                 _mm256_loadu_ps(oRow + l + 16)));
                                _mm256_storeu_ps(oRow + l + 24,
                                                 _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 24),
                                                                 _mm256_loadu_ps(oRow + l + 24)));
                            }
#endif
                            for (; l < v2; l++) {
                                oRow[l] += w * vToken[l];
                            }
                        }
                    }
                }

                // D: 最终归一化并写出
                for (int ho = 0; ho < nHead; ho++) {
                    const int o = qHeadLo + ho;
                    float *oHead = oHeadBase + (size_t)o * oStride;
                    float *lValLocal = lValH + (size_t)ho * rows;
                    float *accLocal = accH + (size_t)ho * rows * v2;
                    for (int t = 0; t < rows; t++) {
                        float denom = std::max(lValLocal[t], 0.1f);
                        float inv = 1.0f / denom;
                        const float *accRow = accLocal + (size_t)t * v2;
                        float *oRow = oHead + (size_t)(qLo + t) * v2;
                        int d = 0;
#ifdef __AVX__
                        __m256 iv = _mm256_set1_ps(inv);
                        for (; d + 7 < v2; d += 8) {
                            _mm256_storeu_ps(oRow + d, _mm256_mul_ps(iv, _mm256_loadu_ps(accRow + d)));
                        }
#endif
                        for (; d < v2; d++) {
                            oRow[d] = accRow[d] * inv;
                        }
                    }
                }
            }
        }
    };

    struct MultiThreadPagedAttentionConvertKVOp : MultiThreadBaseOp {
        uint8_t *kPagedData, *vPagedData;
        const std::vector<int> *kPageIndex, *vPageIndex;
        int pageLen, vPageLen;
        int kNumHeads, kHeadDim, kUnitSize, kLastPageLen;
        int vNumHeads, vHeadDim, vUnitSize, vLastPageLen;
        int k1, headIdx;
        int tokLo, tokHi; // 本任务负责的 token 区间 [tokLo, tokHi)
        DataType kCacheDataType, vCacheDataType;
        bool isBFloat16;
        float *dstK, *dstV;

        MultiThreadPagedAttentionConvertKVOp(
            uint8_t *kPagedData, uint8_t *vPagedData,
            const std::vector<int> *kPageIndex, const std::vector<int> *vPageIndex,
            int pageLen, int vPageLen,
            int kNumHeads, int kHeadDim, int kUnitSize, int kLastPageLen,
            int vNumHeads, int vHeadDim, int vUnitSize, int vLastPageLen,
            int k1, int headIdx,
            DataType kCacheDataType, DataType vCacheDataType,
            bool isBFloat16,
            float *dstK, float *dstV,
            int tokLo = 0, int tokHi = -1) :
            kPagedData(kPagedData), vPagedData(vPagedData),
            kPageIndex(kPageIndex), vPageIndex(vPageIndex),
            pageLen(pageLen), vPageLen(vPageLen),
            kNumHeads(kNumHeads), kHeadDim(kHeadDim), kUnitSize(kUnitSize), kLastPageLen(kLastPageLen),
            vNumHeads(vNumHeads), vHeadDim(vHeadDim), vUnitSize(vUnitSize), vLastPageLen(vLastPageLen),
            k1(k1), headIdx(headIdx),
            tokLo(tokLo), tokHi(tokHi < 0 ? k1 : tokHi),
            kCacheDataType(kCacheDataType), vCacheDataType(vCacheDataType),
            isBFloat16(isBFloat16), dstK(dstK), dstV(dstV) {}

        // 把某个 kv 头在本次请求中用到的 K/V 序列一次性转成连续 fp32，
        // 供同 group 的多个 q 头共享；替代 attention 内核里逐 query 行的
        // 重复转换与逐 (q, kv) 对的堆分配。
        void Run() {
            int kvTokenIdx = 0;
            for (size_t pageIdx = 0; pageIdx < kPageIndex->size(); pageIdx++) {
                int currentPageIdx = (*kPageIndex)[pageIdx];
                int tokensInPage = (pageIdx == kPageIndex->size() - 1) ? kLastPageLen : pageLen;
                for (int t = 0; t < tokensInPage; t++) {
                    if (kvTokenIdx >= tokLo && kvTokenIdx < tokHi) {
                        uint8_t *kDataPtr = kPagedData +
                            ((size_t)currentPageIdx * pageLen * kNumHeads * kHeadDim +
                             t * kNumHeads * kHeadDim +
                             headIdx * kHeadDim) * kUnitSize;
                        float *dst = dstK + (size_t)kvTokenIdx * kHeadDim;
                        if (kCacheDataType == DataType::FLOAT32) {
                            memcpy(dst, kDataPtr, (size_t)kHeadDim * sizeof(float));
                        } else if (isBFloat16) {
                            BFloat16ToFloat32((uint16_t*)kDataPtr, dst, kHeadDim);
                        } else {
                            Float16ToFloat32((uint16_t*)kDataPtr, dst, kHeadDim);
                        }
                    }
                    kvTokenIdx++;
                }
            }
            kvTokenIdx = 0;
            for (size_t pageIdx = 0; pageIdx < vPageIndex->size(); pageIdx++) {
                int currentPageIdx = (*vPageIndex)[pageIdx];
                int tokensInPage = (pageIdx == vPageIndex->size() - 1) ? vLastPageLen : vPageLen;
                for (int t = 0; t < tokensInPage; t++) {
                    if (kvTokenIdx >= tokLo && kvTokenIdx < tokHi) {
                        uint8_t *vDataPtr = vPagedData +
                            ((size_t)currentPageIdx * vPageLen * vNumHeads * vHeadDim +
                             t * vNumHeads * vHeadDim +
                             headIdx * vHeadDim) * vUnitSize;
                        float *dst = dstV + (size_t)kvTokenIdx * vHeadDim;
                        if (vCacheDataType == DataType::FLOAT32) {
                            memcpy(dst, vDataPtr, (size_t)vHeadDim * sizeof(float));
                        } else if (isBFloat16) {
                            BFloat16ToFloat32((uint16_t*)vDataPtr, dst, vHeadDim);
                        } else {
                            Float16ToFloat32((uint16_t*)vDataPtr, dst, vHeadDim);
                        }
                    }
                    kvTokenIdx++;
                }
            }
        }
    };

    struct MultiThreadPagedAttentionFloat16Op : MultiThreadBaseOp {
        uint16_t *qHead, *oHead, *maskHead;
        float scale;
        int q1, q2, k1, v2, group, o;
        int qLo, qHi; // 本任务负责的 query 行区间（绝对行号）
        int pageLen, kNumHeads, kHeadDim, kUnitSize;
        int vPageLen, vNumHeads, vHeadDim, vUnitSize;
        uint8_t *kPagedData, *vPagedData;
        const std::vector<int> *kPageIndex, *vPageIndex;
        int kLastPageLen, vLastPageLen;
        DataType kCacheDataType, vCacheDataType;
        const float *kF32Base, *vF32Base; // 预转换的连续 fp32 K/V，布局 [head][token][dim]

        MultiThreadPagedAttentionFloat16Op(
            uint16_t *qHead, uint16_t *oHead, uint16_t *maskHead, float scale,
            int q1, int q2, int k1, int v2, int group, int o,
            int qLo, int qHi,
            int pageLen, int kNumHeads, int kHeadDim, int kUnitSize,
            int vPageLen, int vNumHeads, int vHeadDim, int vUnitSize,
            uint8_t *kPagedData, uint8_t *vPagedData,
            const std::vector<int> *kPageIndex, const std::vector<int> *vPageIndex,
            int kLastPageLen, int vLastPageLen,
            DataType kCacheDataType, DataType vCacheDataType,
            const float *kF32Base, const float *vF32Base) :
            qHead(qHead), oHead(oHead), maskHead(maskHead), scale(scale),
            q1(q1), q2(q2), k1(k1), v2(v2), group(group), o(o),
            qLo(qLo), qHi(qHi),
            pageLen(pageLen), kNumHeads(kNumHeads), kHeadDim(kHeadDim), kUnitSize(kUnitSize),
            vPageLen(vPageLen), vNumHeads(vNumHeads), vHeadDim(vHeadDim), vUnitSize(vUnitSize),
            kPagedData(kPagedData), vPagedData(vPagedData),
            kPageIndex(kPageIndex), vPageIndex(vPageIndex),
            kLastPageLen(kLastPageLen), vLastPageLen(vLastPageLen),
            kCacheDataType(kCacheDataType), vCacheDataType(vCacheDataType),
            kF32Base(kF32Base), vF32Base(vF32Base) {}

        void Run() {
            const int kvHeadIdx = o / group;
            const float *kF32 = kF32Base + (size_t)kvHeadIdx * k1 * kHeadDim;
            const float *vF32 = vF32Base + (size_t)kvHeadIdx * k1 * vHeadDim;

            const int rows = qHi - qLo;
            const int base = k1 - q1;

            static thread_local std::vector<float> scratchFq;
            scratchFq.assign((size_t)rows * q2, 0.0f);
            float *fq = scratchFq.data();
            Float16ToFloat32(qHead + (size_t)qLo * q2, fq, rows * q2);

            // online softmax（flash）：K/V 按 kvBlock 分块，scores 块缓冲
            // 常驻 L2，不再物化 [rows × k1] 的中间分数矩阵。
            // 临时 buffer 提为线程局部并复用 capacity：本类 task 数量可达
            // 数百个，每 task 重新 malloc 百 KB 会高频触发 mmap/munmap 与
            // TLB shootdown，成为 prefill 长序列的瓶颈。
            const int kvBlock = 128;
            static thread_local std::vector<float> scratchAcc;
            static thread_local std::vector<float> scratchMVal;
            static thread_local std::vector<float> scratchLVal;
            static thread_local std::vector<float> scratchS;
            scratchAcc.assign((size_t)rows * v2, 0.0f);
            scratchMVal.assign((size_t)rows, -1e30f);
            scratchLVal.assign((size_t)rows, 0.0f);
            scratchS.assign((size_t)rows * kvBlock, 0.0f);
            float *acc = scratchAcc.data();
            float *mVal = scratchMVal.data();
            float *lVal = scratchLVal.data();
            float *s = scratchS.data();

            for (int jb = 0; jb < k1; jb += kvBlock) {
                const int jEnd = std::min(jb + kvBlock, k1);
                const int blockLen = jEnd - jb;

                // 整块对本 op 的所有行都不可见（因果）→ 直接跳过
                if (!maskHead && jb > base + qHi - 1) {
                    continue;
                }
                // 本块对所有行完全可见 → 免逐元素遮罩判断
                const bool allVisible = !maskHead && (jEnd - 1) <= base + qLo;

                // A: 本块 scores
                for (int j = jb; j < jEnd; j++) {
                    const float *kToken = kF32 + (size_t)j * kHeadDim;
                    for (int t = 0; t < rows; t++) {
                        const int i = qLo + t;
                        float *dst = s + (size_t)t * blockLen + (j - jb);
                        if (!allVisible) {
                            if (maskHead) {
                                float maskVal = half_to_float(maskHead[i * k1 + j]);
                                if (maskVal > 0.99) {
                                    *dst = -10000.0f;
                                    continue;
                                }
                            } else if (j > base + i) {
                                *dst = -10000.0f;
                                continue;
                            }
                        }

                        const float *qRow = fq + (size_t)t * q2;
                        float dotProduct = 0.0f;
                        int l = 0;
#ifdef __aarch64__
                        float32x4_t sum = {0, 0, 0, 0};
                        for (; l + 3 < q2; l += 4) {
                            sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(qRow + l),
                                                            vld1q_f32(kToken + l)));
                        }
                        dotProduct += sum[0] + sum[1] + sum[2] + sum[3];
#elif defined(__AVX__)
                        __m256 vsum0 = _mm256_setzero_ps();
                        __m256 vsum1 = _mm256_setzero_ps();
                        __m256 vsum2 = _mm256_setzero_ps();
                        __m256 vsum3 = _mm256_setzero_ps();
                        for (; l + 31 < q2; l += 32) {
                            vsum0 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l)),
                                                    _mm256_loadu_ps((const float *) (kToken + l)), vsum0);
                            vsum1 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 8)),
                                                    _mm256_loadu_ps((const float *) (kToken + l + 8)), vsum1);
                            vsum2 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 16)),
                                                    _mm256_loadu_ps((const float *) (kToken + l + 16)), vsum2);
                            vsum3 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 24)),
                                                    _mm256_loadu_ps((const float *) (kToken + l + 24)), vsum3);
                        }
                        __m256 vsum = _mm256_add_ps(_mm256_add_ps(vsum0, vsum1),
                                                    _mm256_add_ps(vsum2, vsum3));
                        for (; l + 7 < q2; l += 8) {
                            vsum = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l)),
                                                   _mm256_loadu_ps((const float *) (kToken + l)), vsum);
                        }
                        dotProduct += Floatsum(vsum);
#endif
                        for (; l < q2; l++) {
                            dotProduct += qRow[l] * kToken[l];
                        }
                        *dst = dotProduct * scale;
                    }
                }

                // B: online softmax 状态更新（行最大值 / 归一化因子 / acc 重标定）
                for (int t = 0; t < rows; t++) {
                    float *sRow = s + (size_t)t * blockLen;
                    float blockMax = -1e30f;
                    for (int jj = 0; jj < blockLen; jj++) {
                        blockMax = std::max(blockMax, sRow[jj]);
                    }
                    float mNew = std::max(mVal[t], blockMax);
                    if (mNew <= -1e29f) {
                        continue; // 到目前为止整行仍无可见 token
                    }
                    float correction = (mVal[t] <= -1e29f) ? 0.0f : expf(mVal[t] - mNew);
                    float lt = 0.0f;
                    for (int jj = 0; jj < blockLen; jj++) {
                        float e = expf(sRow[jj] - mNew);
                        sRow[jj] = e;
                        lt += e;
                    }
                    lVal[t] = lVal[t] * correction + lt;
                    float *accRow = acc + (size_t)t * v2;
                    if (correction != 1.0f) {
                        int d = 0;
#ifdef __aarch64__
                        float32x4_t cv = vdupq_n_f32(correction);
                        for (; d + 3 < v2; d += 4) {
                            vst1q_f32(accRow + d, vmulq_f32(cv, vld1q_f32(accRow + d)));
                        }
#elif defined(__AVX__)
                        __m256 cv = _mm256_set1_ps(correction);
                        for (; d + 7 < v2; d += 8) {
                            _mm256_storeu_ps(accRow + d, _mm256_mul_ps(cv, _mm256_loadu_ps(accRow + d)));
                        }
#endif
                        for (; d < v2; d++) {
                            accRow[d] *= correction;
                        }
                    }
                    mVal[t] = mNew;
                }

                // C: V 块累加（s 已是本块的未归一化概率）
                for (int j = jb; j < jEnd; j++) {
                    const float *vToken = vF32 + (size_t)j * vHeadDim;
                    for (int t = 0; t < rows; t++) {
                        float w = s[(size_t)t * blockLen + (j - jb)];
                        if (w == 0.0f) {
                            continue;
                        }
                        float *oRow = acc + (size_t)t * v2;
                        int l = 0;
#ifdef __aarch64__
                        float32x4_t wv = vdupq_n_f32(w);
                        for (; l + 3 < v2; l += 4) {
                            vst1q_f32(oRow + l,
                                      vaddq_f32(vld1q_f32(oRow + l),
                                                vmulq_f32(wv, vld1q_f32(vToken + l))));
                        }
#elif defined(__AVX__)
                        __m256 wv = _mm256_set1_ps(w);
                        for (; l + 31 < v2; l += 32) {
                            _mm256_storeu_ps(oRow + l,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l),
                                                             _mm256_loadu_ps(oRow + l)));
                            _mm256_storeu_ps(oRow + l + 8,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 8),
                                                             _mm256_loadu_ps(oRow + l + 8)));
                            _mm256_storeu_ps(oRow + l + 16,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 16),
                                                             _mm256_loadu_ps(oRow + l + 16)));
                            _mm256_storeu_ps(oRow + l + 24,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 24),
                                                             _mm256_loadu_ps(oRow + l + 24)));
                        }
                        for (; l + 7 < v2; l += 8) {
                            _mm256_storeu_ps(oRow + l,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l),
                                                             _mm256_loadu_ps(oRow + l)));
                        }
#endif
                        for (; l < v2; l++) {
                            oRow[l] += w * vToken[l];
                        }
                    }
                }
            }

            // 最终归一化并写出
            for (int t = 0; t < rows; t++) {
                float denom = std::max(lVal[t], 0.1f);
                float inv = 1.0f / denom;
                float *accRow = acc + (size_t)t * v2;
                int d = 0;
#ifdef __aarch64__
                float32x4_t iv = vdupq_n_f32(inv);
                for (; d + 3 < v2; d += 4) {
                    vst1q_f32(accRow + d, vmulq_f32(iv, vld1q_f32(accRow + d)));
                }
#elif defined(__AVX__)
                __m256 iv = _mm256_set1_ps(inv);
                for (; d + 7 < v2; d += 8) {
                    _mm256_storeu_ps(accRow + d, _mm256_mul_ps(iv, _mm256_loadu_ps(accRow + d)));
                }
#endif
                for (; d < v2; d++) {
                    accRow[d] *= inv;
                }
                Float32ToFloat16(accRow, oHead + (size_t)(qLo + t) * v2, v2);
            }
        }
    };

    struct MultiThreadPagedAttentionBFloat16Op : MultiThreadBaseOp {
        uint16_t *qHead, *oHead, *maskHead;
        float scale;
        int q1, q2, k1, v2, group, o;
        int qLo, qHi; // 本任务负责的 query 行区间（绝对行号）
        int pageLen, kNumHeads, kHeadDim, kUnitSize;
        int vPageLen, vNumHeads, vHeadDim, vUnitSize;
        uint8_t *kPagedData, *vPagedData;
        const std::vector<int> *kPageIndex, *vPageIndex;
        int kLastPageLen, vLastPageLen;
        DataType kCacheDataType, vCacheDataType;
        const float *kF32Base, *vF32Base; // 预转换的连续 fp32 K/V，布局 [head][token][dim]

        MultiThreadPagedAttentionBFloat16Op(
            uint16_t *qHead, uint16_t *oHead, uint16_t *maskHead, float scale,
            int q1, int q2, int k1, int v2, int group, int o,
            int qLo, int qHi,
            int pageLen, int kNumHeads, int kHeadDim, int kUnitSize,
            int vPageLen, int vNumHeads, int vHeadDim, int vUnitSize,
            uint8_t *kPagedData, uint8_t *vPagedData,
            const std::vector<int> *kPageIndex, const std::vector<int> *vPageIndex,
            int kLastPageLen, int vLastPageLen,
            DataType kCacheDataType, DataType vCacheDataType,
            const float *kF32Base, const float *vF32Base) :
            qHead(qHead), oHead(oHead), maskHead(maskHead), scale(scale),
            q1(q1), q2(q2), k1(k1), v2(v2), group(group), o(o),
            qLo(qLo), qHi(qHi),
            pageLen(pageLen), kNumHeads(kNumHeads), kHeadDim(kHeadDim), kUnitSize(kUnitSize),
            vPageLen(vPageLen), vNumHeads(vNumHeads), vHeadDim(vHeadDim), vUnitSize(vUnitSize),
            kPagedData(kPagedData), vPagedData(vPagedData),
            kPageIndex(kPageIndex), vPageIndex(vPageIndex),
            kLastPageLen(kLastPageLen), vLastPageLen(vLastPageLen),
            kCacheDataType(kCacheDataType), vCacheDataType(vCacheDataType),
            kF32Base(kF32Base), vF32Base(vF32Base) {}

        void Run() {
            const int kvHeadIdx = o / group;
            const float *kF32 = kF32Base + (size_t)kvHeadIdx * k1 * kHeadDim;
            const float *vF32 = vF32Base + (size_t)kvHeadIdx * k1 * vHeadDim;

            const int rows = qHi - qLo;
            const int base = k1 - q1;

            static thread_local std::vector<float> scratchFq;
            scratchFq.assign((size_t)rows * q2, 0.0f);
            float *fq = scratchFq.data();
            BFloat16ToFloat32(qHead + (size_t)qLo * q2, fq, rows * q2);

            // online softmax（flash）：K/V 按 kvBlock 分块，scores 块缓冲
            // 常驻 L2，不再物化 [rows × k1] 的中间分数矩阵。
            // 临时 buffer 提为线程局部并复用 capacity：本类 task 数量可达
            // 数百个，每 task 重新 malloc 百 KB 会高频触发 mmap/munmap 与
            // TLB shootdown，成为 prefill 长序列的瓶颈。
            const int kvBlock = 128;
            static thread_local std::vector<float> scratchAcc;
            static thread_local std::vector<float> scratchMVal;
            static thread_local std::vector<float> scratchLVal;
            static thread_local std::vector<float> scratchS;
            scratchAcc.assign((size_t)rows * v2, 0.0f);
            scratchMVal.assign((size_t)rows, -1e30f);
            scratchLVal.assign((size_t)rows, 0.0f);
            scratchS.assign((size_t)rows * kvBlock, 0.0f);
            float *acc = scratchAcc.data();
            float *mVal = scratchMVal.data();
            float *lVal = scratchLVal.data();
            float *s = scratchS.data();

            for (int jb = 0; jb < k1; jb += kvBlock) {
                const int jEnd = std::min(jb + kvBlock, k1);
                const int blockLen = jEnd - jb;

                // 整块对本 op 的所有行都不可见（因果）→ 直接跳过
                if (!maskHead && jb > base + qHi - 1) {
                    continue;
                }
                // 本块对所有行完全可见 → 免逐元素遮罩判断
                const bool allVisible = !maskHead && (jEnd - 1) <= base + qLo;

                // A: 本块 scores
                for (int j = jb; j < jEnd; j++) {
                    const float *kToken = kF32 + (size_t)j * kHeadDim;
                    for (int t = 0; t < rows; t++) {
                        const int i = qLo + t;
                        float *dst = s + (size_t)t * blockLen + (j - jb);
                        if (!allVisible) {
                            if (maskHead) {
                                float maskVal;
                                uint32_t mx = (uint32_t)maskHead[i * k1 + j] << 16;
                                memcpy(&maskVal, &mx, sizeof(float));
                                if (maskVal > 0.99) {
                                    *dst = -10000.0f;
                                    continue;
                                }
                            } else if (j > base + i) {
                                *dst = -10000.0f;
                                continue;
                            }
                        }

                        const float *qRow = fq + (size_t)t * q2;
                        float dotProduct = 0.0f;
                        int l = 0;
#ifdef __aarch64__
                        float32x4_t sum = {0, 0, 0, 0};
                        for (; l + 3 < q2; l += 4) {
                            sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(qRow + l),
                                                            vld1q_f32(kToken + l)));
                        }
                        dotProduct += sum[0] + sum[1] + sum[2] + sum[3];
#elif defined(__AVX__)
                        __m256 vsum0 = _mm256_setzero_ps();
                        __m256 vsum1 = _mm256_setzero_ps();
                        __m256 vsum2 = _mm256_setzero_ps();
                        __m256 vsum3 = _mm256_setzero_ps();
                        for (; l + 31 < q2; l += 32) {
                            vsum0 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l)),
                                                    _mm256_loadu_ps((const float *) (kToken + l)), vsum0);
                            vsum1 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 8)),
                                                    _mm256_loadu_ps((const float *) (kToken + l + 8)), vsum1);
                            vsum2 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 16)),
                                                    _mm256_loadu_ps((const float *) (kToken + l + 16)), vsum2);
                            vsum3 = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l + 24)),
                                                    _mm256_loadu_ps((const float *) (kToken + l + 24)), vsum3);
                        }
                        __m256 vsum = _mm256_add_ps(_mm256_add_ps(vsum0, vsum1),
                                                    _mm256_add_ps(vsum2, vsum3));
                        for (; l + 7 < q2; l += 8) {
                            vsum = _mm256_fmadd_ps(_mm256_loadu_ps((const float *) (qRow + l)),
                                                   _mm256_loadu_ps((const float *) (kToken + l)), vsum);
                        }
                        dotProduct += Floatsum(vsum);
#endif
                        for (; l < q2; l++) {
                            dotProduct += qRow[l] * kToken[l];
                        }
                        *dst = dotProduct * scale;
                    }
                }

                // B: online softmax 状态更新（行最大值 / 归一化因子 / acc 重标定）
                for (int t = 0; t < rows; t++) {
                    float *sRow = s + (size_t)t * blockLen;
                    float blockMax = -1e30f;
                    for (int jj = 0; jj < blockLen; jj++) {
                        blockMax = std::max(blockMax, sRow[jj]);
                    }
                    float mNew = std::max(mVal[t], blockMax);
                    if (mNew <= -1e29f) {
                        continue; // 到目前为止整行仍无可见 token
                    }
                    float correction = (mVal[t] <= -1e29f) ? 0.0f : expf(mVal[t] - mNew);
                    float lt = 0.0f;
                    for (int jj = 0; jj < blockLen; jj++) {
                        float e = expf(sRow[jj] - mNew);
                        sRow[jj] = e;
                        lt += e;
                    }
                    lVal[t] = lVal[t] * correction + lt;
                    float *accRow = acc + (size_t)t * v2;
                    if (correction != 1.0f) {
                        int d = 0;
#ifdef __aarch64__
                        float32x4_t cv = vdupq_n_f32(correction);
                        for (; d + 3 < v2; d += 4) {
                            vst1q_f32(accRow + d, vmulq_f32(cv, vld1q_f32(accRow + d)));
                        }
#elif defined(__AVX__)
                        __m256 cv = _mm256_set1_ps(correction);
                        for (; d + 7 < v2; d += 8) {
                            _mm256_storeu_ps(accRow + d, _mm256_mul_ps(cv, _mm256_loadu_ps(accRow + d)));
                        }
#endif
                        for (; d < v2; d++) {
                            accRow[d] *= correction;
                        }
                    }
                    mVal[t] = mNew;
                }

                // C: V 块累加（s 已是本块的未归一化概率）
                for (int j = jb; j < jEnd; j++) {
                    const float *vToken = vF32 + (size_t)j * vHeadDim;
                    for (int t = 0; t < rows; t++) {
                        float w = s[(size_t)t * blockLen + (j - jb)];
                        if (w == 0.0f) {
                            continue;
                        }
                        float *oRow = acc + (size_t)t * v2;
                        int l = 0;
#ifdef __aarch64__
                        float32x4_t wv = vdupq_n_f32(w);
                        for (; l + 3 < v2; l += 4) {
                            vst1q_f32(oRow + l,
                                      vaddq_f32(vld1q_f32(oRow + l),
                                                vmulq_f32(wv, vld1q_f32(vToken + l))));
                        }
#elif defined(__AVX__)
                        __m256 wv = _mm256_set1_ps(w);
                        for (; l + 31 < v2; l += 32) {
                            _mm256_storeu_ps(oRow + l,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l),
                                                             _mm256_loadu_ps(oRow + l)));
                            _mm256_storeu_ps(oRow + l + 8,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 8),
                                                             _mm256_loadu_ps(oRow + l + 8)));
                            _mm256_storeu_ps(oRow + l + 16,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 16),
                                                             _mm256_loadu_ps(oRow + l + 16)));
                            _mm256_storeu_ps(oRow + l + 24,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l + 24),
                                                             _mm256_loadu_ps(oRow + l + 24)));
                        }
                        for (; l + 7 < v2; l += 8) {
                            _mm256_storeu_ps(oRow + l,
                                             _mm256_fmadd_ps(wv, _mm256_loadu_ps(vToken + l),
                                                             _mm256_loadu_ps(oRow + l)));
                        }
#endif
                        for (; l < v2; l++) {
                            oRow[l] += w * vToken[l];
                        }
                    }
                }
            }

            // 最终归一化并写出
            for (int t = 0; t < rows; t++) {
                float denom = std::max(lVal[t], 0.1f);
                float inv = 1.0f / denom;
                float *accRow = acc + (size_t)t * v2;
                int d = 0;
#ifdef __aarch64__
                float32x4_t iv = vdupq_n_f32(inv);
                for (; d + 3 < v2; d += 4) {
                    vst1q_f32(accRow + d, vmulq_f32(iv, vld1q_f32(accRow + d)));
                }
#elif defined(__AVX__)
                __m256 iv = _mm256_set1_ps(inv);
                for (; d + 7 < v2; d += 8) {
                    _mm256_storeu_ps(accRow + d, _mm256_mul_ps(iv, _mm256_loadu_ps(accRow + d)));
                }
#endif
                for (; d < v2; d++) {
                    accRow[d] *= inv;
                }
                Float32ToBFloat16(accRow, oHead + (size_t)(qLo + t) * v2, v2);
            }
        }
    };

    void CpuAttentionPagedOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];
    
        std::vector <int> dims = {q.dims[0], q.dims[1], v.dims[2]};
        output.dataType = q.dataType;
        output.Resize(dims);
    }

    void CpuAttentionPagedOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        if (k.isPagedKVCache && v.isPagedKVCache) {
            AssertInFastLLM(k.pagedKVCacheData->dataType != DataType::FP8_E4M3 &&
                            v.pagedKVCacheData->dataType != DataType::FP8_E4M3,
                            "CpuAttentionPagedOp doesn't support fp8_e4m3 KV cache.\n");
            int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];
            float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0;
            
            int q0 = q.dims[0]; // numHeads
            int q1 = q.dims[1]; // seqLen
            int q2 = q.dims[2]; // headDim
            int k0 = k.dims[0]; // numHeads (for k cache)
            int v2 = v.dims[2]; // headDim (for v cache)
            
            // 计算总的 k/v 序列长度
            int k1 = 0;
            if (k.pageIndex.size() > 0) {
                k1 = (k.pageIndex.size() - 1) * k.pageLen + k.lastPageLen;
            }
            
            // 获取 K pagedKVCacheData 的信息
            int pageLen = k.pageLen;
            int kNumHeads = k.pagedKVCacheData->dims[2];
            int kHeadDim = k.pagedKVCacheData->dims[3];
            int kUnitSize = k.unitSize;
            
            // 获取 V pagedKVCacheData 的信息
            int vPageLen = v.pageLen;
            int vNumHeads = v.pagedKVCacheData->dims[2];
            int vHeadDim = v.pagedKVCacheData->dims[3];
            int vUnitSize = v.unitSize;

            // AttentionPaged 路径诊断：确认命中哪个 dtype 分支及形状/并行信息
            static const bool pagedAttnProf = std::getenv("FASTLLM_PROFILE_SLOW_OPS") != nullptr;
            float pagedAttnConvSpend = -1.0f;
            float pagedAttnComputeSpend = -1.0f;
            auto pagedAttnTotalSt = std::chrono::system_clock::now();

            if (q.dataType == DataType::FLOAT32) {
                float *qd = (float*)q.cpuData;
                float *od = (float*)output.cpuData;
                Data *maskPtr = datas.find("mask") != datas.end() ? datas.find("mask")->second : nullptr;
                float *maskd = (maskPtr != nullptr && maskPtr->dims.size() > 0) ? (float*)maskPtr->cpuData : nullptr;
                uint8_t *kPagedData = k.pagedKVCacheData->cpuData;
                uint8_t *vPagedData = v.pagedKVCacheData->cpuData;
                
                int batch = (maskd != nullptr && maskPtr != nullptr && maskPtr->dims.size() == 3) ? maskPtr->dims[0] : 1;
                batch = intParams.find("mask___batch") != intParams.end() ? intParams.find("mask___batch")->second : batch;
                int maskStride = (maskd != nullptr && maskPtr != nullptr) ? (maskPtr->dims.size() == 3 ? maskPtr->strides[0] : maskPtr->Count(0)) : 0;
                
                std::fill(od, od + output.Count(0), 0.0f);
                
                auto *pool = GetAlivePool();
                int threads = pool->threads.size();

                // 先把本次请求用到的 K/V 各 kv 头一次性整理为连续 fp32
                // （布局 [head][token][dim]），供同 group 的 q 头共享。
                auto convSt = std::chrono::system_clock::now();
                std::vector<float> kF32((size_t)k1 * kNumHeads * kHeadDim);
                std::vector<float> vF32((size_t)k1 * vNumHeads * vHeadDim);
                {
                    std::vector<MultiThreadPagedAttentionConvertKVOp*> convOps;
                    // decode 时 kvHeads 仅 2，conv 若按头分只有 2 路并行；
                    // 这里再按 token 区间切分，让全部线程参与转换。
                    int convChunks = std::max(1, threads / std::max(1, kNumHeads));
                    for (int h = 0; h < kNumHeads; h++) {
                        for (int c = 0; c < convChunks; c++) {
                            int tokLo = (int)((size_t)c * k1 / convChunks);
                            int tokHi = (int)((size_t)(c + 1) * k1 / convChunks);
                            if (tokLo >= tokHi) continue;
                            convOps.push_back(new MultiThreadPagedAttentionConvertKVOp(
                                kPagedData, vPagedData,
                                &k.pageIndex, &v.pageIndex,
                                pageLen, vPageLen,
                                kNumHeads, kHeadDim, kUnitSize, k.lastPageLen,
                                vNumHeads, vHeadDim, vUnitSize, v.lastPageLen,
                                k1, h,
                                k.pagedKVCacheData->dataType, v.pagedKVCacheData->dataType,
                                false,
                                kF32.data() + (size_t)h * k1 * kHeadDim,
                                vF32.data() + (size_t)h * k1 * vHeadDim,
                                tokLo, tokHi));
                        }
                    }
                    for (int st = 0; st < (int)convOps.size(); st += threads) {
                        int end = std::min(st + threads, (int)convOps.size());
                        for (int i = st; i < end; i++) {
                            pool->PushOp(i - st, convOps[i]);
                        }
                        for (int i = st; i < end; i++) {
                            pool->Wait(i - st);
                        }
                    }
                    for (auto *op : convOps) delete op;
                }
                pagedAttnConvSpend = GetSpan(convSt, std::chrono::system_clock::now());

                std::vector<MultiThreadPagedAttentionGqaFloat32Op*> ops;
                // 任务粒度 = (同一 kv 头的 8 个 q 头) × 64 行 query 块：
                // K/V 块在外层仅流经内存一次，组内 q 头复用，group=8 时
                // K/V DRAM 流量降为旧实现的 1/8。
                const int qTileRows = 64;
                int qBlocks = (q1 + qTileRows - 1) / qTileRows;
                for (int c = 0; c < q0; c += group) {
                    int qHeadHi = std::min(c + group, q0);
                    for (int b = 0; b < qBlocks; b++) {
                        int qLo = b * qTileRows;
                        int qHi = std::min(qLo + qTileRows, q1);
                        ops.push_back(new MultiThreadPagedAttentionGqaFloat32Op(
                            qd, od, maskd,
                            q.strides[0], output.strides[0], maskStride,
                            q0 / batch,
                            scale, q1, q2, k1, v2, group,
                            kHeadDim, vHeadDim,
                            c, qHeadHi, qLo, qHi,
                            kF32.data(), vF32.data(), c / group));
                    }
                }
                auto computeSt = std::chrono::system_clock::now();
                for (int st = 0; st < (int)ops.size(); st += threads) {
                    int end = std::min(st + threads, (int)ops.size());
                    for (int i = st; i < end; i++) {
                        pool->PushOp(i - st, ops[i]);
                    }
                    for (int i = st; i < end; i++) {
                        pool->Wait(i - st);
                    }
                }
                pagedAttnComputeSpend = GetSpan(computeSt, std::chrono::system_clock::now());
                for (auto *op : ops) delete op;
            } else if (q.dataType == DataType::FLOAT16) {
                uint16_t *qd = (uint16_t*)q.cpuData;
                uint16_t *od = (uint16_t*)output.cpuData;
                Data *maskPtr = datas.find("mask") != datas.end() ? datas.find("mask")->second : nullptr;
                uint16_t *maskd = (maskPtr != nullptr && maskPtr->dims.size() > 0) ? (uint16_t*)maskPtr->cpuData : nullptr;
                uint8_t *kPagedData = k.pagedKVCacheData->cpuData;
                uint8_t *vPagedData = v.pagedKVCacheData->cpuData;
                
                int batch = (maskd != nullptr && maskPtr != nullptr && maskPtr->dims.size() == 3) ? maskPtr->dims[0] : 1;
                batch = intParams.find("mask___batch") != intParams.end() ? intParams.find("mask___batch")->second : batch;
                int maskStride = (maskd != nullptr && maskPtr != nullptr) ? (maskPtr->dims.size() == 3 ? maskPtr->strides[0] : maskPtr->Count(0)) : 0;
                
                std::fill(od, od + output.Count(0), float_to_half(0.0f));
                
                auto *pool = GetAlivePool();
                int threads = pool->threads.size();

                // 先把本次请求用到的 K/V 各 kv 头一次性转换为连续 fp32
                // （布局 [head][token][dim]），供同 group 的 q 头共享。
                auto convSt = std::chrono::system_clock::now();
                std::vector<float> kF32((size_t)k1 * kNumHeads * kHeadDim);
                std::vector<float> vF32((size_t)k1 * vNumHeads * vHeadDim);
                {
                    std::vector<MultiThreadPagedAttentionConvertKVOp*> convOps;
                    for (int h = 0; h < kNumHeads; h++) {
                        convOps.push_back(new MultiThreadPagedAttentionConvertKVOp(
                            kPagedData, vPagedData,
                            &k.pageIndex, &v.pageIndex,
                            pageLen, vPageLen,
                            kNumHeads, kHeadDim, kUnitSize, k.lastPageLen,
                            vNumHeads, vHeadDim, vUnitSize, v.lastPageLen,
                            k1, h,
                            k.pagedKVCacheData->dataType, v.pagedKVCacheData->dataType,
                            false,
                            kF32.data() + (size_t)h * k1 * kHeadDim,
                            vF32.data() + (size_t)h * k1 * vHeadDim));
                    }
                    for (int st = 0; st < (int)convOps.size(); st += threads) {
                        int end = std::min(st + threads, (int)convOps.size());
                        for (int i = st; i < end; i++) {
                            pool->PushOp(i - st, convOps[i]);
                        }
                        for (int i = st; i < end; i++) {
                            pool->Wait(i - st);
                        }
                    }
                    for (auto *op : convOps) delete op;
                }
                pagedAttnConvSpend = GetSpan(convSt, std::chrono::system_clock::now());

                std::vector<MultiThreadPagedAttentionFloat16Op*> ops;
                // 任务粒度 = (q 头 × 64 行 query 块)：q0 小于线程数时也能
                // 填满线程池；同时 K/V 在块内被整块复用（j 外层扫描）。
                const int qTileRows = 64;
                int qBlocks = (q1 + qTileRows - 1) / qTileRows;
                for (int o = 0; o < q0; o++) {
                    uint16_t *qHead = qd + o * q.strides[0];
                    uint16_t *oHead = od + o * output.strides[0];
                    uint16_t *maskHead = maskd ? (maskd + (o / (q0 / batch)) * maskStride) : nullptr;
                    for (int b = 0; b < qBlocks; b++) {
                        int qLo = b * qTileRows;
                        int qHi = std::min(qLo + qTileRows, q1);
                        ops.push_back(new MultiThreadPagedAttentionFloat16Op(
                            qHead, oHead, maskHead, scale,
                            q1, q2, k1, v2, group, o,
                            qLo, qHi,
                            pageLen, kNumHeads, kHeadDim, kUnitSize,
                            vPageLen, vNumHeads, vHeadDim, vUnitSize,
                            kPagedData, vPagedData,
                            &k.pageIndex, &v.pageIndex,
                            k.lastPageLen, v.lastPageLen,
                            k.pagedKVCacheData->dataType, v.pagedKVCacheData->dataType,
                            kF32.data(), vF32.data()));
                    }
                }
                auto computeSt = std::chrono::system_clock::now();
                for (int st = 0; st < (int)ops.size(); st += threads) {
                    int end = std::min(st + threads, (int)ops.size());
                    for (int i = st; i < end; i++) {
                        pool->PushOp(i - st, ops[i]);
                    }
                    for (int i = st; i < end; i++) {
                        pool->Wait(i - st);
                    }
                }
                pagedAttnComputeSpend = GetSpan(computeSt, std::chrono::system_clock::now());
                for (auto *op : ops) delete op;
            } else if (q.dataType == DataType::BFLOAT16) {
                uint16_t *qd = (uint16_t*)q.cpuData;
                uint16_t *od = (uint16_t*)output.cpuData;
                Data *maskPtr = datas.find("mask") != datas.end() ? datas.find("mask")->second : nullptr;
                uint16_t *maskd = (maskPtr != nullptr && maskPtr->dims.size() > 0) ? (uint16_t*)maskPtr->cpuData : nullptr;
                uint8_t *kPagedData = k.pagedKVCacheData->cpuData;
                uint8_t *vPagedData = v.pagedKVCacheData->cpuData;
                
                int batch = (maskd != nullptr && maskPtr != nullptr && maskPtr->dims.size() == 3) ? maskPtr->dims[0] : 1;
                batch = intParams.find("mask___batch") != intParams.end() ? intParams.find("mask___batch")->second : batch;
                int maskStride = (maskd != nullptr && maskPtr != nullptr) ? (maskPtr->dims.size() == 3 ? maskPtr->strides[0] : maskPtr->Count(0)) : 0;
                
                {
                    uint16_t zero_bf16 = 0;
                    std::fill(od, od + output.Count(0), zero_bf16);
                }
                
                auto *pool = GetAlivePool();
                int threads = pool->threads.size();

                // 先把本次请求用到的 K/V 各 kv 头一次性转换为连续 fp32
                // （布局 [head][token][dim]），供同 group 的 q 头共享。
                auto convSt = std::chrono::system_clock::now();
                std::vector<float> kF32((size_t)k1 * kNumHeads * kHeadDim);
                std::vector<float> vF32((size_t)k1 * vNumHeads * vHeadDim);
                {
                    std::vector<MultiThreadPagedAttentionConvertKVOp*> convOps;
                    for (int h = 0; h < kNumHeads; h++) {
                        convOps.push_back(new MultiThreadPagedAttentionConvertKVOp(
                            kPagedData, vPagedData,
                            &k.pageIndex, &v.pageIndex,
                            pageLen, vPageLen,
                            kNumHeads, kHeadDim, kUnitSize, k.lastPageLen,
                            vNumHeads, vHeadDim, vUnitSize, v.lastPageLen,
                            k1, h,
                            k.pagedKVCacheData->dataType, v.pagedKVCacheData->dataType,
                            true,
                            kF32.data() + (size_t)h * k1 * kHeadDim,
                            vF32.data() + (size_t)h * k1 * vHeadDim));
                    }
                    for (int st = 0; st < (int)convOps.size(); st += threads) {
                        int end = std::min(st + threads, (int)convOps.size());
                        for (int i = st; i < end; i++) {
                            pool->PushOp(i - st, convOps[i]);
                        }
                        for (int i = st; i < end; i++) {
                            pool->Wait(i - st);
                        }
                    }
                    for (auto *op : convOps) delete op;
                }
                pagedAttnConvSpend = GetSpan(convSt, std::chrono::system_clock::now());

                std::vector<MultiThreadPagedAttentionBFloat16Op*> ops;
                // 任务粒度 = (q 头 × 64 行 query 块)：q0 小于线程数时也能
                // 填满线程池；同时 K/V 在块内被整块复用（j 外层扫描）。
                const int qTileRows = 64;
                int qBlocks = (q1 + qTileRows - 1) / qTileRows;
                for (int o = 0; o < q0; o++) {
                    uint16_t *qHead = qd + o * q.strides[0];
                    uint16_t *oHead = od + o * output.strides[0];
                    uint16_t *maskHead = maskd ? (maskd + (o / (q0 / batch)) * maskStride) : nullptr;
                    for (int b = 0; b < qBlocks; b++) {
                        int qLo = b * qTileRows;
                        int qHi = std::min(qLo + qTileRows, q1);
                        ops.push_back(new MultiThreadPagedAttentionBFloat16Op(
                            qHead, oHead, maskHead, scale,
                            q1, q2, k1, v2, group, o,
                            qLo, qHi,
                            pageLen, kNumHeads, kHeadDim, kUnitSize,
                            vPageLen, vNumHeads, vHeadDim, vUnitSize,
                            kPagedData, vPagedData,
                            &k.pageIndex, &v.pageIndex,
                            k.lastPageLen, v.lastPageLen,
                            k.pagedKVCacheData->dataType, v.pagedKVCacheData->dataType,
                            kF32.data(), vF32.data()));
                    }
                }
                auto computeSt = std::chrono::system_clock::now();
                for (int st = 0; st < (int)ops.size(); st += threads) {
                    int end = std::min(st + threads, (int)ops.size());
                    for (int i = st; i < end; i++) {
                        pool->PushOp(i - st, ops[i]);
                    }
                    for (int i = st; i < end; i++) {
                        pool->Wait(i - st);
                    }
                }
                pagedAttnComputeSpend = GetSpan(computeSt, std::chrono::system_clock::now());
                for (auto *op : ops) delete op;
            } else {
                ErrorInFastLLM("CpuAttentionPagedOp error: unsupport dataType.\n");
            }

            if (pagedAttnProf) {
                static int pagedAttnLogCnt = 0;
                int cnt = pagedAttnLogCnt++;
                float totalSpend = GetSpan(pagedAttnTotalSt, std::chrono::system_clock::now());
                printf("[fastllm-paged-attn] #%d dtype=%s cache=%s q0=%d q1=%d q2=%d k1=%d v2=%d group=%d kvHeads=%d threads=%d conv=%.4f compute=%.4f total=%.4f s\n",
                       cnt, GetDataTypeName(q.dataType).c_str(),
                       GetDataTypeName(k.pagedKVCacheData->dataType).c_str(),
                       q0, q1, q2, k1, v2, group, kNumHeads,
                       (int)GetAlivePool()->threads.size(), pagedAttnConvSpend,
                       pagedAttnComputeSpend, totalSpend);
                fflush(stdout);
            }
        }
    }

    void CpuAppendPagedCacheBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data **currentCaches = (Data**)(datas.find("currentCaches")->second);
        Data &insertIndexs = *(datas.find("insertIndexs")->second);
        Data &insertPositions = *(datas.find("insertPositions")->second);
        PagedCacheManager &manager = *(PagedCacheManager*)datas.find("pagedCacheManager")->second;
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16 ||
                        input.dataType == DataType::BFLOAT16,
                        "CpuAppendPagedCacheBatchOp's input type should be float32, float16 or bfloat16.\n");
        AssertInFastLLM(input.dims.size() == 3,
                        "CpuAppendPagedCacheBatchOp's input should have 3 dimensions [numHeads, batch, headDim].\n");
        AssertInFastLLM(insertIndexs.dims.size() == 1 && insertIndexs.dims[0] == input.dims[0],
                        "CpuAppendPagedCacheBatchOp's insertIndexs length should match batch.\n");
        AssertInFastLLM(insertPositions.dims.size() == 1 && insertPositions.dims[0] == input.dims[0],
                        "CpuAppendPagedCacheBatchOp's insertPositions length should match batch.\n");
        AssertInFastLLM(((Data*)&manager)->dataType != DataType::FP8_E4M3,
                        "CpuAppendPagedCacheBatchOp doesn't support fp8_e4m3 KV cache.\n");
        AssertInFastLLM(((Data*)&manager)->dims.size() == 4,
                        "CpuAppendPagedCacheBatchOp's pagedCacheManager storage should have 4 dimensions.\n");
        int batch = input.dims[0];
        for (int i = 0; i < batch; i++) {
            Data *currentCache = currentCaches[i];
            int pageLen = currentCache->pageLen;
            if (currentCache->pageIndex.empty() || currentCache->lastPageLen >= pageLen) {
                currentCache->pageIndex.push_back(manager.GetUnusedPageIndex(true));
                currentCache->lastPageLen = 1;
            } else {
                currentCache->lastPageLen++;
            }
        }
    }

    void CpuAppendPagedCacheBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second); // batch, num_heads, head_dim
        Data &insertIndexs = *(datas.find("insertIndexs")->second);
        Data &insertPositions = *(datas.find("insertPositions")->second);
        PagedCacheManager &manager = *(PagedCacheManager*)datas.find("pagedCacheManager")->second;

        int batch = input.dims[0], numHeads = input.dims[1], headDim = input.dims[2];
        int pageLen = manager.pageLen;
        int unitSize = input.unitSize;
        uint8_t *pagedData = (uint8_t*)((Data*)&manager)->cpuData;
        uint8_t *inputData = (uint8_t*)input.cpuData;
        int32_t *idxData = (int32_t*)insertIndexs.cpuData;
        int32_t *posData = (int32_t*)insertPositions.cpuData;

        for (int b = 0; b < batch; b++) {
            int pageIdx = idxData[b];
            int pageOffset = posData[b];
            for (int h = 0; h < numHeads; h++) {
                uint8_t *dst = pagedData +
                    ((size_t)pageIdx * pageLen * numHeads * headDim + pageOffset * numHeads * headDim + h * headDim) * unitSize;
                uint8_t *src = inputData + (b * numHeads * headDim + h * headDim) * unitSize;
                memcpy(dst, src, headDim * unitSize);
            }
        }
    }

    void CpuGenerateAppendPagedCacheBatchParamsOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &insertIndexs = *(datas.find("insertIndexs")->second);
        Data &insertPositions = *(datas.find("insertPositions")->second);
        int batch = intParams.find("pastKeys___batch") != intParams.end() ? intParams.find("pastKeys___batch")->second : 1;
        if (batch <= 0 && intParams.find("batch") != intParams.end()) {
            batch = intParams.find("batch")->second;
        }
        insertIndexs.dataType = DataType::INT32;
        insertIndexs.Resize({batch});
        insertPositions.dataType = DataType::INT32;
        insertPositions.Resize({batch});
    }

    void CpuGenerateAppendPagedCacheBatchParamsOp::Run(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        PagedCacheManager &manager = *(PagedCacheManager*)datas.find("pagedCacheManager")->second;
        Data **pastKeys = (Data**)(datas.find("pastKeys")->second);
        Data **pastValues = (Data**)(datas.find("pastValues")->second);
        Data &insertIndexs = *(datas.find("insertIndexs")->second);
        Data &insertPositions = *(datas.find("insertPositions")->second);
        int batch = intParams.find("pastKeys___batch") != intParams.end() ? intParams.find("pastKeys___batch")->second : 1;
        if (batch <= 0 && intParams.find("batch") != intParams.end()) {
            batch = intParams.find("batch")->second;
        }
        AssertInFastLLM(batch > 0, "CpuGenerateAppendPagedCacheBatchParamsOp: batch must be positive.\n");

        insertIndexs.Allocate();
        insertPositions.Allocate();
        int32_t *idxData = (int32_t*)insertIndexs.cpuData;
        int32_t *posData = (int32_t*)insertPositions.cpuData;

        int newPageCount = 0;
        for (int b = 0; b < batch; b++) {
            Data *pk = pastKeys[b];
            if (pk->pageIndex.empty() || pk->lastPageLen >= pk->pageLen) {
                newPageCount++;
            }
        }

        std::vector<int> previewNewPages;
        previewNewPages.reserve(newPageCount);
        if (newPageCount > 0) {
            std::lock_guard<std::mutex> guard(manager.pageIndexLocker);
            for (int i = 0; i < newPageCount && i < (int)manager.freePages.size(); i++) {
                previewNewPages.push_back(manager.freePages[(int)manager.freePages.size() - 1 - i]);
            }
            int trieOffset = newPageCount - (int)previewNewPages.size();
            for (int i = 0; i < trieOffset && i < (int)manager.triePages.size(); i++) {
                previewNewPages.push_back(manager.triePages[(int)manager.triePages.size() - 1 - i]);
            }
        }
        AssertInFastLLM((int)previewNewPages.size() >= newPageCount,
                        "CpuGenerateAppendPagedCacheBatchParamsOp: no enough pages for batch append.\n");

        int newPageOffset = 0;
        for (int b = 0; b < batch; b++) {
            Data *pk = pastKeys[b];
            int pageLen = pk->pageLen;
            int insertIdx, insertPos;
            if (pk->pageIndex.empty() || pk->lastPageLen >= pageLen) {
                insertIdx = previewNewPages[newPageOffset++];
                insertPos = 0;
            } else {
                insertIdx = pk->pageIndex.back();
                insertPos = pk->lastPageLen;
            }
            idxData[b] = insertIdx;
            posData[b] = insertPos;
        }
    }

    void CpuAttentionPagedBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &vCaches = *(datas.find("vCaches")->second);
        Data &output = *(datas.find("output")->second);

        std::vector <int> dims = {q.dims[0], q.dims[1], vCaches.pagedKVCacheData->dims[3]};
        output.dataType = q.dataType;
        output.Resize(dims);
    }

    void CpuAttentionPagedBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &kCaches = *(datas.find("kCaches")->second);
        Data &vCaches = *(datas.find("vCaches")->second);
        Data &qSizes = *(datas.find("qSizes")->second);
        Data &pageSizes = *(datas.find("pageSizes")->second);
        Data &pageIndexs = *(datas.find("pageIndexs")->second);
        Data &lastPageLens = *(datas.find("lastPageLens")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / kCaches.dims[0];
        float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0;
        int attentionType = intParams.find("attentionType") != intParams.end() ? intParams.find("attentionType")->second : 0;

        int32_t *qSizesData = (int32_t*)qSizes.cpuData;
        int32_t *pageSizesData = (int32_t*)pageSizes.cpuData;
        int32_t *pageIndexsData = (int32_t*)pageIndexs.cpuData;
        int32_t *lastPageLensData = (int32_t*)lastPageLens.cpuData;
        int batch = lastPageLens.dims[0];

        int numHeads = q.dims[0];
        int headDim = q.dims[2];
        int vHeadDim = vCaches.pagedKVCacheData->dims[3];
        int pageLen = kCaches.pageLen;

        int totalSeqLen = q.dims[1];
        CpuAttentionPagedOp pagedOp;
        for (int b = 0; b < batch; b++) {
            int qStart = qSizesData[b];
            int qEnd = qSizesData[b + 1];
            int qLen = qEnd - qStart;
            int pageStart = pageSizesData[b];
            int pageEnd = pageSizesData[b + 1];

            Data batchQ, batchK, batchV, batchOutput;

            batchQ.dataType = q.dataType;
            batchQ.Resize({numHeads, qLen, headDim});
            batchQ.Allocate();
            for (int h = 0; h < numHeads; h++) {
                memcpy(batchQ.cpuData + (size_t)h * qLen * headDim * q.unitSize,
                       q.cpuData + ((size_t)h * totalSeqLen + qStart) * headDim * q.unitSize,
                       (size_t)qLen * headDim * q.unitSize);
            }

            batchK.isFake = true;
            batchK.isPagedKVCache = true;
            batchK.dataType = kCaches.dataType;
            batchK.pagedKVCacheData = kCaches.pagedKVCacheData;
            batchK.pageLen = pageLen;
            batchK.unitSize = kCaches.unitSize;
            batchK.dims = kCaches.dims;
            batchK.pageIndex.assign(pageIndexsData + pageStart, pageIndexsData + pageEnd);
            batchK.lastPageLen = lastPageLensData[b];

            batchV.isFake = true;
            batchV.isPagedKVCache = true;
            batchV.dataType = vCaches.dataType;
            batchV.pagedKVCacheData = vCaches.pagedKVCacheData;
            batchV.pageLen = pageLen;
            batchV.unitSize = vCaches.unitSize;
            batchV.dims = vCaches.dims;
            batchV.pageIndex.assign(pageIndexsData + pageStart, pageIndexsData + pageEnd);
            batchV.lastPageLen = lastPageLensData[b];

            batchOutput.dataType = output.dataType;
            batchOutput.Resize({numHeads, qLen, vHeadDim});

            fastllm::DataDict tempDatas = {
                {"q", &batchQ}, {"k", &batchK}, {"v", &batchV}, {"output", &batchOutput}
            };
            fastllm::FloatDict tempFloatParams = {{"scale", scale}};
            fastllm::IntDict tempIntParams = {{"group", group}, {"attentionType", attentionType}};

            pagedOp.Reshape("AttentionPaged", tempDatas, tempFloatParams, tempIntParams);
            pagedOp.Run("AttentionPaged", tempDatas, tempFloatParams, tempIntParams);

            for (int h = 0; h < numHeads; h++) {
                memcpy(output.cpuData + ((size_t)h * totalSeqLen + qStart) * vHeadDim * output.unitSize,
                       batchOutput.cpuData + (size_t)h * qLen * vHeadDim * output.unitSize,
                       (size_t)qLen * vHeadDim * output.unitSize);
            }
        }


// ((fastllm::Data*)&output)->Resize({output.dims[1], output.dims[0], output.dims[2]});
DoCpuPermuteSelf(*((fastllm::Data*)&output), {1, 0, 2});
((fastllm::Data*)&output)->Resize({1, output.dims[0], output.dims[1] * output.dims[2]});
    }

    void CpuGeneratePagedBatchParamsOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data **pastKeys = (Data**)(datas.find("pastKeys")->second);
        Data &qSizes = *(datas.find("qSizes")->second);
        Data &pageSizes = *(datas.find("pageSizes")->second);
        Data &pageIndexs = *(datas.find("pageIndexs")->second);
        Data &lastPageLens = *(datas.find("lastPageLens")->second);

        int batch = intParams.find("pastKeys___batch") != intParams.end() ? intParams.find("pastKeys___batch")->second : 1;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        
        // qSizes: [batch + 1]
        qSizes.dataType = DataType::INT32;
        qSizes.Resize({batch + 1});
        
        // pageSizes: [batch + 1]
        pageSizes.dataType = DataType::INT32;
        pageSizes.Resize({batch + 1});
        
        // lastPageLens: [batch]
        lastPageLens.dataType = DataType::INT32;
        lastPageLens.Resize({batch});
        
        // 计算总的page数量
        int totalPages = 0;
        for (int b = 0; b < batch; b++) {
            totalPages += pastKeys[b]->pageIndex.size();
        }
        pageIndexs.dataType = DataType::INT32;
        pageIndexs.Resize({totalPages});
    }

    void CpuGeneratePagedBatchParamsOp::Run(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data **pastKeys = (Data**)(datas.find("pastKeys")->second);
        Data &qSizes = *(datas.find("qSizes")->second);
        Data &pageSizes = *(datas.find("pageSizes")->second);
        Data &pageIndexs = *(datas.find("pageIndexs")->second);
        Data &lastPageLens = *(datas.find("lastPageLens")->second);
        
        int batch = intParams.find("pastKeys___batch") != intParams.end() ? intParams.find("pastKeys___batch")->second : 1;
        
        // 分配输出内存
        qSizes.Allocate();
        pageSizes.Allocate();
        lastPageLens.Allocate();
        pageIndexs.Allocate();
        
        int32_t *qSizesData = (int32_t*)qSizes.cpuData;
        int32_t *pageSizesData = (int32_t*)pageSizes.cpuData;
        int32_t *lastPageLensData = (int32_t*)lastPageLens.cpuData;
        int32_t *pageIndexsData = (int32_t*)pageIndexs.cpuData;
        
        // 生成qSizes: 如果有 seqLens 则按 seqLens 的前缀和，否则按 [0, 1, 2, ..., batch]
        int seqLensSize = intParams.find("seqLens___size") != intParams.end() ? intParams.find("seqLens___size")->second : 0;
        qSizesData[0] = 0;
        if (seqLensSize > 0) {
            for (int b = 0; b < batch; b++) {
                int seqLen = intParams.find("seqLens___" + std::to_string(b))->second;
                qSizesData[b + 1] = qSizesData[b] + seqLen;
            }
        } else {
            for (int b = 0; b < batch; b++) {
                qSizesData[b + 1] = b + 1;
            }
        }
        
        // 生成pageSizes, pageIndexs, lastPageLens
        int pageOffset = 0;
        pageSizesData[0] = 0;
        for (int b = 0; b < batch; b++) {
            int numPages = pastKeys[b]->pageIndex.size();
            pageSizesData[b + 1] = pageSizesData[b] + numPages;
            
            // 复制pageIndex
            for (int i = 0; i < numPages; i++) {
                pageIndexsData[pageOffset + i] = pastKeys[b]->pageIndex[i];
            }
            pageOffset += numPages;
            
            // 设置lastPageLen
            lastPageLensData[b] = pastKeys[b]->lastPageLen;
        }
    }
}
