// Resident V4.1 GGUF experts: one launch per projection for all GPU routes.
#include "../fastllm-gguf-mmvq-common.cuh"
#include "../fastllm-gguf-q3-k-gemv.cuh"
#include "../fastllm-gguf-iq1-dot.cuh"
#include "fastllm-moe-gguf-q8.cuh"
#include "fastllm-moe-gguf-common.cuh"
#include "fastllm-cuda-v41-gguf.h"
#include "fastllm-cuda.cuh"
#include <cuda_fp8.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../fastllm-gguf-iq1-grid.cuh"
#include "../fastllm-gguf-iq2-dot.cuh"
#include <type_traits>

namespace {
__device__ __forceinline__ float V41Fp8Boundary(float x) {
    const float maximum = fmaxf(warp_reduce_max(fabsf(x)), 1e-4f);
    const unsigned bits = __float_as_uint(maximum * (1.0f/448.0f));
    const int exponent = int((bits>>23)&255)-127+int((bits&0x7fffff)!=0);
    const float scale = exp2f(float(exponent));
    return __bfloat162float(__float2bfloat16_rn(
        float(__nv_fp8_e4m3(fminf(448.f, fmaxf(-448.f, x/scale))))*scale));
}
__device__ __forceinline__ void V41StoreQ8(float x, block_q8_1 &q, int lane) {
    const float maximum = warp_reduce_max(fabsf(x));
    const float sum = warp_reduce_sum(x), scale = maximum/127.f;
    q.qs[lane] = maximum == 0 ? 0 : int8_t(roundf(x/scale));
    if (lane == 0) q.ds = __floats2half2_rn(scale, sum);
}
// The same FP8 block-32 -> BF16 -> Q8_1 boundaries as the ordinary CUDA
// Linear path. Input conversion is shared by every selected expert.
template<bool Fp8Boundary>
__global__ void V41GGUFQuantize(const __nv_bfloat16 *input, block_q8_1 *output, int columns) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= columns) return;
    float x = __bfloat162float(input[size_t(blockIdx.y)*columns+c]);
    if constexpr (Fp8Boundary) x = V41Fp8Boundary(x);
    auto &q = output[size_t(blockIdx.y)*(columns/32)+c/32];
    V41StoreQ8(x, q, c%32);
}

// Keep projection dispatch inline: sharing the fused gate/up helper changes
// NVCC register allocation for generic projection kernels despite equal math.
template<ggml_type Type, int VirtualWarps = 4, int LaneWidth = 32>
__device__ __forceinline__ float V41GGUFDot(const void *weight, const block_q8_1 *x, int columns, const uint32_t *iqGrid) {
    if constexpr (Type == GGML_TYPE_Q2_0) {
        return gguf_cache_q8::RowDot<Type>(weight, x, columns, nullptr);
    } else {
        constexpr int qi = [] {
            if constexpr (Type == GGML_TYPE_IQ1_S) return QI1_S;
            else if constexpr (Type == GGML_TYPE_IQ1_M) return QI1_M;
            else if constexpr (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_S) return 16;
            else return ggml_cuda_type_traits<Type>::qi;
        }();
        constexpr int vdr = (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_S) ? 2 : get_vdr_mmvq(Type);
        constexpr int step = vdr*LaneWidth*VirtualWarps/qi;
        const int lane = threadIdx.x%LaneWidth;
        float partial[VirtualWarps];
#pragma unroll
        for (int warp = 0; warp < VirtualWarps; ++warp) {
            const int tid = lane+LaneWidth*warp;
            float value = 0;
            for (int b = tid/(qi/vdr); b < columns/256; b += step) {
                if constexpr (Type == GGML_TYPE_IQ1_S)
                    value += vec_dot_iq1_s_q8_1_grid(weight, x+b*8, b, vdr*(tid%(qi/vdr)), iqGrid);
                else if constexpr (Type == GGML_TYPE_IQ1_M)
                    value += vec_dot_iq1_m_q8_1_grid(weight, x+b*8, b, vdr*(tid%(qi/vdr)), iqGrid);
                else if constexpr (Type == GGML_TYPE_IQ2_XXS)
                    value += iq2_decode::DotXXS<(VirtualWarps != 4)>(weight, x+b*8, b, vdr*(tid%(qi/vdr)), reinterpret_cast<const uint64_t *>(iqGrid));
                else if constexpr (Type == GGML_TYPE_IQ2_XS)
                    value += iq2_decode::DotXS<(VirtualWarps != 4)>(weight, x+b*8, b, vdr*(tid%(qi/vdr)), reinterpret_cast<const uint64_t *>(iqGrid));
                else if constexpr (Type == GGML_TYPE_IQ2_S)
                    value += iq2_decode::DotS<(VirtualWarps != 4)>(weight, x+b*8, b, vdr*(tid%(qi/vdr)), reinterpret_cast<const uint64_t *>(iqGrid));
                else if constexpr (Type == GGML_TYPE_Q3_K)
                    value += fastllm_gguf_q3_k::Dot(weight, x+b*8, b, vdr*(tid%(qi/vdr)));
                else value += get_vec_dot_q_cuda(Type)(weight, x+b*8, b, vdr*(tid%(qi/vdr)));
            }
            partial[warp] = value;
        }
        // Retain the four-warp reference order. A half warp calculates both
        // corresponding logical lanes before the remaining shuffle reductions.
        float sum;
        if constexpr (VirtualWarps == 8 && LaneWidth == 16) {
            float a = partial[0], b = partial[1];
#pragma unroll
            for (int warp=1;warp<4;++warp) {a+=partial[2*warp];b+=partial[2*warp+1];}
            sum = a+b;
        } else if constexpr (VirtualWarps == 16 && LaneWidth == 8) {
            float a=partial[0], b=partial[1], c=partial[2], d=partial[3];
#pragma unroll
            for (int warp=1; warp<4; ++warp) {
                a+=partial[4*warp]; b+=partial[4*warp+1];
                c+=partial[4*warp+2]; d+=partial[4*warp+3];
            }
            // Reconstruct the reference xor-16, then xor-8 reductions locally.
            sum=(a+c)+(b+d);
        } else {
            sum=partial[0];
#pragma unroll
            for (int warp=1;warp<VirtualWarps;++warp)sum+=partial[warp];
        }
#pragma unroll
        for (int mask = LaneWidth/2; mask; mask >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, mask, LaneWidth);
        return sum;
    }
}

template<ggml_type Type>
__device__ __forceinline__ float V41BlockDot(const void *weight,const block_q8_1 *x,int b,int q,const uint32_t*grid) {
    if constexpr(Type==GGML_TYPE_IQ1_S)return vec_dot_iq1_s_q8_1_grid(weight,x+b*8,b,q,grid);
    else if constexpr(Type==GGML_TYPE_IQ1_M)return vec_dot_iq1_m_q8_1_grid(weight,x+b*8,b,q,grid);
    else if constexpr(Type==GGML_TYPE_IQ2_XXS)return iq2_decode::DotXXS<true>(weight,x+b*8,b,q,reinterpret_cast<const uint64_t*>(grid));
    else if constexpr(Type==GGML_TYPE_IQ2_XS)return iq2_decode::DotXS<true>(weight,x+b*8,b,q,reinterpret_cast<const uint64_t*>(grid));
    else if constexpr(Type==GGML_TYPE_IQ2_S)return iq2_decode::DotS<true>(weight,x+b*8,b,q,reinterpret_cast<const uint64_t*>(grid));
    else if constexpr(Type==GGML_TYPE_Q3_K)return fastllm_gguf_q3_k::Dot(weight,x+b*8,b,q);
    else return get_vec_dot_q_cuda(Type)(weight,x+b*8,b,q);
}
template<ggml_type Type,int LaneWidth>
__device__ __forceinline__ float2 V41PairDot(const void *gate,const void *up,const block_q8_1*x,int columns,const uint32_t*grid) {
    if constexpr(Type==GGML_TYPE_Q2_0)return make_float2(V41GGUFDot<Type,1,LaneWidth>(gate,x,columns,grid),V41GGUFDot<Type,1,LaneWidth>(up,x,columns,grid));
    else {
        constexpr int qi=[] {
            if constexpr(Type==GGML_TYPE_IQ1_S)return QI1_S;
            else if constexpr(Type==GGML_TYPE_IQ1_M)return QI1_M;
            else if constexpr(Type==GGML_TYPE_IQ2_XXS||Type==GGML_TYPE_IQ2_XS||Type==GGML_TYPE_IQ2_S)return 16;
            else return ggml_cuda_type_traits<Type>::qi;
        }();
        constexpr int vdr=(Type==GGML_TYPE_IQ2_XXS||Type==GGML_TYPE_IQ2_XS||Type==GGML_TYPE_IQ2_S)?2:get_vdr_mmvq(Type);
        constexpr int step=vdr*LaneWidth/qi;
        const int lane=threadIdx.x%LaneWidth;
        float pg[4],pu[4];
#pragma unroll
        for(int w=0;w<4;++w) {
            const int tid=lane+LaneWidth*w,q=vdr*(tid%(qi/vdr));
            float g=0,u=0;
            for(int b=tid/(qi/vdr);b<columns/256;b+=step*4) {
                g+=V41BlockDot<Type>(gate,x,b,q,grid);
                u+=V41BlockDot<Type>(up,x,b,q,grid);
            }
            pg[w]=g;pu[w]=u;
        }
        float g=pg[0],u=pu[0];
#pragma unroll
        for(int w=1;w<4;++w){g+=pg[w];u+=pu[w];}
#pragma unroll
        for(int mask=LaneWidth/2;mask;mask>>=1){g+=__shfl_xor_sync(0xffffffff,g,mask,LaneWidth);u+=__shfl_xor_sync(0xffffffff,u,mask,LaneWidth);}
        return make_float2(g,u);
    }
}

__device__ const uint8_t *V41Weight(const uint8_t *const *weights,
        const int32_t *indices, int route, int experts, bool gate) {
    const int expert = indices[route];
    return expert >= 0 && expert < experts ? weights[2*expert+(gate ? 0 : 1)] : nullptr;
}
__device__ const uint8_t *V41Weight(const FastllmCudaMoeGGUFCacheView &view,
        const int32_t *, int route, int, bool gate) {
    const int slot = view.routeSlots[route];
    if (slot < 0) return nullptr;
    if (view.recordPointers)
        return view.recordPointers[slot] + (gate ? 0 : view.downOffset);
    const size_t offset = view.slotOffsets ? view.slotOffsets[slot] : size_t(slot)*view.recordStride;
    return view.records + offset + (gate ? 0 : view.downOffset);
}

__device__ int V41Route(const uint8_t *const *, int route) { return route; }
__device__ int V41Route(const FastllmCudaMoeGGUFCacheView &view, int route) {
    return view.routeMap ? view.routeMap[route] : route;
}

__device__ int V41WeightRow(const uint8_t *const *, int row, int, bool) { return row; }
__device__ int V41WeightRow(const FastllmCudaMoeGGUFCacheView &view, int row, int rows, bool gate) {
    // NUMA shards concatenate into [gate0, up0, gate1, up1, ...]. Preserve
    // the canonical output order without copying either projection matrix.
    return gate && view.numaGateType >= 0 ? 2*(row % (rows/2)) + row/(rows/2) : row;
}

// Compact only on the GPU: no device-to-host route-count synchronization.
// The launch grid is bounded by the public route count; inactive CTAs exit
// before loading a codebook or writing a zero expert matrix.
struct V41EpWeights {
    const uint8_t *const *weights;
    const int32_t *routes; // count, packed -> original, original -> packed
};
__device__ const uint8_t *V41Weight(V41EpWeights view, const int32_t *indices,
        int route, int experts, bool gate) {
    return V41Weight(view.weights, indices, view.routes[1+route], experts, gate);
}
__device__ int V41Route(V41EpWeights view, int route) { return view.routes[1+route]; }
__device__ int V41WeightRow(V41EpWeights, int row, int, bool) { return row; }
template<class Weights> __device__ bool V41ActiveRoute(Weights, int) { return true; }
__device__ bool V41ActiveRoute(V41EpWeights view, int route) { return route < view.routes[0]; }

__global__ void V41EpRoutes(const uint8_t *const *weights, const int32_t *indices,
        int32_t *routes, int count, int experts) {
    if (threadIdx.x) return;
    int active = 0;
    for (int r = 0; r < count; ++r) {
        const int e = indices[r];
        const bool local = e >= 0 && e < experts && weights[2*e] && weights[2*e+1];
        routes[1+count+r] = local ? active : -1;
        if (local) routes[1+active++] = r;
    }
    routes[0] = active;
}

__global__ void V41EpSwigluQ8(const __nv_bfloat16 *gateup, block_q8_1 *output,
        const int32_t *routes, const float *scores, int inter, float limit) {
    const int route = blockIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    if (route >= routes[0] || c >= inter) return;
    const auto *input = gateup + size_t(route)*2*inter;
    float gate = __bfloat162float(input[c]), up = __bfloat162float(input[c+inter]);
    if (limit > 0) {
        gate = fminf(gate, limit);
        up = fmaxf(-limit, fminf(up, limit));
    }
    float x = (gate/(1.f+expf(-gate)))*up;
    x *= scores[routes[1+route]];
    x = __bfloat162float(__float2bfloat16_rn(x));
    x = V41Fp8Boundary(x);
    V41StoreQ8(x, output[size_t(route)*(inter/32)+c/32], c%32);
}

template<ggml_type Type, bool Gate, class Weights, int RowsPerBlock = 8, int StaticColumns = 0, int VirtualWarps = 4, int LaneWidth = 32>
__global__ void V41GGUFProjection(const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices,
        int experts, int topk, int columns, int rows, size_t rowBytes) {
    if constexpr (StaticColumns) {
        columns = StaticColumns;
    }
    const int route = blockIdx.y;
    if (!V41ActiveRoute(weights, route)) return;
    const uint8_t *weight = V41Weight(weights, indices, route, experts, Gate);
    const int firstRow = blockIdx.x*RowsPerBlock + threadIdx.x/LaneWidth;
    // The decision is uniform within the CTA. Missing TP experts must still
    // clear every output row, but need neither the codebook nor activation.
    if (!weight) {
        for (int row = firstRow; row < rows && row < (blockIdx.x+1)*RowsPerBlock; row += 256/LaneWidth)
            if (threadIdx.x%LaneWidth == 0) output[size_t(route)*rows+row] = __float2bfloat16_rn(0.f);
        return;
    }
    const auto *x = input + size_t(Gate ? V41Route(weights, route)/topk : route)*(columns/32);
    extern __shared__ uint32_t activation[];
    constexpr int gridWords = (Type == GGML_TYPE_IQ1_S || Type == GGML_TYPE_IQ1_M || Type == GGML_TYPE_IQ2_S) ? 2048 :
        Type == GGML_TYPE_IQ2_XS ? 1024 : Type == GGML_TYPE_IQ2_XXS ? 512 : 2;
    __shared__ uint64_t gridStorage[gridWords/2];
    auto *iqGrid = reinterpret_cast<uint32_t *>(gridStorage);
    if constexpr (Type == GGML_TYPE_IQ1_S || Type == GGML_TYPE_IQ1_M) {
        for (int i = threadIdx.x; i < 2048; i += blockDim.x)
            iqGrid[i] = fastllm_iq1_grid_packed[i];
    } else if constexpr (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_S) {
        const uint64_t *source = Type == GGML_TYPE_IQ2_S ? iq2s_grid : Type == GGML_TYPE_IQ2_XS ? iq2xs_grid : iq2xxs_grid;
        for (int i = threadIdx.x; i < gridWords/2; i += blockDim.x) gridStorage[i] = source[i];
    }
    for (int i = threadIdx.x; i < columns/32*int(sizeof(block_q8_1))/4; i += blockDim.x)
        activation[i] = reinterpret_cast<const uint32_t *>(x)[i];
    __syncthreads();
    for (int row = firstRow; row < rows && row < (blockIdx.x+1)*RowsPerBlock; row += 256/LaneWidth) {
        const float value = V41GGUFDot<Type, VirtualWarps, LaneWidth>(weight+size_t(V41WeightRow(weights, row, rows, Gate))*rowBytes,
            reinterpret_cast<const block_q8_1 *>(activation), columns, iqGrid);
        if (threadIdx.x%LaneWidth == 0) output[size_t(route)*rows+row] = __float2bfloat16_rn(value);
    }
}

// Fuse both projections and weighted SwiGLU; FP8/Q8 conversion stays separate.
template<ggml_type Type, class Weights, int Tile = 8, int LaneWidth = 32, int Threads = 256, int StaticColumns = 0>
__device__ __forceinline__ void V41GGUFGateUpActBody(const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices, const float *scores,
        int experts, int topk, int columns, int inter, size_t rowBytes, float limit) {
    if constexpr (StaticColumns) columns = StaticColumns;
    const int route = blockIdx.y;
    const uint8_t *weight = V41Weight(weights, indices, route, experts, true);
    const auto *x = input + size_t(V41Route(weights, route)/topk)*(columns/32);
    extern __shared__ uint32_t activation[];
    constexpr int gridWords = (Type == GGML_TYPE_IQ1_S || Type == GGML_TYPE_IQ1_M || Type == GGML_TYPE_IQ2_S) ? 2048 :
        Type == GGML_TYPE_IQ2_XS ? 1024 : Type == GGML_TYPE_IQ2_XXS ? 512 : 2;
    __shared__ uint64_t gridStorage[gridWords/2];
    auto *iqGrid = reinterpret_cast<uint32_t *>(gridStorage);
    if constexpr (Type == GGML_TYPE_IQ1_S || Type == GGML_TYPE_IQ1_M) {
        for (int i = threadIdx.x; i < 2048; i += Threads) iqGrid[i] = fastllm_iq1_grid_packed[i];
    } else if constexpr (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ2_S) {
        const uint64_t *source = Type == GGML_TYPE_IQ2_S ? iq2s_grid : Type == GGML_TYPE_IQ2_XS ? iq2xs_grid : iq2xxs_grid;
        for (int i = threadIdx.x; i < gridWords/2; i += Threads) gridStorage[i] = source[i];
    }
    for (int i = threadIdx.x; i < columns/32*int(sizeof(block_q8_1))/4; i += Threads)
        activation[i] = reinterpret_cast<const uint32_t *>(x)[i];
    __syncthreads();
    for (int r = threadIdx.x/LaneWidth; r < Tile; r += Threads/LaneWidth) {
        const int row = blockIdx.x*Tile+r;
        float g=0, u=0;
        if (weight && row < inter) {
            const auto pair=V41PairDot<Type,LaneWidth>(
                weight+size_t(V41WeightRow(weights,row,2*inter,true))*rowBytes,
                weight+size_t(V41WeightRow(weights,row+inter,2*inter,true))*rowBytes,
                reinterpret_cast<const block_q8_1 *>(activation),columns,iqGrid);
            g=pair.x;u=pair.y;
        }
        if (threadIdx.x%LaneWidth==0) {
            g=__bfloat162float(__float2bfloat16_rn(g));
            u=__bfloat162float(__float2bfloat16_rn(u));
            if(limit>0){g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}
            float h=(g/(1.0f+expf(-g)))*u;
            if(scores)h*=scores[route];
            if(row<inter) output[size_t(route)*inter+row]=__float2bfloat16_rn(h);
        }
    }
}

// Keep unbounded kernels for the formats whose existing register allocation
// already performs well. IQ2 uses explicit residency targets measured on SM80.
template<ggml_type Type, class Weights, int Tile, int LaneWidth, int Threads, int StaticColumns>
__global__ void V41GGUFGateUpAct(const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices, const float *scores,
        int experts, int topk, int columns, int inter, size_t rowBytes, float limit) {
    V41GGUFGateUpActBody<Type,Weights,Tile,LaneWidth,Threads,StaticColumns>(
        input,output,weights,indices,scores,experts,topk,columns,inter,rowBytes,limit);
}

template<ggml_type Type, class Weights, int Tile, int Threads, int StaticColumns, int MinBlocks>
__global__ __launch_bounds__(Threads,MinBlocks)
void V41GGUFGateUpActBound(const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices, const float *scores,
        int experts, int topk, int columns, int inter, size_t rowBytes, float limit) {
    V41GGUFGateUpActBody<Type,Weights,Tile,32,Threads,StaticColumns>(
        input,output,weights,indices,scores,experts,topk,columns,inter,rowBytes,limit);
}

__global__ void V41GGUFReduce(const __nv_bfloat16 *partial, float *output,
        const int32_t *indices, int hidden, int topk, const int32_t *inverse = nullptr,
        int accumulate = 0) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    const int base = blockIdx.y*topk;
    int order[32];
    for (int k = 0; k < topk; ++k) {
        int pos = k;
        while (pos && indices[base+order[pos-1]] > indices[base+k]) {
            order[pos] = order[pos-1]; --pos;
        }
        order[pos] = k;
    }
    float sum = 0;
    for (int k = 0; k < topk; ++k) {
        const int route = inverse ? inverse[base+order[k]] : base+order[k];
        const float value = route >= 0 ? __bfloat162float(partial[size_t(route)*hidden+c]) : 0.f;
        sum = __fadd_rn(sum, value);
    }
    // The caller adds the shared expert in FP32 before the BF16 HC boundary.
    float *dst = output + size_t(blockIdx.y)*hidden+c;
    *dst = accumulate ? __fadd_rn(*dst, sum) : sum;
}

// Keep the reference path available for numeric/performance regression checks.
// Small matrices retain the existing tile choices.
bool V41UseTunedGGUF(int hidden, int inter) {
    const char *legacy = std::getenv("FASTLLM_V41_GGUF_LEGACY");
    return hidden >= 1024 && inter >= 1024 && (!legacy || legacy[0] != '1');
}

template<class Weights>
void ProjectGateActivation(int type, const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices, const float *scores,
        int experts, int topk, int routes, int hidden, int inter, float limit) {
    const size_t shared = size_t(hidden/32)*sizeof(block_q8_1);
    switch(type) {
#define GATE_ACT_ARGS(Type) input,output,weights,indices,scores,experts,topk,hidden,inter,ggml_row_size(GGML_TYPE_##Type,hidden),limit
#define LAUNCH_GATE_ACT(Type, Static) \
        if constexpr (GGML_TYPE_##Type == GGML_TYPE_IQ2_XXS || GGML_TYPE_##Type == GGML_TYPE_IQ2_XS || GGML_TYPE_##Type == GGML_TYPE_IQ2_S) { \
            constexpr int blocks = threads == 128 ? 8 : 4; \
            V41GGUFGateUpActBound<GGML_TYPE_##Type,Weights,tile,threads,Static,blocks><<<dim3((inter+tile-1)/tile,routes),threads,shared,cudaStreamPerThread>>>(GATE_ACT_ARGS(Type)); \
        } else { \
            V41GGUFGateUpAct<GGML_TYPE_##Type,Weights,tile,32,threads,Static><<<dim3((inter+tile-1)/tile,routes),threads,shared,cudaStreamPerThread>>>(GATE_ACT_ARGS(Type)); \
        }
#define GATE_ACT(Type) case GGML_TYPE_##Type: { \
        constexpr int threads = GGML_TYPE_##Type == GGML_TYPE_IQ2_XXS || GGML_TYPE_##Type == GGML_TYPE_IQ2_XS || GGML_TYPE_##Type == GGML_TYPE_Q3_K ? 128 : 256; \
        constexpr int tile = GGML_TYPE_##Type == GGML_TYPE_IQ2_S || GGML_TYPE_##Type == GGML_TYPE_Q4_K ? 16 : 8; \
        if (hidden == 5120) { LAUNCH_GATE_ACT(Type, 5120); } else { LAUNCH_GATE_ACT(Type, 0); } break; }
        GATE_ACT(Q2_0) GATE_ACT(Q2_K) GATE_ACT(Q3_K) GATE_ACT(Q4_K) GATE_ACT(IQ1_S) GATE_ACT(IQ1_M)
        GATE_ACT(IQ2_XXS) GATE_ACT(IQ2_XS) GATE_ACT(IQ2_S) GATE_ACT(IQ3_XXS)
#undef GATE_ACT
#undef LAUNCH_GATE_ACT
#undef GATE_ACT_ARGS
    }
}

template<bool Gate, class Weights>
void Project(int type, const block_q8_1 *input, __nv_bfloat16 *output,
        Weights weights, const int32_t *indices,
        int experts, int topk, int routes, int columns, int rows, bool tuned = false) {
    const size_t shared = size_t(columns/32)*sizeof(block_q8_1);
    if constexpr (!Gate) {
        if (tuned) {
            // A quarter warp keeps the 128 logical reference lanes, while a
            // fixed column count removes the short-loop bounds and tail work.
            if (columns == 2304) {
                switch (type) {
#define DOWN_2304(Type) case GGML_TYPE_##Type: \
                    V41GGUFProjection<GGML_TYPE_##Type,false,Weights,32,2304,16,8><<<dim3((rows+31)/32,routes),256,shared,cudaStreamPerThread>>>( \
                        input,output,weights,indices,experts,topk,columns,rows,ggml_row_size(GGML_TYPE_##Type,columns)); return;
                    DOWN_2304(IQ2_XXS) DOWN_2304(IQ2_XS) DOWN_2304(IQ2_S) DOWN_2304(IQ3_XXS)
#undef DOWN_2304
                }
            }
            switch(type) {
#define DOWN(Type) case GGML_TYPE_##Type: { \
                constexpr bool full = GGML_TYPE_##Type == GGML_TYPE_Q2_0 || GGML_TYPE_##Type == GGML_TYPE_IQ3_XXS; \
                constexpr int lanes = full ? 32 : 16; \
                constexpr int tile = GGML_TYPE_##Type == GGML_TYPE_Q3_K || GGML_TYPE_##Type == GGML_TYPE_Q4_K ? 16 : 32; \
                V41GGUFProjection<GGML_TYPE_##Type, false, Weights, tile, 0, full ? 4 : 8, lanes><<<dim3((rows+tile-1)/tile,routes),256,shared,cudaStreamPerThread>>>( \
                    input,output,weights,indices,experts,topk,columns,rows,ggml_row_size(GGML_TYPE_##Type,columns)); return; }
                DOWN(Q2_0) DOWN(Q2_K) DOWN(Q3_K) DOWN(Q4_K) DOWN(IQ1_S) DOWN(IQ1_M)
                DOWN(IQ2_XXS) DOWN(IQ2_XS) DOWN(IQ2_S) DOWN(IQ3_XXS)
#undef DOWN
            }
        }
    }
    if constexpr (Gate) {
        // Amortize the staged IQ1_S codebook over four rows per warp.
        if (type == GGML_TYPE_IQ1_S && rows >= 1024) {
            constexpr int tile = 32;
            V41GGUFProjection<GGML_TYPE_IQ1_S, Gate, Weights, tile>
                <<<dim3((rows+tile-1)/tile, routes), 256, shared, cudaStreamPerThread>>>(
                    input, output, weights, indices, experts, topk, columns, rows,
                    ggml_row_size(GGML_TYPE_IQ1_S, columns));
            return;
        }
    }
    if constexpr (!Gate) {
        // The V4.1 down projection reuses each Q8 activation across 16 rows.
        // Keep the four virtual-warp sums in their original order.
        if (type == GGML_TYPE_Q3_K && columns == 2304 && rows >= 1024) {
            constexpr int tile = 16;
            V41GGUFProjection<GGML_TYPE_Q3_K, false, Weights, tile, 2304>
                <<<dim3((rows+tile-1)/tile, routes), 256, shared, cudaStreamPerThread>>>(
                    input, output, weights, indices, experts, topk, columns, rows,
                    ggml_row_size(GGML_TYPE_Q3_K, columns));
            return;
        }
    }
    // The wider IQ tiles are tuned for resident weights, including EP.
    // Hybrid cache routes keep smaller tiles for low expert occupancy.
    constexpr bool resident = std::is_same_v<Weights, const uint8_t *const *> ||
        std::is_same_v<Weights, V41EpWeights>;
    switch (type) {
#define PROJECT(Type) case GGML_TYPE_##Type: { \
        constexpr int tile = resident && (GGML_TYPE_##Type == GGML_TYPE_IQ2_XXS || GGML_TYPE_##Type == GGML_TYPE_IQ1_M) ? 32 : \
            (GGML_TYPE_##Type == GGML_TYPE_IQ1_S || GGML_TYPE_##Type == GGML_TYPE_IQ1_M) ? 16 : 8; \
        V41GGUFProjection<GGML_TYPE_##Type, Gate, Weights, tile><<<dim3((rows+tile-1)/tile, routes), 256, shared, cudaStreamPerThread>>>( \
            input, output, weights, indices, experts, topk, columns, rows, \
            ggml_row_size(GGML_TYPE_##Type, columns)); break; }
        PROJECT(Q2_0) PROJECT(Q2_K) PROJECT(Q3_K) PROJECT(Q4_K) PROJECT(IQ1_S) PROJECT(IQ1_M)
        PROJECT(IQ2_XXS) PROJECT(IQ2_XS) PROJECT(IQ2_S) PROJECT(IQ3_XXS)
#undef PROJECT
    }
}
}

bool FastllmCudaRunV41GGUF(
        const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &activation,
        fastllm::Data &workspace, fastllm::Data &output, const void *weightPointers,
        const int32_t *indices, const float *scores, int gateType, int downType,
        int hidden, int inter, int experts, int topk, float swigluLimit,
        bool compactEpRoutes, bool accumulate) {
    const int rows = input.dims[0], routes = rows*topk;
    auto *qInput = static_cast<block_q8_1 *>(workspace.cudaData);
    auto *qGate = qInput + size_t(rows)*(hidden/32);
    auto *partial = reinterpret_cast<__nv_bfloat16 *>(qGate+size_t(routes)*(inter/32));
    const auto *weights = static_cast<const uint8_t *const *>(weightPointers);
    const size_t baseBytes = (size_t(rows)*(hidden/32) + size_t(routes)*(inter/32))*sizeof(block_q8_1) +
        size_t(routes)*hidden*sizeof(__nv_bfloat16);
    const size_t offset = (baseBytes+255)&~size_t(255);
    void *groupedScratch = static_cast<char *>(workspace.cudaData)+offset;
    V41GGUFQuantize<true><<<dim3((hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData), qInput, hidden);
    if (compactEpRoutes) {
        auto *routeMap = static_cast<int32_t *>(groupedScratch);
        V41EpRoutes<<<1, 32, 0, cudaStreamPerThread>>>(weights, indices, routeMap, routes, experts);
        const V41EpWeights view{weights, routeMap};
        Project<true>(gateType, qInput, static_cast<__nv_bfloat16 *>(gate.cudaData),
            view, indices, experts, topk, routes, hidden, 2*inter);
        // Fuse the two activation kernels but retain every BF16/FP8 rounding
        // boundary, including the score multiplication before FP8 quantization.
        V41EpSwigluQ8<<<dim3((inter+255)/256, routes), 256, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(gate.cudaData), qGate, routeMap, scores, inter, swigluLimit);
        Project<false>(downType, qGate, partial, view, indices, experts, topk, routes, inter, hidden);
        V41GGUFReduce<<<dim3((hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
            partial, static_cast<float *>(output.cudaData), indices, hidden, topk, routeMap+1+routes);
        return cudaGetLastError() == cudaSuccess;
    }
    const bool grouped = workspace.GetBytes() > offset;
    const bool tuned = !grouped && V41UseTunedGGUF(hidden, inter);
    if (tuned) {
        ProjectGateActivation(gateType,qInput,static_cast<__nv_bfloat16 *>(gate.cudaData),
            weights,indices,scores,experts,topk,routes,hidden,inter,swigluLimit);
        V41GGUFQuantize<true><<<dim3((inter+255)/256,routes),256,0,cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(gate.cudaData),qGate,inter);
    } else {
        if (grouped) {
            if (!FastllmCudaV41GGUFGroupedProjection(true, gateType, qInput, gate.cudaData,
                weights, indices, groupedScratch, rows, hidden, inter, experts, topk)) return false;
        } else Project<true>(gateType, qInput, static_cast<__nv_bfloat16 *>(gate.cudaData),
            weights, indices, experts, topk, routes, hidden, 2*inter);
        if (!FastllmCudaDeepSeekV41SharedSwigluQuantized(gate, swigluLimit, activation, scores)) return false;
        V41GGUFQuantize<false><<<dim3((inter+255)/256, routes), 256, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(activation.cudaData), qGate, inter);
    }
    if (grouped) {
        if (!FastllmCudaV41GGUFGroupedProjection(false, downType, qGate, partial,
            weights, indices, groupedScratch, rows, hidden, inter, experts, topk)) return false;
    } else Project<false>(downType, qGate, partial, weights, indices, experts, topk, routes, inter, hidden, tuned);
    V41GGUFReduce<<<dim3((hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
        partial, static_cast<float *>(output.cudaData), indices, hidden, topk,
        nullptr, accumulate ? 1 : 0);
    return cudaGetLastError() == cudaSuccess;
}

namespace {
size_t V41CacheWorkspaceBytes(int rows, int routes, int hidden, int inter) {
    return (size_t(rows)*(hidden/32) + size_t(routes)*(inter/32))*sizeof(block_q8_1) +
        size_t(routes)*(2*size_t(inter)+hidden)*sizeof(__nv_bfloat16) + size_t(routes)*sizeof(float);
}

__global__ void V41CachePartials(const __nv_bfloat16 *partial, float *output,
        int hidden, int count, const int32_t *routeMap, const int32_t *validSlots) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) {
        const int route = i / hidden;
        if (validSlots && validSlots[route] < 0) return;
        const size_t destination = size_t(routeMap ? routeMap[route] : route)*hidden + i%hidden;
        output[destination] = __bfloat162float(partial[i]);
    }
}
__global__ void V41CacheScores(const float *scores, float *selected,
        const int32_t *routeMap, int routes) {
    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r < routes) selected[r] = scores[routeMap[r]];
}
}

bool FastllmCudaV41GGUFSupported(int gateType, int downType) {
    const auto supported = [](int type) {
        return type == GGML_TYPE_Q2_0 || type == GGML_TYPE_Q2_K ||
            type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ||
            type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ||
            type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
            type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS;
    };
    return supported(gateType) && supported(downType);
}
size_t FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(int hidden, int inter) {
    if (hidden <= 0 || inter <= 0 || hidden%256 || inter%256) return 0;
    // Decode and DSpark verification, at most eight rows and sixteen routes/row.
    return V41CacheWorkspaceBytes(8, 8*16, hidden, inter);
}
bool FastllmCudaMoeV41CanonicalGGUFCacheCompute(
        const fastllm::Data &input, fastllm::Data &activation,
        const FastllmCudaMoeGGUFCacheView &view, const float *scores,
        int topk, float swigluLimit, float *perExpert) {
    if (input.dataType != fastllm::BFLOAT16 || input.dataDevice != fastllm::CUDA ||
        !input.cudaData || input.dims.size() != 2 || input.dims[0] < 1 || input.dims[0] > 8 ||
        input.dims[1] != view.hidden || view.hidden <= 0 || view.inter <= 0 ||
        view.hidden%256 || view.inter%256 || topk < 1 || topk > 16 ||
        !FastllmCudaV41GGUFSupported(view.gateType, view.downType) ||
        (!view.records && !view.recordPointers) || !view.routeSlots || !view.workspace || !scores || !perExpert ||
        view.activeRouteCount || (view.routeMap && (view.routeCount < 1 || view.routeCount > input.dims[0]*topk)) ||
        !((view.numaGateType < 0 && view.numaDownType < 0) ||
          (view.numaGateType == view.gateType && view.numaDownType == view.downType)))
        return false;
    const int rows = input.dims[0], routes = view.routeMap ? view.routeCount : rows*topk;
    const int hidden = view.hidden, inter = view.inter;
    const size_t bytes = V41CacheWorkspaceBytes(rows, routes, hidden, inter);
    if (view.workspaceBytes < bytes) return false;
    auto *qInput = static_cast<block_q8_1 *>(view.workspace);
    auto *qGate = qInput + size_t(rows)*(hidden/32);
    auto *gateData = reinterpret_cast<__nv_bfloat16 *>(qGate + size_t(routes)*(inter/32));
    auto *partial = gateData + size_t(routes)*2*inter;
    fastllm::Data gate(fastllm::BFLOAT16, {routes, 2*inter}, fastllm::CUDA, gateData);
    gate.dataDeviceIds = input.dataDeviceIds;
    if (!view.q8InputPrepared)
        V41GGUFQuantize<true><<<dim3((hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(input.cudaData), qInput, hidden);
    if (view.routeMap) {
        auto *selectedScores = reinterpret_cast<float *>(partial + size_t(routes)*hidden);
        V41CacheScores<<<(routes+255)/256, 256, 0, cudaStreamPerThread>>>(
            scores, selectedScores, view.routeMap, routes);
        scores = selectedScores;
    }
    const bool tuned = V41UseTunedGGUF(hidden, inter);
    if (tuned) {
        ProjectGateActivation(view.gateType,qInput,gateData,view,nullptr,scores,0,topk,routes,hidden,inter,swigluLimit);
        V41GGUFQuantize<true><<<dim3((inter+255)/256,routes),256,0,cudaStreamPerThread>>>(gateData,qGate,inter);
    } else {
        Project<true>(view.gateType, qInput, gateData, view, nullptr, 0, topk, routes, hidden, 2*inter);
        if (!FastllmCudaDeepSeekV41SharedSwigluQuantized(gate, swigluLimit, activation, scores)) return false;
        V41GGUFQuantize<false><<<dim3((inter+255)/256, routes), 256, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(activation.cudaData), qGate, inter);
    }
    Project<false>(view.downType, qGate, partial, view, nullptr, 0, topk, routes, inter, hidden, tuned);
    V41CachePartials<<<(routes*hidden+255)/256, 256, 0, cudaStreamPerThread>>>(
        partial, perExpert, hidden, routes*hidden, view.routeMap,
        view.preserveMissing ? view.routeSlots : nullptr);
    return cudaGetLastError() == cudaSuccess;
}

// Defined in fastllm-cuda.cu: H2D upload through a pinned staging ring, which
// returns false when the tensor does not fit the ring. Declared at global scope
// because the definition is a plain C++ symbol shared with cudadevice.cpp.
bool FastllmCudaUploadHostToDevicePinnedStaged(void *dst, const void *src, size_t size, void *stream);

namespace {
// Pageable host experts only; NUMA/disk keep their own registered pipelines.
const uint8_t *V41HostStreamBytes(const fastllm::Data &w) {
    if (w.dataType != fastllm::DATA_GGUF_FORMAT || w.isDiskWeight) return nullptr;
    return static_cast<const uint8_t *>(w.cpuData);
}

double V41HostStreamNowMs() {
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() * 1e-3;
}

struct V41HostStreamBuffer {
    void *ptr = nullptr;
    ~V41HostStreamBuffer() { if (ptr != nullptr) FastllmCudaFree(ptr); }
};

bool V41HostStreamDebug() {
    static const bool enabled = std::getenv("FASTLLM_PROFILE_HOST_MOE") != nullptr;
    return enabled;
}

// Every rejection is silent by default: the CPU MoE path stays authoritative.
// With FASTLLM_PROFILE_HOST_MOE set, report the first few reasons verbatim.
void V41HostStreamReject(const char *why, const char *extra = nullptr) {
    if (!V41HostStreamDebug()) return;
    static std::atomic<int> printed{0};
    if (printed.fetch_add(1) >= 8) return;
    printf("[fastllm-host-moe] skip: %s%s%s\n", why, extra ? " " : "",
        extra ? extra : "");
    fflush(stdout);
}
}

bool FastllmCudaRunV41GGUFHostStream(
        const fastllm::Data &input, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *indices, const float *scores,
        int topk, float swigluLimit, size_t budgetBytes) {
    const auto reject = [](const char *why, const char *extra = nullptr) {
        V41HostStreamReject(why, extra);
        return false;
    };
    if (!weights || !indices || !scores || topk < 1 || topk > 32 ||
        weightsBatch < 2 || input.dims.size() != 2) return reject("bad routes/weights");
    if (input.dataType != fastllm::BFLOAT16 || !input.cudaData)
        return reject("input is not BF16 on CUDA");
    const int experts = weightsBatch / 2;
    const int rows = input.dims[0], hidden = input.dims[1], routes = rows*topk;
    if (experts <= 0 || rows <= 0 || hidden <= 0 || hidden%256 != 0 || hidden > 24576)
        return reject("unsupported hidden size");
    const int device = FastllmCudaGetDevice();
    if (input.dataDevice != fastllm::CUDA || input.dataDeviceIds != std::vector<int>{device})
        return reject("input is not on the current device");

    // One launch serves the whole layer, so every routed expert of it must share
    // the same two formats. Anything else keeps the per-expert CPU order.
    int gateType = -1, downType = -1, inter = 0;
    for (int e = 0; e < experts; ++e) {
        const fastllm::Data *gate = weights[2*e], *down = weights[2*e+1];
        if (!gate || !down) return reject("missing expert weight");
        if (!V41HostStreamBytes(*gate) || !V41HostStreamBytes(*down)) {
            char extra[96];
            snprintf(extra, sizeof(extra), "expert=%d type=%d/%d host=%d/%d", e,
                gate->dataType, down->dataType, gate->cpuData != nullptr,
                down->cpuData != nullptr);
            return reject("expert weights are not host GGUF", extra);
        }
        if (gateType < 0) {
            gateType = gate->ggmlType;
            downType = down->ggmlType;
            inter = gate->dims.empty() ? 0 : gate->dims[0] / 2;
        } else if (gate->ggmlType != gateType || down->ggmlType != downType) {
            return reject("mixed ggml types inside one layer");
        }
        if (inter <= 0 || gate->dims.size() != 2 || gate->dims[0] != 2*inter ||
            gate->dims[1] != hidden || down->dims.size() != 2 ||
            down->dims[0] != hidden || down->dims[1] != inter) {
            char extra[96];
            snprintf(extra, sizeof(extra), "gate=%dx%d down=%dx%d inter=%d hidden=%d",
                gate->dims.empty() ? 0 : gate->dims[0],
                gate->dims.size() < 2 ? 0 : gate->dims[1],
                down->dims.empty() ? 0 : down->dims[0],
                down->dims.size() < 2 ? 0 : down->dims[1], inter, hidden);
            return reject("unexpected expert shapes", extra);
        }
    }
    if (inter%256 != 0 || inter > 24576) return reject("unsupported intermediate size");
    if (!FastllmCudaV41GGUFSupported(gateType, downType)) {
        char extra[64];
        snprintf(extra, sizeof(extra), "gate=%d down=%d", gateType, downType);
        return reject("ggml format not supported on CUDA", extra);
    }
    const size_t gateBytes = ggml_row_size(static_cast<ggml_type>(gateType), hidden) * size_t(2*inter);
    const size_t downBytes = ggml_row_size(static_cast<ggml_type>(downType), inter) * size_t(hidden);
    if (gateBytes == 0 || downBytes == 0) return reject("bad row size");

    std::vector<int> counts(experts, 0);
    for (int i = 0; i < routes; ++i) {
        const int expert = indices[i];
        if (expert < 0 || expert >= experts) return reject("route out of range");
        counts[expert]++;
    }
    std::vector<int> order;
    order.reserve(experts);
    for (int e = 0; e < experts; ++e)
        if (counts[e] > 0) order.push_back(e);
    if (order.empty()) return reject("no routed expert");

    const size_t expertBytes = gateBytes + downBytes;
    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) return reject("cudaMemGetInfo failed");
    // The previous layer's arenas are idle in the allocator pool, not free in
    // the driver. This layer asks for the same shapes, so it reuses them; budget
    // against driver-free plus pooled-idle instead of driver-free alone.
    const size_t pooledBytes = FastllmCudaGetPooledIdleBytes(device);
    const size_t usableBytes = freeBytes + pooledBytes;
    const size_t reserve = 256ull << 20, minArena = 64ull << 20;
    if (usableBytes <= reserve) {
        char extra[128];
        snprintf(extra, sizeof(extra), "free=%.0fMB pooled=%.0fMB reserve=%.0fMB",
            freeBytes / 1048576.0, pooledBytes / 1048576.0, reserve / 1048576.0);
        return reject("too little free VRAM", extra);
    }
    size_t available = usableBytes - reserve;
    if (budgetBytes > 0 && budgetBytes < available) available = budgetBytes;
    if (available <= minArena || expertBytes > available - minArena) {
        char extra[160];
        snprintf(extra, sizeof(extra), "free=%.0fMB pooled=%.0fMB available=%.0fMB expert=%.1fMB",
            freeBytes / 1048576.0, pooledBytes / 1048576.0,
            available / 1048576.0, expertBytes / 1048576.0);
        return reject("not enough free VRAM for experts", extra);
    }

    // Rows are blocked as well: the reduction/gate scratch of one 4096-row chunk
    // is more than a card with mostly dense weights can spare, while the expert
    // arena has to grow. Both curves bound the resident set independently.
    const auto blockScratchBytes = [&](int blockRows) -> size_t {
        const size_t blockRoutes = size_t(blockRows) * size_t(topk);
        size_t bytes = (size_t(blockRows)*(hidden/32) + blockRoutes*(inter/32))*sizeof(block_q8_1) +
            blockRoutes*hidden*sizeof(__nv_bfloat16);
        bytes += blockRoutes*size_t(2*inter)*sizeof(__nv_bfloat16);
        bytes += blockRoutes*size_t(inter)*sizeof(__nv_bfloat16);
        bytes += FastllmCudaV41GGUFGroupedWorkspaceBytes(
            gateType, downType, blockRows, hidden, inter, experts, topk);
        return bytes;
    };
    const int candidates[] = {4096, 2048, 1024, 512, 256, 128, 64};
    int blockRows = 0;
    for (int candidate : candidates) {
        if (candidate > rows) continue;
        if (blockScratchBytes(candidate) + minArena <= available) {
            blockRows = candidate;
            break;
        }
    }
    if (blockRows <= 0) {
        char extra[160];
        snprintf(extra, sizeof(extra), "free=%.0fMB pooled=%.0fMB available=%.0fMB scratch4096=%.0fMB",
            freeBytes / 1048576.0, pooledBytes / 1048576.0,
            available / 1048576.0, blockScratchBytes(4096) / 1048576.0);
        return reject("no row block fits the free VRAM", extra);
    }
    size_t arenaBudget = available - blockScratchBytes(blockRows);
    if (arenaBudget < expertBytes) {
        char extra[128];
        snprintf(extra, sizeof(extra), "block=%d arena=%.0fMB expert=%.1fMB",
            blockRows, arenaBudget / 1048576.0, expertBytes / 1048576.0);
        return reject("arena smaller than one expert", extra);
    }

    const size_t blockBase = (size_t(blockRows)*(hidden/32) + size_t(blockRows)*topk*(inter/32))*sizeof(block_q8_1) +
        size_t(blockRows)*topk*hidden*sizeof(__nv_bfloat16);
    const size_t workspaceOffset = (blockBase + 255) & ~size_t(255);
    const size_t groupedBytes = FastllmCudaV41GGUFGroupedWorkspaceBytes(
        gateType, downType, blockRows, hidden, inter, experts, topk);
    const int blockRoutes = blockRows * topk;
    fastllm::Data gate, activation, workspace;
    fastllm_gguf_moe::AllocateTensor(gate, fastllm::BFLOAT16, {blockRoutes, 2*inter}, device);
    fastllm_gguf_moe::AllocateTensor(activation, fastllm::BFLOAT16, {blockRoutes, inter}, device);
    fastllm_gguf_moe::AllocateTensor(workspace, fastllm::INT8,
        {int(workspaceOffset + groupedBytes)}, device);

    if (output.dataType != fastllm::FLOAT32 || output.dataDevice != fastllm::CUDA ||
        output.dataDeviceIds != std::vector<int>{device}) {
        output.FreeSpace();
        output.dataType = fastllm::FLOAT32;
        output.UpdateUnitSize();
        output.dataDevice = fastllm::CUDA;
        output.dataDeviceIds = {device};
    }
    output.Resize({rows, hidden});
    output.Allocate(false);
    if (output.cudaData == nullptr) return false;
    // Groups and row blocks accumulate into this buffer; zero it once here.
    FastllmCudaMemset0(output.cudaData, output.GetBytes());

    // Route-descending groups keep the grouped tiles dense; ascending ids inside
    // a group keep the reduction order stable for a given request.
    std::stable_sort(order.begin(), order.end(), [&counts](int a, int b) {
        return counts[a] != counts[b] ? counts[a] > counts[b] : a < b;
    });
    // Half of the arena per group: the other half receives the next group while
    // this one computes.
    const size_t groupBudget = std::max(expertBytes, arenaBudget / 2);
    std::vector<std::vector<int> > batches(1);
    size_t batchBytes = 0, arenaBytes = 0;
    for (int e : order) {
        if (!batches.back().empty() && batchBytes + expertBytes > groupBudget) {
            arenaBytes = std::max(arenaBytes, batchBytes);
            batches.emplace_back();
            batchBytes = 0;
        }
        batches.back().push_back(e);
        batchBytes += expertBytes;
    }
    arenaBytes = std::max(arenaBytes, batchBytes);

    V41HostStreamBuffer arenas[2], tables[2], indexBuffer, scoreBuffer;
    arenas[0].ptr = FastllmCudaMalloc(arenaBytes);
    arenas[1].ptr = FastllmCudaMalloc(arenaBytes);
    // One weight table per arena: the next group rebuilds its own while the
    // current group is still reading the other.
    tables[0].ptr = FastllmCudaMalloc(sizeof(void*) * size_t(experts) * 2);
    tables[1].ptr = FastllmCudaMalloc(sizeof(void*) * size_t(experts) * 2);
    indexBuffer.ptr = FastllmCudaMalloc(size_t(routes) * sizeof(int32_t));
    scoreBuffer.ptr = FastllmCudaMalloc(size_t(routes) * sizeof(float));
    if (!arenas[0].ptr || !arenas[1].ptr || !tables[0].ptr || !tables[1].ptr ||
        !indexBuffer.ptr || !scoreBuffer.ptr) return false;
    FastllmCudaCopyFromHostToDevice(indexBuffer.ptr, const_cast<int32_t *>(indices),
        size_t(routes) * sizeof(int32_t));
    FastllmCudaCopyFromHostToDevice(scoreBuffer.ptr, const_cast<float *>(scores),
        size_t(routes) * sizeof(float));
    const int32_t *deviceIndex = static_cast<const int32_t *>(indexBuffer.ptr);
    const float *deviceScore = static_cast<const float *>(scoreBuffer.ptr);

    struct V41HostStreamPipeline {
        cudaStream_t copy = nullptr;
        cudaEvent_t uploaded[2] = {nullptr, nullptr};
        cudaEvent_t computed[2] = {nullptr, nullptr};
        bool ok = false;
        V41HostStreamPipeline() {
            ok = cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking) == cudaSuccess;
            for (int i = 0; ok && i < 2; ++i) {
                ok = cudaEventCreateWithFlags(&uploaded[i], cudaEventDisableTiming) == cudaSuccess &&
                     cudaEventCreateWithFlags(&computed[i], cudaEventDisableTiming) == cudaSuccess;
            }
        }
        ~V41HostStreamPipeline() {
            if (copy != nullptr) cudaStreamDestroy(copy);
            for (int i = 0; i < 2; ++i) {
                if (uploaded[i] != nullptr) cudaEventDestroy(uploaded[i]);
                if (computed[i] != nullptr) cudaEventDestroy(computed[i]);
            }
        }
    } pipeline;
    if (!pipeline.ok) return reject("pinned copy stream unavailable");

    std::vector<const void *> hostTables[2] = {
        std::vector<const void *>(size_t(experts) * 2, nullptr),
        std::vector<const void *>(size_t(experts) * 2, nullptr)};
    int slotGroup[2] = {-1, -1};
    size_t uploadedBytes = 0;
    double uploadMs = 0.0;
    const auto issueGroup = [&](size_t b) {
        const int slot = int(b & 1);
        auto &pointer = hostTables[slot];
        if (slotGroup[slot] >= 0) {
            // That group's compute still reads this arena: wait, then drop the
            // stale entries before the table copy below.
            cudaStreamWaitEvent(pipeline.copy, pipeline.computed[slot], 0);
            for (int e : batches[slotGroup[slot]]) pointer[2*e] = pointer[2*e+1] = nullptr;
        }
        const double started = V41HostStreamDebug() ? V41HostStreamNowMs() : 0.0;
        auto *cursor = static_cast<uint8_t *>(arenas[slot].ptr);
        for (int e : batches[b]) {
            const uint8_t *gateSource = V41HostStreamBytes(*weights[2*e]);
            const uint8_t *downSource = V41HostStreamBytes(*weights[2*e+1]);
            // Pinned staging keeps the host from blocking on the whole tensor,
            // so the copies overlap the previous group's kernels.
            if (!FastllmCudaUploadHostToDevicePinnedStaged(cursor, gateSource, gateBytes, pipeline.copy))
                cudaMemcpyAsync(cursor, gateSource, gateBytes, cudaMemcpyHostToDevice, pipeline.copy);
            pointer[2*e] = cursor;
            cursor += gateBytes;
            if (!FastllmCudaUploadHostToDevicePinnedStaged(cursor, downSource, downBytes, pipeline.copy))
                cudaMemcpyAsync(cursor, downSource, downBytes, cudaMemcpyHostToDevice, pipeline.copy);
            pointer[2*e+1] = cursor;
            cursor += downBytes;
        }
        cudaMemcpyAsync(tables[slot].ptr, pointer.data(), pointer.size() * sizeof(void*),
            cudaMemcpyHostToDevice, pipeline.copy);
        cudaEventRecord(pipeline.uploaded[slot], pipeline.copy);
        slotGroup[slot] = int(b);
        uploadedBytes += size_t(cursor - static_cast<uint8_t *>(arenas[slot].ptr));
        if (V41HostStreamDebug()) uploadMs += V41HostStreamNowMs() - started;
    };

    issueGroup(0);
    if (batches.size() > 1) issueGroup(1);
    for (size_t b = 0; b < batches.size(); ++b) {
        const int slot = int(b & 1);
        cudaStreamWaitEvent(cudaStreamPerThread, pipeline.uploaded[slot], 0);
        for (int rowStart = 0; rowStart < rows; rowStart += blockRows) {
            const int count = std::min(blockRows, rows - rowStart);
            fastllm::Data inputView, outputView;
            inputView.FakeFrom(input, size_t(rowStart) * hidden * sizeof(uint16_t));
            inputView.Resize({count, hidden});
            outputView.FakeFrom(output, size_t(rowStart) * hidden * sizeof(float));
            outputView.Resize({count, hidden});
            const size_t routeOffset = size_t(rowStart) * topk;
            const bool ok = FastllmCudaRunV41GGUF(inputView, gate, activation, workspace,
                outputView, tables[slot].ptr, deviceIndex + routeOffset, deviceScore + routeOffset,
                gateType, downType, hidden, inter, experts, topk, swigluLimit, false, true);
            if (!ok) return false;
        }
        cudaEventRecord(pipeline.computed[slot], cudaStreamPerThread);
        // Two groups ahead: that slot's arena has just been released by the
        // compute recorded above.
        if (b + 2 < batches.size()) issueGroup(b + 2);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    if (V41HostStreamDebug()) {
        printf("[fastllm-host-moe] groups=%zu block=%d arena=%.0fMB uploaded=%.0fMB upload=%.1fms\n",
            batches.size(), blockRows, arenaBytes / 1048576.0, uploadedBytes / 1048576.0, uploadMs);
        fflush(stdout);
    }
    return true;
}
