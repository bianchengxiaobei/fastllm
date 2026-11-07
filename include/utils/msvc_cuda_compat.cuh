#pragma once

// The CUDA kernels in this project use the `__is_same` builtin provided by
// GCC/Clang, which nvcc's MSVC front end does not implement. Map it onto the
// libcu++ trait so those translation units build on Windows.
#ifdef _MSC_VER
// Vendored device headers (FlashInfer) use the GCC/Clang `ushort` alias.
typedef unsigned short ushort;
#if defined(__CUDACC__)
// The CUDA kernels in this project use the `__is_same` builtin provided by
// GCC/Clang, which nvcc's MSVC front end does not implement. Map it onto the
// libcu++ trait so those translation units build on Windows.
#include <cuda/std/type_traits>
#define __is_same(a, b) ::cuda::std::is_same<a, b>::value

// Host-side control words (host upload rings, disk MoE graphs, pageable
// sources) use GCC's __atomic builtins, which the MSVC host front end lacks.
// Map them onto std::atomic_ref with GCC's memory-order numbering. Every use
// is a host function over a plain non-volatile word that device stream memory
// operations also touch, so the reference stays lock-free.
#include <atomic>
#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5
#endif
inline std::memory_order __atomic_order_msvc(int order) {
    switch (order) {
        case 0: return std::memory_order_relaxed;
        case 1: return std::memory_order_consume;
        case 2: return std::memory_order_acquire;
        case 3: return std::memory_order_release;
        case 4: return std::memory_order_acq_rel;
        default: return std::memory_order_seq_cst;
    }
}
template <typename T>
inline T __atomic_load_n(const T *ptr, int order) {
    return std::atomic_ref<T>(const_cast<T &>(*ptr)).load(__atomic_order_msvc(order));
}
template <typename T>
inline void __atomic_store_n(T *ptr, T value, int order) {
    std::atomic_ref<T>(*ptr).store(value, __atomic_order_msvc(order));
}
#endif
#endif
