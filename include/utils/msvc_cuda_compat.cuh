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
#endif
#endif
