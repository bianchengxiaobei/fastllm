#pragma once

// MSVC has no 128-bit integer type.  The byte and page budgets in basellm.cpp
// use it only to keep 64-bit multiplications from overflowing, and the operands
// there stay far below 2^63, so a 64-bit type is enough on Windows.
#ifdef _MSC_VER
#define FASTLLM_INT128 long long
#else
#define FASTLLM_INT128 __int128
#endif
