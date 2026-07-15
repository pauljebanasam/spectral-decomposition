// stn/util.hpp
// Small shared utilities for the STN core. No Max SDK dependencies anywhere in core/.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
  #include <xmmintrin.h>
  #define STN_HAVE_SSE 1
#endif

namespace stn {

inline bool is_pow2(std::size_t v) { return v != 0 && (v & (v - 1)) == 0; }

// Round v to the nearest power of two within [lo, hi] (lo and hi must be powers of two).
inline std::size_t clamp_pow2(std::size_t v, std::size_t lo, std::size_t hi)
{
    if (v <= lo) return lo;
    if (v >= hi) return hi;
    std::size_t below = lo;
    while ((below << 1) <= v) below <<= 1;
    const std::size_t above = below << 1;
    return (v - below < above - v) ? below : (above <= hi ? above : below);
}

inline int force_odd(int v) { return (v % 2 == 0) ? v + 1 : v; }

// RAII flush-to-zero / denormals-are-zero for the audio thread.
// x86: MXCSR FTZ+DAZ. arm64: FPCR.FZ. Elsewhere: no-op (the core also uses
// epsilon guards in every feedback path, so this is belt-and-braces).
class ScopedNoDenormals {
public:
    ScopedNoDenormals()
    {
#if defined(STN_HAVE_SSE)
        saved_ = _mm_getcsr();
        _mm_setcsr(saved_ | 0x8040u);          // FTZ (bit 15) | DAZ (bit 6)
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(saved_));
        std::uint64_t v = saved_ | (1ull << 24);   // FZ
        __asm__ __volatile__("msr fpcr, %0" ::"r"(v));
#endif
    }
    ~ScopedNoDenormals()
    {
#if defined(STN_HAVE_SSE)
        _mm_setcsr(saved_);
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        __asm__ __volatile__("msr fpcr, %0" ::"r"(saved_));
#endif
    }
private:
#if defined(STN_HAVE_SSE)
    unsigned int saved_ {0};
#elif defined(__aarch64__)
    std::uint64_t saved_ {0};
#endif
};

inline double db_to_lin(double db) { return std::pow(10.0, db / 20.0); }

} // namespace stn
