// stn/fft.hpp
// Self-contained real FFT for the STN core.
//
// Design: RealFft is the *interface the rest of the core codes against*
// (N real samples <-> N/2+1 complex bins, inverse(forward(x)) == x).
// The reference backend below is a plain iterative radix-2 complex FFT with a
// half-size real pack/unpack — portable, dependency-free, and fast enough for
// frame-rate STFT work at N <= 8192.
//
// SWAP HOOK: to substitute pffft (BSD) or vDSP/Accelerate, reimplement only
// RealFft::init/forward/inverse keeping the same scaling convention:
//   forward : unscaled DFT, bins k = 0..N/2 (X[0], X[N/2] purely real)
//   inverse : includes the 1/N normalisation (round-trip identity)
#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "util.hpp"

namespace stn {

// ---------------------------------------------------------------------------
// Iterative radix-2 complex FFT (decimation-in-time), precomputed twiddles
// and bit-reversal table. In-place on interleaved std::complex<double>.
// ---------------------------------------------------------------------------
class ComplexFft {
public:
    void init(std::size_t n)
    {
        assert(is_pow2(n));
        n_ = n;
        std::size_t bits = 0;
        while ((std::size_t(1) << bits) < n) ++bits;

        rev_.assign(n, 0);
        for (std::size_t i = 1; i < n; ++i)
            rev_[i] = (rev_[i >> 1] >> 1) | ((i & 1u) << (bits - 1));

        tw_.resize(n / 2);
        for (std::size_t k = 0; k < n / 2; ++k)
            tw_[k] = std::polar(1.0, -2.0 * M_PI * double(k) / double(n));
    }

    void forward(std::complex<double>* a) const { transform(a, false); }

    void inverse(std::complex<double>* a) const
    {
        transform(a, true);
        const double s = 1.0 / double(n_);
        for (std::size_t i = 0; i < n_; ++i) a[i] *= s;
    }

private:
    void transform(std::complex<double>* a, bool inv) const
    {
        for (std::size_t i = 0; i < n_; ++i)
            if (i < rev_[i]) std::swap(a[i], a[rev_[i]]);

        for (std::size_t len = 2; len <= n_; len <<= 1) {
            const std::size_t half = len >> 1;
            const std::size_t step = n_ / len;
            for (std::size_t i = 0; i < n_; i += len) {
                for (std::size_t j = 0; j < half; ++j) {
                    std::complex<double> w = tw_[j * step];
                    if (inv) w = std::conj(w);
                    const std::complex<double> u = a[i + j];
                    const std::complex<double> v = a[i + j + half] * w;
                    a[i + j]        = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }
    }

    std::size_t n_ {0};
    std::vector<std::size_t> rev_;
    std::vector<std::complex<double>> tw_;
};

// ---------------------------------------------------------------------------
// Real FFT via half-size complex FFT.
//   forward: pack z[m] = x[2m] + j x[2m+1], FFT_{N/2}, untangle:
//            Ze[k] = (Z[k] + conj(Z[M-k]))/2,  Zo[k] = -j(Z[k] - conj(Z[M-k]))/2
//            X[k]  = Ze[k] + e^{-j2πk/N} Zo[k],  k = 0..M,  Z[M] := Z[0]
//   inverse: exact algebraic inversion of the above, then IFFT_{N/2}.
// X[0] and X[N/2] are computed with real arithmetic so their imaginary parts
// are exactly zero (keeps Hermitian synthesis bit-clean).
// ---------------------------------------------------------------------------
class RealFft {
public:
    void init(std::size_t n)
    {
        assert(is_pow2(n) && n >= 4);
        n_ = n;
        m_ = n / 2;
        cf_.init(m_);
        buf_.assign(m_, {0.0, 0.0});
        ut_.resize(m_ + 1);
        for (std::size_t k = 0; k <= m_; ++k)
            ut_[k] = std::polar(1.0, -2.0 * M_PI * double(k) / double(n));
    }

    std::size_t size() const { return n_; }
    std::size_t bins() const { return m_ + 1; }   // N/2 + 1

    // x: n_ reals in, X: bins() complex out. Unscaled DFT.
    void forward(const double* x, std::complex<double>* X)
    {
        for (std::size_t m = 0; m < m_; ++m)
            buf_[m] = {x[2 * m], x[2 * m + 1]};
        cf_.forward(buf_.data());

        X[0]  = {buf_[0].real() + buf_[0].imag(), 0.0};
        X[m_] = {buf_[0].real() - buf_[0].imag(), 0.0};

        for (std::size_t k = 1; k < m_; ++k) {
            const std::complex<double> Zk  = buf_[k];
            const std::complex<double> Zmk = std::conj(buf_[m_ - k]);
            const std::complex<double> Ze  = 0.5 * (Zk + Zmk);
            const std::complex<double> Zo  = std::complex<double>(0.0, -0.5) * (Zk - Zmk);
            X[k] = Ze + ut_[k] * Zo;
        }
    }

    // X: bins() complex in (Hermitian implied), x: n_ reals out.
    // Includes 1/N scaling: inverse(forward(x)) == x.
    void inverse(const std::complex<double>* X, double* x)
    {
        buf_[0] = {0.5 * (X[0].real() + X[m_].real()),
                   0.5 * (X[0].real() - X[m_].real())};

        for (std::size_t k = 1; k < m_; ++k) {
            const std::complex<double> Xk  = X[k];
            const std::complex<double> Xmk = std::conj(X[m_ - k]);
            const std::complex<double> Ze  = 0.5 * (Xk + Xmk);
            const std::complex<double> Zo  = 0.5 * std::conj(ut_[k]) * (Xk - Xmk);
            buf_[k] = Ze + std::complex<double>(0.0, 1.0) * Zo;
        }

        cf_.inverse(buf_.data());   // includes 1/M
        for (std::size_t m = 0; m < m_; ++m) {
            x[2 * m]     = buf_[m].real();
            x[2 * m + 1] = buf_[m].imag();
        }
    }

private:
    std::size_t n_ {0}, m_ {0};
    ComplexFft cf_;
    std::vector<std::complex<double>> buf_;
    std::vector<std::complex<double>> ut_;   // e^{-j2πk/N}, k = 0..M
};

} // namespace stn
