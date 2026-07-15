// stn/engine.hpp
// StnEngine — SDK-agnostic top level of the STN core (Stage A).
//
// Owns: one StftSplitter (single resolution), the input delay line, the
// complementary noise subtraction, and the output stage (per-stream gain,
// solo, delay-matched bypass, pre-gain sum tap).
//
// NULL-SUM GUARANTEE: noise[n] = x[n−D] − tonal[n] − transient[n] in the
// time domain, so tonal + transient + noise ≡ delay(x, D) up to one float
// rounding of the additions — independent of mask behaviour, smoothing,
// or classifier choice. The sum outlet recomputes tonal+transient+noise
// (rather than short-circuiting to the delayed input) so the null test
// measures the real signal path.
//
// STAGE B SEAM: add `StftSplitter pass2_` (small N, its own estimator),
// feed it r1 = delayed − tonal per sample, take its "transient" output,
// and derive noise = delay(r1, D2) − transient. latency() becomes
// pass1.latency() + pass2.latency(); the extra delay line on r1 replaces
// part of the current input delay. Nothing else changes.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "mask_estimator.hpp"
#include "stft_splitter.hpp"
#include "util.hpp"

namespace stn {

struct EngineConfig : SplitterConfig {
    // Stage B adds: std::size_t fft_size_transient; (pass-2 window)
};

enum class Solo : int { none = 0, tonal = 1, transient = 2, noise = 3 };

class StnEngine {
public:
    // Not RT-safe (allocates). sample_rate drives smoothing coefficients only.
    void configure(const EngineConfig& cfg, double sample_rate)
    {
        cfg_ = cfg;
        sr_  = (sample_rate > 0.0) ? sample_rate : 48000.0;
        splitter_.configure(cfg, &median_est_);
        // STAGE C SEAM: select estimator by `classifier` here once the
        // structure-tensor / peak-stability implementations exist.

        delay_.assign(splitter_.latency(), 0.0);
        delay_w_ = 0;

        smooth_coef_ = std::exp(-1.0 / (0.010 * sr_));   // ~10 ms parameter ramps
        g_t_ = g_tr_ = g_n_ = 1.0;
        m_t_ = m_tr_ = m_n_ = 1.0;
        byp_ = 0.0;
        reset();
    }

    void reset()
    {
        splitter_.reset();
        std::fill(delay_.begin(), delay_.end(), 0.0);
        delay_w_ = 0;
    }

    std::size_t latency() const { return splitter_.latency(); }
    double cola_deviation() const { return splitter_.cola_deviation(); }

    // ---- RT-safe parameter setters (called per block from the glue) ----
    void set_mask_params(const MaskParams& p) { splitter_.set_params(p); }
    void set_identity(bool b)                 { splitter_.set_identity(b); }
    void set_gains_db(double t, double tr, double n)
    {
        tgt_g_t_  = (t  <= -69.5) ? 0.0 : db_to_lin(std::min(t,  12.0));
        tgt_g_tr_ = (tr <= -69.5) ? 0.0 : db_to_lin(std::min(tr, 12.0));
        tgt_g_n_  = (n  <= -69.5) ? 0.0 : db_to_lin(std::min(n,  12.0));
    }
    void set_solo(Solo s)    { solo_ = s; }
    void set_bypass(bool b)  { tgt_byp_ = b ? 1.0 : 0.0; }

    // Allocation-free; any block size n (Max vector size need not equal hop).
    void process(const double* in,
                 double* out_tonal, double* out_transient, double* out_noise,
                 double* out_sum, std::size_t n)
    {
        const double c = smooth_coef_;
        const double st  = (solo_ == Solo::none || solo_ == Solo::tonal)     ? 1.0 : 0.0;
        const double str = (solo_ == Solo::none || solo_ == Solo::transient) ? 1.0 : 0.0;
        const double sn  = (solo_ == Solo::none || solo_ == Solo::noise)     ? 1.0 : 0.0;
        const std::size_t D = delay_.size();

        for (std::size_t i = 0; i < n; ++i) {
            const double x = in[i];

            // Matched input delay (read-then-write ring => exactly D samples).
            const double delayed = delay_[delay_w_];
            delay_[delay_w_] = x;
            if (++delay_w_ == D) delay_w_ = 0;

            double t, tr;
            splitter_.tick(x, t, tr);

            // Complementary reconstruction: the residual is whatever the two
            // masked streams did not claim. This is the null-sum mechanism.
            const double noise = delayed - t - tr;
            const double sum   = t + tr + noise;   // honest recomputation for the null test

            // Output stage (post-split; never part of the null-sum math).
            g_t_  = tgt_g_t_  * st  + (g_t_  - tgt_g_t_  * st ) * c;
            g_tr_ = tgt_g_tr_ * str + (g_tr_ - tgt_g_tr_ * str) * c;
            g_n_  = tgt_g_n_  * sn  + (g_n_  - tgt_g_n_  * sn ) * c;
            byp_  = tgt_byp_ + (byp_ - tgt_byp_) * c;

            // Delay-matched bypass: crossfade tonal outlet to the dry (delayed)
            // input, other streams to silence; sum stays == delayed throughout.
            const double wet = 1.0 - byp_;
            out_tonal[i]     = t     * g_t_  * wet + delayed * byp_;
            out_transient[i] = tr    * g_tr_ * wet;
            out_noise[i]     = noise * g_n_  * wet;
            if (out_sum) out_sum[i] = sum;
        }
    }

private:
    EngineConfig cfg_;
    double sr_ {48000.0};

    MedianMaskEstimator median_est_;
    StftSplitter splitter_;

    std::vector<double> delay_;
    std::size_t delay_w_ {0};

    double smooth_coef_ {0.0};
    double g_t_ {1.0}, g_tr_ {1.0}, g_n_ {1.0};
    double tgt_g_t_ {1.0}, tgt_g_tr_ {1.0}, tgt_g_n_ {1.0};
    double m_t_ {1.0}, m_tr_ {1.0}, m_n_ {1.0};
    double byp_ {0.0}, tgt_byp_ {0.0};
    Solo solo_ {Solo::none};
};

} // namespace stn
