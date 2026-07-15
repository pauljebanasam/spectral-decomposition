// stn/stft_splitter.hpp
// Stage A single-resolution STFT split pipeline.
//
// One analysis (√Hann, WOLA) drives TWO masked syntheses (tonal, transient)
// from the same complex frames. The caller (StnEngine) derives the noise
// stream by time-domain subtraction against a matched delay of the input —
// complementary reconstruction, so null-sum holds by construction and does
// not depend on masks summing to unity.
//
// STAGE B SEAM: in the multiresolution build this class is instantiated
// twice — pass 1 (large N, tonal mask only) feeding a time-domain remainder
// r1 = delay(x) - tonal into pass 2 (small N, transient mask only). The
// tick() API (one sample in, masked streams out, fixed latency) was shaped
// so pass 2 can consume pass 1's remainder sample-by-sample with no extra
// buffering layer. Total latency then composes as D = D1 + D2.
//
// LATENCY (exact, derived from this buffering — not estimated):
//   Frame f is analysed when input sample count reaches (f+1)·H; it covers
//   input samples [(f+1)H − N, (f+1)H − 1] and its OLA contribution starts
//   at output position (f+1)H − N. With K frames of median lookahead
//   (K = (L_h−1)/2 symmetric, 0 causal), frame g is synthesised once frame
//   g+K is analysed. Output position q is complete once its last covering
//   frame, g_max = floor((q+N)/H) − 1, is synthesised — which is guaranteed
//   for q = t − (N + K·H) after processing input sample t. Hence exactly
//       latency D = N + K·H  samples.
#pragma once

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fft.hpp"
#include "mask_estimator.hpp"
#include "util.hpp"

namespace stn {

struct SplitterConfig {
    std::size_t fft_size  = 4096;   // pow2, 128..8192
    std::size_t overlap   = 4;      // 2 / 4 / 8 (hop = fft_size / overlap)
    std::size_t harm_len  = 17;     // horizontal (time) median length, odd
    std::size_t perc_len  = 31;     // vertical (frequency) median length, odd
    bool symmetric_median = true;   // true: centred median (+K·hop latency, better
                                    // tonal/transient discrimination);
                                    // false: trailing median (min latency, tonal
                                    // onsets classified slightly "late")
};

class StftSplitter {
public:
    // Not RT-safe (allocates). Call from a non-audio thread / dspsetup.
    void configure(const SplitterConfig& cfg, IMaskEstimator* estimator)
    {
        assert(is_pow2(cfg.fft_size));
        assert(cfg.overlap == 2 || cfg.overlap == 4 || cfg.overlap == 8);
        assert(cfg.harm_len % 2 == 1 && cfg.perc_len % 2 == 1);

        cfg_  = cfg;
        est_  = estimator;
        n_    = cfg.fft_size;
        hop_  = n_ / cfg.overlap;
        bins_ = n_ / 2 + 1;
        k_    = cfg.symmetric_median ? (cfg.harm_len - 1) / 2 : 0;

        fft_.init(n_);

        // Periodic √Hann on analysis and synthesis => product is periodic
        // Hann, which is exactly COLA for hop = N/overlap, overlap in {2,4,8}.
        window_.resize(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            const double hann = 0.5 * (1.0 - std::cos(2.0 * M_PI * double(i) / double(n_)));
            window_[i] = std::sqrt(std::max(0.0, hann));
        }

        // Verify COLA numerically and derive the synthesis normalisation.
        double cmin = std::numeric_limits<double>::max(), cmax = 0.0, csum = 0.0;
        for (std::size_t o = 0; o < hop_; ++o) {
            double s = 0.0;
            for (std::size_t i = o; i < n_; i += hop_) s += window_[i] * window_[i];
            cmin = std::min(cmin, s); cmax = std::max(cmax, s); csum += s;
        }
        cola_dev_  = cmax - cmin;
        cola_gain_ = 1.0 / (csum / double(hop_));

        in_ring_.assign(n_, 0.0);
        frame_.assign(n_, 0.0);
        time_t_.assign(n_, 0.0);
        time_tr_.assign(n_, 0.0);
        spec_.assign(bins_, {0.0, 0.0});
        spec_t_.assign(bins_, {0.0, 0.0});
        spec_tr_.assign(bins_, {0.0, 0.0});
        mask_t_.assign(bins_, 0.0);
        mask_tr_.assign(bins_, 0.0);

        mag_ring_.assign(cfg.harm_len * bins_, 0.0);
        zero_frame_.assign(bins_, 0.0);
        ctx_ptrs_.assign(cfg.harm_len, zero_frame_.data());
        spec_ring_.assign((k_ + 1) * bins_, {0.0, 0.0});

        ola_size_ = 2 * n_;
        ola_t_.assign(ola_size_, 0.0);
        ola_tr_.assign(ola_size_, 0.0);

        if (est_) est_->configure(bins_, cfg.harm_len, cfg.perc_len);
        reset();
    }

    // Clears streaming state (memset-scale work; call outside audio when possible).
    void reset()
    {
        std::fill(in_ring_.begin(), in_ring_.end(), 0.0);
        std::fill(mag_ring_.begin(), mag_ring_.end(), 0.0);
        std::fill(spec_ring_.begin(), spec_ring_.end(), std::complex<double>(0.0, 0.0));
        std::fill(ola_t_.begin(), ola_t_.end(), 0.0);
        std::fill(ola_tr_.begin(), ola_tr_.end(), 0.0);
        in_write_ = 0;
        hop_count_ = 0;
        samples_in_ = 0;
        if (est_) est_->reset();
    }

    std::size_t latency()        const { return n_ + k_ * hop_; }   // exact; see header
    std::size_t hop()            const { return hop_; }
    std::size_t bins()           const { return bins_; }
    double      cola_deviation() const { return cola_dev_; }

    void set_params(const MaskParams& p) { params_ = p; }
    // Debug/verification: masks forced to M_t = 1, M_tr = 0 -> the tonal
    // stream must reproduce delay(x, latency()) to float precision. This is
    // the test that pins down windowing, COLA gain, FFT round-trip and the
    // latency formula simultaneously.
    void set_identity(bool b) { identity_ = b; }

    // Push one input sample; receive the two masked streams (each delayed by
    // exactly latency() samples). Allocation-free.
    inline void tick(double in, double& tonal, double& transient)
    {
        in_ring_[in_write_] = in;
        in_write_ = (in_write_ + 1) % n_;
        ++samples_in_;
        if (++hop_count_ == hop_) {
            hop_count_ = 0;
            process_frame();
        }
        const std::int64_t q = std::int64_t(samples_in_) - 1 - std::int64_t(latency());
        if (q < 0) {
            tonal = 0.0;
            transient = 0.0;
        } else {
            const std::size_t slot = std::size_t(q) % ola_size_;
            tonal     = ola_t_[slot];
            transient = ola_tr_[slot];
            ola_t_[slot]  = 0.0;   // slot consumed; ready for reuse
            ola_tr_[slot] = 0.0;
        }
    }

private:
    void process_frame()
    {
        // Assemble the most recent N samples (oldest first) and window.
        for (std::size_t i = 0; i < n_; ++i)
            frame_[i] = in_ring_[(in_write_ + i) % n_] * window_[i];

        fft_.forward(frame_.data(), spec_.data());

        const std::int64_t f = std::int64_t(samples_in_ / hop_) - 1;

        // Store magnitude + complex frames in their rings.
        double* mag_slot = mag_frame(std::size_t(f % std::int64_t(cfg_.harm_len)));
        for (std::size_t kk = 0; kk < bins_; ++kk)
            mag_slot[kk] = std::abs(spec_[kk]);
        std::complex<double>* spec_slot = spec_frame(std::size_t(f % std::int64_t(k_ + 1)));
        std::copy(spec_.begin(), spec_.end(), spec_slot);

        const std::int64_t g = f - std::int64_t(k_);   // frame to synthesise
        if (g < 0) return;

        if (identity_) {
            for (std::size_t kk = 0; kk < bins_; ++kk) { mask_t_[kk] = 1.0; mask_tr_[kk] = 0.0; }
        } else {
            // Median window = the last harm_len analysed frames: in symmetric
            // mode this centres on g (since K = (L_h−1)/2); in causal mode the
            // target g == f is the newest frame -> trailing median. Pre-roll
            // frames (< 0) read as the shared zero frame; masks therefore ramp
            // in from "everything is noise" during the first L_h hops.
            for (std::size_t j = 0; j < cfg_.harm_len; ++j) {
                const std::int64_t idx = f - std::int64_t(cfg_.harm_len - 1) + std::int64_t(j);
                ctx_ptrs_[j] = (idx >= 0)
                             ? mag_frame(std::size_t(idx % std::int64_t(cfg_.harm_len)))
                             : zero_frame_.data();
            }
            SpectralContext ctx;
            ctx.mag_frames   = ctx_ptrs_.data();
            ctx.num_frames   = cfg_.harm_len;
            ctx.target_index = cfg_.harm_len - 1 - k_;
            ctx.num_bins     = bins_;
            est_->estimate(ctx, params_, mask_t_.data(), mask_tr_.data());
        }

        // Real-valued masks on the complex frame: original phase preserved,
        // Hermitian symmetry preserved (we hold only bins 0..N/2 and the
        // inverse reconstructs the negative frequencies by conjugation).
        // PHASE HOOK (Stage C): per-stream complex modification goes here —
        // e.g. noise-stream phase decorrelation, transient phase-locking, or
        // RTPGHI (Průša et al. 2017) re-synthesis of a transformed magnitude.
        const std::complex<double>* src = spec_frame(std::size_t(g % std::int64_t(k_ + 1)));
        for (std::size_t kk = 0; kk < bins_; ++kk) {
            spec_t_[kk]  = src[kk] * mask_t_[kk];
            spec_tr_[kk] = src[kk] * mask_tr_[kk];
        }

        fft_.inverse(spec_t_.data(),  time_t_.data());
        fft_.inverse(spec_tr_.data(), time_tr_.data());

        // Weighted overlap-add at the frame's true output position.
        // Positions < 0 belong to the zero-padded pre-roll and are skipped
        // (they are never emitted; wrapping them would corrupt the ring).
        const std::int64_t s0 = (g + 1) * std::int64_t(hop_) - std::int64_t(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            const std::int64_t pos = s0 + std::int64_t(i);
            if (pos < 0) continue;
            const std::size_t slot = std::size_t(pos) % ola_size_;
            const double w = window_[i] * cola_gain_;
            ola_t_[slot]  += time_t_[i]  * w;
            ola_tr_[slot] += time_tr_[i] * w;
        }
    }

    double* mag_frame(std::size_t slot) { return mag_ring_.data() + slot * bins_; }
    std::complex<double>* spec_frame(std::size_t slot) { return spec_ring_.data() + slot * bins_; }

    SplitterConfig cfg_;
    IMaskEstimator* est_ {nullptr};
    MaskParams params_;
    bool identity_ {false};

    std::size_t n_ {0}, hop_ {0}, bins_ {0}, k_ {0};
    RealFft fft_;
    std::vector<double> window_;
    double cola_gain_ {1.0}, cola_dev_ {0.0};

    std::vector<double> in_ring_, frame_, time_t_, time_tr_;
    std::vector<std::complex<double>> spec_, spec_t_, spec_tr_;
    std::vector<double> mask_t_, mask_tr_;

    std::vector<double> mag_ring_, zero_frame_;
    std::vector<const double*> ctx_ptrs_;
    std::vector<std::complex<double>> spec_ring_;

    std::vector<double> ola_t_, ola_tr_;
    std::size_t ola_size_ {0};

    std::size_t in_write_ {0}, hop_count_ {0};
    std::uint64_t samples_in_ {0};
};

} // namespace stn
