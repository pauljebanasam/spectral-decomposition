// stn/mask_estimator.hpp
// Swappable classifier strategy (IMaskEstimator) with the median-filter ratio
// estimator (Fitzgerald 2010; Driedger/Müller/Disch 2014 soft-mask form) as
// the first concrete implementation.
//
// STAGE C SEAMS (stubs at bottom of file):
//   - StructureTensorMaskEstimator  (Füg et al., ICASSP 2016): local
//     orientation/anisotropy on the log-spectrogram rescues vibrato/FM
//     partials that a pure horizontal median throws into the residual.
//   - PeakStabilityMaskEstimator    (US 10,430,154 B2): spectral-peak
//     tracking with frame-to-frame stability thresholding.
// Both only need to implement estimate() against the same SpectralContext.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace stn {

// Median of the first `count` values of v (scratch is modified).
// Odd counts give the exact sample median; even counts (which occur only when
// the vertical window shrinks at the spectrum edges) give the upper median.
inline double median_inplace(double* v, std::size_t count)
{
    const std::size_t mid = count / 2;
    std::nth_element(v, v + mid, v + count);
    return v[mid];
}

// Runtime (RT-safe) mask parameters.
struct MaskParams {
    double beta_tonal     = 2.0;  // separation factor, pass-1 mask (>= 1)
    double beta_transient = 2.0;  // separation factor, transient mask (>= 1)
    double softness_p     = 2.0;  // mask exponent p: 2 = Wiener; LARGER p -> harder/more
                                  // binary decisions; p -> 0 degenerates to 0.5 everywhere.
                                  // NOTE: the brief's table described this inverted
                                  // ("->0 = binary"); the maths is M = 1/(1 + (βv/h)^p),
                                  // which hardens as p grows. Range enforced [0.25, 8].
    double smoothing      = 0.0;  // per-bin one-pole temporal smoothing of mask values,
                                  // 0 = off. Frame-rate dependent (documented in README).
};

// Read-only view of the magnitude history handed to an estimator.
// mag_frames[0..num_frames-1] point at magnitude frames oldest -> newest
// (missing pre-roll frames point at a shared zero frame). target_index is
// the frame for which masks must be produced: centre of the window in
// symmetric mode, newest frame in causal mode. Estimators are agnostic to
// the lookahead policy — they just see a window and a target.
struct SpectralContext {
    const double* const* mag_frames = nullptr;
    std::size_t num_frames   = 0;
    std::size_t target_index = 0;
    std::size_t num_bins     = 0;
};

class IMaskEstimator {
public:
    virtual ~IMaskEstimator() = default;
    // Allocations happen here only; estimate() must be allocation-free.
    virtual void configure(std::size_t num_bins, std::size_t harm_len, std::size_t perc_len) = 0;
    virtual void reset() = 0;
    // Fill mask_tonal / mask_transient (num_bins each) with values in [0, 1].
    virtual void estimate(const SpectralContext& ctx, const MaskParams& p,
                          double* mask_tonal, double* mask_transient) = 0;
};

// ---------------------------------------------------------------------------
// Median-filter ratio estimator.
//   Ỹh[k] = median over time  (harm_len frames)  at bin k   — tonal-enhanced
//   Ỹp[k] = median over freq  (perc_len bins)    in target  — transient-enhanced
//   M_t [k] = Ỹh^p / (Ỹh^p + (β_t · Ỹp)^p + ε)
//   M_tr[k] = Ỹp^p / (Ỹp^p + (β_tr · Ỹh)^p + ε)
// For β >= 1, M_t + M_tr <= 1 per bin, so the implied residual mask is
// non-negative — but note null-sum does NOT depend on this: the noise stream
// is produced by time-domain subtraction (see stft_splitter.hpp).
//
// Complexity: O(bins · (harm_len + perc_len)) per hop via nth_element on
// preallocated scratch. OPTIMISATION HOOK: replace with a sliding-window
// histogram (magnitudes quantised to dB buckets) or two-heap median if the
// frame rate * filter length product ever matters; it does not at defaults.
// ---------------------------------------------------------------------------
class MedianMaskEstimator : public IMaskEstimator {
public:
    void configure(std::size_t num_bins, std::size_t harm_len, std::size_t perc_len) override
    {
        bins_ = num_bins;
        lh_   = harm_len;
        lp_   = perc_len;
        scratch_.assign(std::max(lh_, lp_), 0.0);
        htilde_.assign(bins_, 0.0);
        vtilde_.assign(bins_, 0.0);
        prev_t_.assign(bins_, 0.0);
        prev_tr_.assign(bins_, 0.0);
        first_frame_ = true;
    }

    void reset() override
    {
        std::fill(prev_t_.begin(), prev_t_.end(), 0.0);
        std::fill(prev_tr_.begin(), prev_tr_.end(), 0.0);
        first_frame_ = true;
    }

    void estimate(const SpectralContext& ctx, const MaskParams& prm,
                  double* mask_tonal, double* mask_transient) override
    {
        const std::size_t bins = ctx.num_bins;

        // Horizontal (time) median per bin over the full provided window.
        // Symmetric mode: window is centred on the target frame by construction.
        // Causal mode: the target IS the newest frame -> trailing median.
        for (std::size_t k = 0; k < bins; ++k) {
            for (std::size_t j = 0; j < ctx.num_frames; ++j)
                scratch_[j] = ctx.mag_frames[j][k];
            htilde_[k] = median_inplace(scratch_.data(), ctx.num_frames);
        }

        // Vertical (frequency) median within the target frame; window clamped
        // (shrunk) at the spectrum edges rather than padded.
        const double* target = ctx.mag_frames[ctx.target_index];
        const std::ptrdiff_t half = std::ptrdiff_t(lp_ - 1) / 2;
        for (std::ptrdiff_t k = 0; k < std::ptrdiff_t(bins); ++k) {
            const std::ptrdiff_t lo = std::max<std::ptrdiff_t>(0, k - half);
            const std::ptrdiff_t hi = std::min<std::ptrdiff_t>(std::ptrdiff_t(bins) - 1, k + half);
            const std::size_t count = std::size_t(hi - lo + 1);
            for (std::size_t j = 0; j < count; ++j)
                scratch_[j] = target[lo + std::ptrdiff_t(j)];
            vtilde_[std::size_t(k)] = median_inplace(scratch_.data(), count);
        }

        // Soft Wiener-style masks with separation factors.
        const double p      = std::clamp(prm.softness_p, 0.25, 8.0);
        const double bt     = std::max(1.0, prm.beta_tonal);
        const double btr    = std::max(1.0, prm.beta_transient);
        const double alpha  = std::clamp(prm.smoothing, 0.0, 0.999);
        constexpr double eps = 1e-30;
        const bool pow2fast = (p == 2.0);

        for (std::size_t k = 0; k < bins; ++k) {
            const double h = htilde_[k];
            const double v = vtilde_[k];
            double hp, vp, bvp, bhp;
            if (pow2fast) {
                hp  = h * h;            vp  = v * v;
                bvp = (bt * v) * (bt * v);
                bhp = (btr * h) * (btr * h);
            } else {
                hp  = std::pow(h, p);   vp  = std::pow(v, p);
                bvp = std::pow(bt * v, p);
                bhp = std::pow(btr * h, p);
            }
            double mt  = hp / (hp + bvp + eps);
            double mtr = vp / (vp + bhp + eps);
            if (!std::isfinite(mt))  mt  = 0.0;   // NaN/Inf guard (poisoned input)
            if (!std::isfinite(mtr)) mtr = 0.0;

            // Temporal mask smoothing (reduces musical noise / birdies,
            // cf. Eventide "Smoothing"). Cannot break null-sum: whatever the
            // masks do, the residual absorbs the complement via subtraction.
            if (alpha > 0.0 && !first_frame_) {
                mt  = alpha * prev_t_[k]  + (1.0 - alpha) * mt;
                mtr = alpha * prev_tr_[k] + (1.0 - alpha) * mtr;
            }
            prev_t_[k]  = mt;
            prev_tr_[k] = mtr;
            mask_tonal[k]     = mt;
            mask_transient[k] = mtr;
        }
        first_frame_ = false;
    }

private:
    std::size_t bins_ {0}, lh_ {0}, lp_ {0};
    std::vector<double> scratch_, htilde_, vtilde_, prev_t_, prev_tr_;
    bool first_frame_ {true};
};

// --------------------------- STAGE C STUBS --------------------------------
// class StructureTensorMaskEstimator : public IMaskEstimator {
//   // Füg et al. 2016: smooth log-magnitude S with a small Gaussian; form
//   // T = G_σ * [Sx²  SxSy; SxSy  Sy²] from finite-difference gradients;
//   // eigen-decompose per bin -> orientation angle α and anisotropy C;
//   // map (α, C) to tonal/transient membership with angle thresholds that
//   // account for the STFT's time/frequency aspect ratio (hop vs bin width).
// };
//
// class PeakStabilityMaskEstimator : public IMaskEstimator {
//   // US 10,430,154 B2 flavour: per-frame peak picking; track peak
//   // frequency/amplitude across frames; peaks stable for >= S frames are
//   // tonal; energy in short-lived broadband clusters is transient.
// };
// ---------------------------------------------------------------------------

} // namespace stn
