/// @file  stn.split_tilde.cpp
/// @brief stn.split~ — real-time tonal / transient / noise splitter.
///
/// SDK CHOICE: Min-DevKit (min-api, modern C++). The DSP core under core/stn/
/// is pure C++ with zero Max dependencies (see engine.hpp) so it can later be
/// wrapped in JUCE for VST/AU unchanged. This file is glue only: attribute
/// plumbing, thread-safe engine rebuilds, and the perform loop.
///
/// PROJECT STRUCTURE (drop this whole folder into min-devkit/source/projects/):
///   stn.split_tilde/
///     stn.split_tilde.cpp   <- this file (Min glue)
///     CMakeLists.txt        <- standard min-devkit project script + core include
///     core/stn/*.hpp        <- SDK-agnostic DSP core (StnEngine et al.)
///     test/                 <- headless acceptance tests (no Max required)
///     README.md             <- build steps, parameter reference, null-test patch
///
/// STAGES: A implemented (single-resolution three-way, complementary
/// reconstruction, null-sum verified at ~ -300 dB). B/C seams are marked in
/// core/stn/engine.hpp, stft_splitter.hpp and mask_estimator.hpp.
///
/// INLETS / OUTLETS
///   inlet  1: (signal) input
///   outlet 1: (signal) tonal      — sustained partials
///   outlet 2: (signal) transient  — onsets / attacks
///   outlet 3: (signal) noise      — residual (input - tonal - transient)
///   outlet 4: (signal) sum        — tonal+transient+noise for null-testing
///                                    (silent unless @sumout 1)
///   outlet 5: (message) "latency <samples>" on dsp start / reconfigure,
///             and in response to the 'latency' message
///
/// THREADING MODEL: heavy attributes (fftsize, overlap, filter lengths,
/// latencymode) rebuild a fresh StnEngine off the audio thread and swap it
/// under a mutex; the perform routine uses try_lock and outputs silence for
/// the (rare) block during a swap. Light parameters (betas, softness,
/// smoothing, gains, solo, bypass) are plain attribute reads pushed into the
/// engine once per vector — engine-side they are simple scalar stores.

#include "c74_min.h"
#include "stn/engine.hpp"

#include <memory>
#include <mutex>

using namespace c74::min;

class stn_split : public object<stn_split>, public vector_operator<> {
private:
    // --- state that attribute setters touch: MUST be declared before the
    // --- attributes themselves (member initialisation order).
    stn::EngineConfig               m_cfg;        // pending/current heavy config
    double                          m_sr {0.0};   // 0 until dspsetup
    std::mutex                      m_engine_mutex;
    std::unique_ptr<stn::StnEngine> m_engine;

public:
    MIN_DESCRIPTION { "Split a signal into tonal, transient and noise streams. "
                      "Median-filtering HPR decomposition with complementary "
                      "(subtractive) reconstruction: the three outlets sum "
                      "exactly to the latency-delayed input. Stage A: single "
                      "resolution." };
    MIN_TAGS        { "audio, spectral, analysis" };
    MIN_AUTHOR      { "STN project" };
    MIN_RELATED     { "pfft~, fluid.hpss~, fluid.transients~" };

    inlet<>  m_in    { this, "(signal) input" };
    outlet<> m_out_t { this, "(signal) tonal", "signal" };
    outlet<> m_out_r { this, "(signal) transient", "signal" };
    outlet<> m_out_n { this, "(signal) noise", "signal" };
    outlet<> m_out_s { this, "(signal) sum for null-testing (enable with @sumout 1)", "signal" };
    outlet<> m_info  { this, "(message) latency <samples>" };

    // ------------------------------------------------------------ heavy attrs
    attribute<int> fftsize { this, "fftsize", 4096,
        description { "FFT/window size in samples (power of two, 256-8192). "
                      "Stage A single resolution; Stage B splits this into "
                      "fftsize_tonal / fftsize_transient." },
        setter { MIN_FUNCTION {
            const int v = int(stn::clamp_pow2(std::size_t(std::max(1, int(args[0]))), 256, 8192));
            if (std::size_t(v) != m_cfg.fft_size) { m_cfg.fft_size = std::size_t(v); rebuild(); }
            return { v };
        }}};

    attribute<int> overlap { this, "overlap", 4,
        description { "Overlap factor 2/4/8 (hop = fftsize / overlap)." },
        setter { MIN_FUNCTION {
            const int in = int(args[0]);
            const int v  = (in <= 2) ? 2 : (in <= 5 ? 4 : 8);
            if (std::size_t(v) != m_cfg.overlap) { m_cfg.overlap = std::size_t(v); rebuild(); }
            return { v };
        }}};

    attribute<int> harmfiltersize { this, "harmfiltersize", 17,
        description { "Horizontal (time) median length in frames, odd, 3-101. "
                      "Longer = stricter tonal stability; in symmetric mode "
                      "adds (len-1)/2 hops of latency." },
        setter { MIN_FUNCTION {
            const int v = stn::force_odd(std::clamp(int(args[0]), 3, 101));
            if (std::size_t(v) != m_cfg.harm_len) { m_cfg.harm_len = std::size_t(v); rebuild(); }
            return { v };
        }}};

    attribute<int> percfiltersize { this, "percfiltersize", 31,
        description { "Vertical (frequency) median length in bins, odd, 3-101. "
                      "Longer = broader events required to count as transient." },
        setter { MIN_FUNCTION {
            const int v = stn::force_odd(std::clamp(int(args[0]), 3, 101));
            if (std::size_t(v) != m_cfg.perc_len) { m_cfg.perc_len = std::size_t(v); rebuild(); }
            return { v };
        }}};

    attribute<symbol> latencymode { this, "latencymode", "symmetric",
        description { "'symmetric': centred time-median, best quality, adds "
                      "(harmfiltersize-1)/2 hops of latency. 'causal': trailing "
                      "median, minimum latency (= fftsize), slightly late tonal "
                      "onset classification." },
        setter { MIN_FUNCTION {
            const symbol s   = args[0];
            const bool   sym = !(s == symbol("causal"));
            if (sym != m_cfg.symmetric_median) { m_cfg.symmetric_median = sym; rebuild(); }
            return { sym ? symbol("symmetric") : symbol("causal") };
        }}};

    // ------------------------------------------------------------ light attrs
    // Read once per signal vector in operator(); engine-side these are plain
    // scalar stores, safe against the audio thread.
    attribute<number> betatonal { this, "betatonal", 2.0,
        description { "Separation factor for the tonal mask, 1-5. Larger pushes "
                      "more energy toward the noise stream." },
        setter { MIN_FUNCTION { return { std::clamp(double(args[0]), 1.0, 5.0) }; }}};

    attribute<number> betatransient { this, "betatransient", 2.0,
        description { "Separation factor for the transient mask, 1-5." },
        setter { MIN_FUNCTION { return { std::clamp(double(args[0]), 1.0, 5.0) }; }}};

    attribute<number> masksoftness { this, "masksoftness", 2.0,
        description { "Mask exponent p, 0.25-8. p=2 is Wiener-like (smoothest); "
                      "LARGER p hardens toward binary masks. (Note: direction "
                      "corrected relative to the design brief.)" },
        setter { MIN_FUNCTION { return { std::clamp(double(args[0]), 0.25, 8.0) }; }}};

    attribute<number> masksmoothing { this, "masksmoothing", 0.0,
        description { "Temporal one-pole smoothing of mask values, 0-1. Reduces "
                      "musical noise / birdies at the cost of transient snap. "
                      "Frame-rate dependent. Cannot break the null-sum." },
        setter { MIN_FUNCTION { return { std::clamp(double(args[0]), 0.0, 0.999) }; }}};

    attribute<symbol> classifier { this, "classifier", "median",
        description { "Mask estimator. Stage A implements 'median' only; "
                      "'structuretensor' and 'peakstability' are Stage C seams." },
        setter { MIN_FUNCTION {
            const symbol s = args[0];
            if (!(s == symbol("median")))
                cerr << "classifier not built yet (Stage C); using 'median'" << endl;
            return { symbol("median") };
        }}};

    attribute<number> gaintonal     { this, "gaintonal", 0.0,
        description { "Tonal stream output gain, dB (-70 = -inf, up to +12). Post-split; never affects the sum outlet." } };
    attribute<number> gaintransient { this, "gaintransient", 0.0,
        description { "Transient stream output gain, dB." } };
    attribute<number> gainnoise     { this, "gainnoise", 0.0,
        description { "Noise stream output gain, dB." } };

    attribute<symbol> solo { this, "solo", "none",
        description { "Audition one stream: none / tonal / transient / noise. "
                      "Others fade to silence (~10 ms ramp)." } };

    attribute<bool> sumout { this, "sumout", false,
        description { "Enable the 4th signal outlet: tonal+transient+noise, "
                      "delay-aligned. Null it against a delay~ of the input set "
                      "to the reported latency to verify reconstruction." } };

    attribute<bool> bypass { this, "bypass", false,
        description { "Delay-matched bypass: tonal outlet crossfades to the "
                      "delayed dry input, other streams to silence." } };

    attribute<bool> identitymode { this, "identitymode", false,
        description { "Debug: force masks to (1,0). Tonal outlet must then null "
                      "exactly against the delayed input — verifies the STFT "
                      "chain and latency report from inside Max." } };

    // ------------------------------------------------------------ messages
    message<> latency { this, "latency", "Report total latency in samples out the rightmost outlet.",
        MIN_FUNCTION {
            report_latency();
            return {};
        }};

    message<> reset { this, "reset", "Clear all internal state (rings, medians, delay).",
        MIN_FUNCTION {
            std::lock_guard<std::mutex> lock(m_engine_mutex);
            if (m_engine) m_engine->reset();   // audio outputs silence for this block (try_lock)
            return {};
        }};

    message<> dspsetup { this, "dspsetup",
        MIN_FUNCTION {
            m_sr = double(args[0]);
            rebuild_now();   // allocation is safe here: DSP chain is being compiled
            return {};
        }};

    // ------------------------------------------------------------ perform
    void operator()(audio_bundle input, audio_bundle output)
    {
        std::unique_lock<std::mutex> lock(m_engine_mutex, std::try_to_lock);
        if (!lock.owns_lock() || !m_engine) {           // engine being swapped/reset
            for (auto ch = 0; ch < output.channel_count(); ++ch) {
                double* o = output.samples(ch);
                for (auto i = 0; i < output.frame_count(); ++i) o[i] = 0.0;
            }
            return;
        }

        stn::ScopedNoDenormals no_denormals;

        stn::MaskParams mp;
        mp.beta_tonal     = betatonal;
        mp.beta_transient = betatransient;
        mp.softness_p     = masksoftness;
        mp.smoothing      = masksmoothing;
        m_engine->set_mask_params(mp);
        m_engine->set_gains_db(gaintonal, gaintransient, gainnoise);
        m_engine->set_bypass(bypass);
        m_engine->set_identity(identitymode);

        const symbol s = solo;
        m_engine->set_solo(s == symbol("tonal")     ? stn::Solo::tonal
                         : s == symbol("transient") ? stn::Solo::transient
                         : s == symbol("noise")     ? stn::Solo::noise
                                                    : stn::Solo::none);

        m_engine->process(input.samples(0),
                          output.samples(0), output.samples(1), output.samples(2),
                          output.samples(3), std::size_t(input.frame_count()));

        if (!sumout) {
            double* o = output.samples(3);
            for (auto i = 0; i < output.frame_count(); ++i) o[i] = 0.0;
        }
    }

private:
    // Heavy-attribute change while DSP is running: build the replacement
    // engine on the calling (non-audio) thread, then swap under the lock.
    // The block that collides with the swap outputs silence — an acceptable
    // Stage A simplification; a click-free crossfade between old and new
    // engines is a straightforward later refinement (run both for D samples).
    void rebuild()
    {
        if (m_sr <= 0.0) return;   // pre-dsp: dspsetup will build
        rebuild_now();
    }

    void rebuild_now()
    {
        auto fresh = std::make_unique<stn::StnEngine>();
        fresh->configure(m_cfg, m_sr);
        {
            std::lock_guard<std::mutex> lock(m_engine_mutex);
            m_engine = std::move(fresh);
        }
        report_latency();
    }

    void report_latency()
    {
        std::lock_guard<std::mutex> lock(m_engine_mutex);
        if (m_engine) m_info.send("latency", int(m_engine->latency()));
    }
};

MIN_EXTERNAL(stn_split);
