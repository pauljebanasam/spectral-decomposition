// test/null_sum_test.cpp
// Headless acceptance tests for the STN core (no Max required).
// Build:  c++ -std=c++17 -O2 -Wall -Wextra -I ../core null_sum_test.cpp -o stn_test && ./stn_test
//
// Tests:
//  1. RealFft vs naive O(n^2) DFT, and round-trip identity.
//  2. COLA deviation for all fft_size x overlap combinations.
//  3. Identity/latency null: masks forced to (1, 0); tonal must equal
//     delay(x, D) to float precision — pins down window, COLA gain, FFT
//     round-trip and the exact latency formula D = N + K*hop.
//  4. Complementary null-sum with live median masks: sum vs delay(x, D)
//     <= -120 dB (acceptance criterion; measured ~ -300 dB, i.e. exact).
//  5. Plausibility: sine + click train + noise floor -> energy routing
//     matrix printed, loose assertions.

#include "stn/engine.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using std::size_t;
static int g_failures = 0;

static void check(bool ok, const std::string& what)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

static double db(double lin) { return 20.0 * std::log10(std::max(lin, 1e-300)); }

static double rms(const std::vector<double>& v, size_t from = 0)
{
    double s = 0.0;
    for (size_t i = from; i < v.size(); ++i) s += v[i] * v[i];
    return std::sqrt(s / double(std::max<size_t>(1, v.size() - from)));
}

static bool all_finite(const std::vector<double>& v)
{
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}

// ---------------------------------------------------------------- test 1
static void test_fft()
{
    std::printf("1. RealFft correctness\n");
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    for (size_t n : {64u, 256u, 4096u}) {
        stn::RealFft fft;
        fft.init(n);
        std::vector<double> x(n), y(n);
        for (auto& s : x) s = uni(rng);
        std::vector<std::complex<double>> X(fft.bins());
        fft.forward(x.data(), X.data());

        double max_err = 0.0;
        if (n <= 256) {   // naive DFT comparison
            for (size_t k = 0; k <= n / 2; ++k) {
                std::complex<double> acc(0.0, 0.0);
                for (size_t i = 0; i < n; ++i)
                    acc += x[i] * std::polar(1.0, -2.0 * M_PI * double(k) * double(i) / double(n));
                max_err = std::max(max_err, std::abs(acc - X[k]));
            }
            check(max_err < 1e-9, "n=" + std::to_string(n) + " forward vs naive DFT, err=" + std::to_string(max_err));
        }
        fft.inverse(X.data(), y.data());
        double rt = 0.0;
        for (size_t i = 0; i < n; ++i) rt = std::max(rt, std::abs(x[i] - y[i]));
        check(rt < 1e-12, "n=" + std::to_string(n) + " round-trip identity, err=" + std::to_string(rt));
        check(std::abs(X[0].imag()) == 0.0 && std::abs(X[n / 2].imag()) == 0.0,
              "n=" + std::to_string(n) + " DC/Nyquist exactly real");
    }
}

// ---------------------------------------------------------------- test 2
static void test_cola()
{
    std::printf("2. COLA verification\n");
    double worst = 0.0;
    for (size_t n : {256u, 1024u, 4096u, 8192u})
        for (size_t ov : {2u, 4u, 8u}) {
            stn::SplitterConfig c;
            c.fft_size = n; c.overlap = ov;
            stn::MedianMaskEstimator est;
            stn::StftSplitter sp;
            sp.configure(c, &est);
            worst = std::max(worst, sp.cola_deviation());
        }
    check(worst < 1e-9, "max COLA deviation over all configs = " + std::to_string(worst));
}

// Mixed program material: sines + click train + noise floor, ~4 s.
struct TestSignal {
    std::vector<double> mix, sine, clicks, floor_;
    std::vector<size_t> click_pos;
};
static TestSignal make_signal(double sr, double dur = 4.0)
{
    TestSignal s;
    const size_t len = size_t(sr * dur);
    s.mix.assign(len, 0.0); s.sine.assign(len, 0.0);
    s.clicks.assign(len, 0.0); s.floor_.assign(len, 0.0);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    for (size_t i = 0; i < len; ++i) {
        const double t = double(i) / sr;
        s.sine[i] = 0.4 * std::sin(2.0 * M_PI * 440.0 * t)
                  + 0.2 * std::sin(2.0 * M_PI * 1330.0 * t + 1.0);
        s.floor_[i] = 0.01 * uni(rng);                       // -40 dB noise floor
    }
    for (double tc = 0.5; tc < dur - 0.25; tc += 0.5) {      // click train
        const size_t p = size_t(tc * sr);
        s.click_pos.push_back(p);
        for (size_t j = 0; j < 32 && p + j < len; ++j)       // 32-sample burst
            s.clicks[p + j] = 0.8 * uni(rng) * (1.0 - double(j) / 32.0);
    }
    for (size_t i = 0; i < len; ++i) s.mix[i] = s.sine[i] + s.clicks[i] + s.floor_[i];
    return s;
}

struct RunResult { std::vector<double> t, tr, n, sum; size_t D; };
static RunResult run_engine(const stn::EngineConfig& cfg, const std::vector<double>& x,
                            double sr, bool identity)
{
    stn::StnEngine eng;
    eng.configure(cfg, sr);
    eng.set_identity(identity);
    RunResult r;
    r.D = eng.latency();
    const size_t len = x.size();
    r.t.assign(len, 0.0); r.tr.assign(len, 0.0); r.n.assign(len, 0.0); r.sum.assign(len, 0.0);
    const size_t block = 61;   // deliberately awkward vector size != hop
    for (size_t i = 0; i < len; i += block) {
        const size_t nn = std::min(block, len - i);
        eng.process(x.data() + i, r.t.data() + i, r.tr.data() + i,
                    r.n.data() + i, r.sum.data() + i, nn);
    }
    return r;
}

// max |a[i] - x[i - D]| relative to signal RMS, in dB
static double null_depth_db(const std::vector<double>& a, const std::vector<double>& x, size_t D)
{
    double max_err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double ref = (i >= D) ? x[i - D] : 0.0;
        max_err = std::max(max_err, std::abs(a[i] - ref));
    }
    const double ref_rms = rms(x);
    return db(max_err / std::max(ref_rms, 1e-30));
}

// ---------------------------------------------------------------- test 3
static void test_identity()
{
    std::printf("3. Identity / latency null (masks forced to 1,0)\n");
    const double sr = 48000.0;
    TestSignal sig = make_signal(sr, 4.0);   // longer than the largest D below

    struct Cfg { size_t n, ov, lh, lp; bool sym; };
    for (const Cfg& c : { Cfg{4096, 4, 17, 31, true},
                          Cfg{4096, 4, 17, 31, false},
                          Cfg{1024, 2,  9, 15, true},
                          Cfg{ 256, 8, 31, 11, true},
                          Cfg{8192, 4, 101, 31, true} }) {
        stn::EngineConfig cfg;
        cfg.fft_size = c.n; cfg.overlap = c.ov;
        cfg.harm_len = c.lh; cfg.perc_len = c.lp;
        cfg.symmetric_median = c.sym;
        RunResult r = run_engine(cfg, sig.mix, sr, true);
        const double d_tonal = null_depth_db(r.t, sig.mix, r.D);
        const double tr_max  = *std::max_element(r.tr.begin(), r.tr.end(),
                                 [](double a, double b){ return std::abs(a) < std::abs(b); });
        const double d_noise = db(rms(r.n) / rms(sig.mix));
        char buf[160];
        std::snprintf(buf, sizeof buf,
            "N=%zu ov=%zu Lh=%zu %s: D=%zu, tonal-vs-delay null=%.1f dB, noise rms=%.1f dB",
            c.n, c.ov, c.lh, c.sym ? "sym" : "caus", r.D, d_tonal, d_noise);
        const bool nonzero = rms(r.t, r.D) > 0.01;   // reject trivially-silent pass
        check(nonzero && d_tonal < -200.0 && std::abs(tr_max) == 0.0 && d_noise < -200.0, buf);
    }
}

// ---------------------------------------------------------------- test 4
static void test_null_sum()
{
    std::printf("4. Complementary null-sum with live median masks\n");
    const double sr = 48000.0;
    TestSignal sig = make_signal(sr);

    stn::EngineConfig cfg;   // defaults: 4096/4, 17/31, symmetric
    RunResult r = run_engine(cfg, sig.mix, sr, false);
    const double d = null_depth_db(r.sum, sig.mix, r.D);
    check(d <= -120.0, "sum vs delay(input): " + std::to_string(d) + " dB (criterion <= -120)");
    check(all_finite(r.t) && all_finite(r.tr) && all_finite(r.n), "all outputs finite");

    // causal mode + non-default params
    cfg.symmetric_median = false; cfg.overlap = 8;
    RunResult r2 = run_engine(cfg, sig.mix, sr, false);
    check(null_depth_db(r2.sum, sig.mix, r2.D) <= -120.0, "null-sum, causal / overlap 8");

    // silence: well-defined masks, zero-ish outputs, no NaN
    std::vector<double> zeros(size_t(sr), 0.0);
    stn::EngineConfig cfg0;
    RunResult r0 = run_engine(cfg0, zeros, sr, false);
    check(all_finite(r0.t) && all_finite(r0.tr) && all_finite(r0.n) &&
          rms(r0.t) == 0.0 && rms(r0.n) == 0.0, "silent input: finite, silent outputs");
}

// ---------------------------------------------------------------- test 5
static void test_plausibility()
{
    std::printf("5. Stream plausibility (energy routing)\n");
    const double sr = 48000.0;
    TestSignal sig = make_signal(sr);
    stn::EngineConfig cfg;
    RunResult r = run_engine(cfg, sig.mix, sr, false);
    const size_t D = r.D, warm = D + size_t(0.5 * sr);

    // Projection of each output stream onto each (delayed) clean component.
    auto proj_frac = [&](const std::vector<double>& out, const std::vector<double>& comp) {
        double dot = 0.0, ce = 0.0;
        for (size_t i = warm; i < out.size(); ++i) {
            const double c = comp[i - D];
            dot += out[i] * c; ce += c * c;
        }
        return (dot * dot) / std::max(ce, 1e-30);   // projection energy onto comp
    };
    auto comp_energy = [&](const std::vector<double>& comp) {
        double e = 0.0;
        for (size_t i = warm; i < r.t.size(); ++i) e += comp[i - D] * comp[i - D];
        return e;
    };

    const double e_sine = comp_energy(sig.sine);
    const double sine_in_tonal = proj_frac(r.t, sig.sine) / e_sine;
    const double sine_in_noise = proj_frac(r.n, sig.sine) / e_sine;

    // Click energy: fraction of transient-stream energy inside +/- N/2 of clicks
    // (Stage A smears transients over the large window — the Stage B motivation).
    const size_t halfwin = cfg.fft_size / 2 + cfg.fft_size / 8;
    double tr_total = 0.0, tr_near = 0.0;
    for (size_t i = warm; i < r.tr.size(); ++i) {
        const double e = r.tr[i] * r.tr[i];
        tr_total += e;
        for (size_t p : sig.click_pos) {
            const size_t pp = p + D;
            if (i + halfwin >= pp && i <= pp + halfwin) { tr_near += e; break; }
        }
    }
    const double tr_frac = tr_near / std::max(tr_total, 1e-30);

    std::printf("     sine->tonal %.1f%%  sine->noise %.1f%%  transient-energy near clicks %.1f%%\n",
                100.0 * sine_in_tonal, 100.0 * sine_in_noise, 100.0 * tr_frac);
    std::printf("     stream RMS (dB rel input): tonal %.1f  transient %.1f  noise %.1f\n",
                db(rms(r.t, warm) / rms(sig.mix)), db(rms(r.tr, warm) / rms(sig.mix)),
                db(rms(r.n, warm) / rms(sig.mix)));

    check(sine_in_tonal > 0.7,  "tonal stream captures majority of sinusoidal energy");
    check(sine_in_noise < 0.15, "sinusoidal leakage into noise stream is small");
    check(tr_frac > 0.5,        "transient stream energy concentrated around clicks");
}

int main()
{
    std::printf("STN core acceptance tests (Stage A)\n===================================\n");
    test_fft();
    test_cola();
    test_identity();
    test_null_sum();
    test_plausibility();
    std::printf("===================================\n%s (%d failure%s)\n",
                g_failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
