// test/glue_smoke_test.cpp
// Headless end-to-end test of the *Min glue* (not just the core) using
// min-api's mock kernel: constructs stn.split~, exercises attribute clamping
// through the real min-api setters, fires dspsetup, streams audio through
// operator() via audio_bundle, and verifies the null-sum at the object's
// outlets. No Max runtime required.
//
// Build (from this directory, with min-api cloned as a sibling of the
// project or inside min-devkit as usual):
//   MIN=path/to/min-api
//   c++ -std=c++17 -O2 -w -I $MIN/include -I $MIN/test \
//       -I $MIN/max-sdk-base/c74support \
//       -I $MIN/max-sdk-base/c74support/max-includes \
//       -I ../core $MIN/test/mock/c74_mock.cpp glue_smoke_test.cpp -o glue_smoke
//   ./glue_smoke
//
// NOTE (Linux/GNU ld): list the mock TU (c74_mock.cpp) BEFORE this file.
// GNU ld runs static initialisers in reverse link order; the min headers
// create static symbols at init time that call into the mock kernel's
// tables, and the wrong order SIGFPEs before main. (min-api's bundled
// Catch harness hits the same issue; this plain-main test avoids Catch.)
#include "c74_min.h"
#include "../stn.split_tilde.cpp"
#include <cstdio>
#include <cmath>
#include <vector>
using namespace c74::min;

int main() {
    std::printf("construct...\n");
    stn_split obj;   // direct construction (attribute defaults + setters)
    std::printf("attrs...\n");
    obj.fftsize = 3000;
    if (int(obj.fftsize) != 2048) { std::printf("FAIL pow2 clamp\n"); return 1; }
    obj.fftsize = 1024;
    obj.harmfiltersize = 8;
    if (int(obj.harmfiltersize) != 9) { std::printf("FAIL odd clamp\n"); return 1; }
    obj.sumout = true;

    std::printf("dspsetup...\n");
    atoms setup_args { 48000.0, 64.0 };
    obj.dspsetup(setup_args);

    const size_t D = 1024 + 4 * 256;   // fftsize + K*hop; harm=9 -> K=4, hop=256
    const size_t len = 48000;
    std::vector<double> x(len), t(len), tr(len), n(len), s(len);
    const double sr = 48000.0;
    for (size_t i = 0; i < len; ++i)
        x[i] = 0.5 * std::sin(2.0 * M_PI * 220.0 * double(i) / sr)
             + ((i % 12000) < 8 ? 0.7 : 0.0);

    std::printf("perform...\n");
    const int vs = 64;
    for (size_t i = 0; i < len; i += vs) {
        double* ins[1]  = { x.data() + i };
        double* outs[4] = { t.data() + i, tr.data() + i, n.data() + i, s.data() + i };
        audio_bundle in_b(ins, 1, vs);
        audio_bundle out_b(outs, 4, vs);
        obj(in_b, out_b);
    }
    double max_err = 0.0, out_e = 0.0;
    for (size_t i = D; i < len; ++i) {
        max_err = std::max(max_err, std::abs(s[i] - x[i - D]));
        out_e += t[i] * t[i];
    }
    std::printf("out energy %.3f, null-sum max err %.3e (%.1f dB)\n",
                out_e, max_err, 20.0 * std::log10(std::max(max_err, 1e-300)));
    if (out_e <= 0.0 || max_err > 1e-12) { std::printf("FAIL\n"); return 1; }
    std::printf("GLUE END-TO-END PASS\n");
    return 0;
}
