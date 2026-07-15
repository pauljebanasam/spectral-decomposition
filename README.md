# stn.split~ — tonal / transient / noise splitter (Stage A)

Real-time STFT median-filtering HPR decomposition (Fitzgerald 2010; Driedger,
Müller & Disch 2014) with **complementary subtractive reconstruction**: the
three signal outlets sum to the latency-delayed input by construction,
independent of mask behaviour.

**SDK: Min-DevKit (min-api).** The DSP core (`core/stn/`, class `StnEngine`)
is pure C++17 with zero Max dependencies, ready for a later JUCE VST/AU wrap.
`stn.split_tilde.cpp` is thin glue: attributes, thread-safe rebuilds, perform.

**Implemented: Stage A** — single resolution, three-way split, null-sum
verified. Stage B (multiresolution two-pass) and Stage C (swappable
classifiers, phase hooks) seams are marked in the headers — see Roadmap.

## Layout

```
stn.split_tilde/
├── stn.split_tilde.cpp      Min-DevKit glue (the Max object)
├── CMakeLists.txt           standard min-devkit project script
├── core/stn/                SDK-agnostic DSP core (header-only)
│   ├── engine.hpp           StnEngine: delay, complementary noise, output stage
│   ├── stft_splitter.hpp    WOLA STFT pipeline, dual masked synthesis, latency
│   ├── mask_estimator.hpp   IMaskEstimator + median estimator (+ Stage C stubs)
│   ├── fft.hpp              real FFT behind a swappable interface
│   └── util.hpp             pow2/odd helpers, ScopedNoDenormals
└── test/
    ├── null_sum_test.cpp    headless core acceptance tests (no Max needed)
    └── glue_smoke_test.cpp  end-to-end test of the Min glue via min-api's
                             mock kernel (build notes in the file header)
```

## Build

Headless core tests first (any platform, no Max):

```sh
cd test
c++ -std=c++17 -O2 -Wall -Wextra -I ../core null_sum_test.cpp -o stn_test
./stn_test
```

The external (macOS shown; Windows analogous with the VS generator):

```sh
git clone --recursive https://github.com/Cycling74/min-devkit.git
cp -r stn.split_tilde min-devkit/source/projects/
cd min-devkit
cmake -S . -B build -G Xcode
cmake --build build --config Release --target stn.split_tilde
# -> min-devkit/externals/stn.split~.mxo
```

Add the `externals` folder to Max's search path (or copy the .mxo into a
package). Min-devkit requires CMake ≥ 3.19 and, on macOS, Xcode command-line
tools.

## Object reference

One signal inlet. Outlets, left to right:

1. **tonal** (signal) — sustained partials
2. **transient** (signal) — onsets/attacks
3. **noise** (signal) — residual: `delay(input) − tonal − transient`
4. **sum** (signal) — `tonal + transient + noise` for null-testing; silent
   unless `@sumout 1`
5. **info** (message) — `latency <samples>`, sent on DSP start, on any
   reconfiguration, and in response to the `latency` message

### Attributes

Heavy (rebuild the engine; audio emits one silent vector during the swap —
click-free crossfade is a noted later refinement):

| attr | range | default | notes |
|---|---|---|---|
| `fftsize` | 256–8192 pow2 | 4096 | Stage A single resolution |
| `overlap` | 2/4/8 | 4 | hop = fftsize/overlap |
| `harmfiltersize` | 3–101 odd | 17 | time-median length (frames) |
| `percfiltersize` | 3–101 odd | 31 | frequency-median length (bins) |
| `latencymode` | causal/symmetric | symmetric | see Latency |

Light (smooth, RT-safe): `betatonal`, `betatransient` (1–5, default 2),
`masksoftness` (0.25–8, default 2), `masksmoothing` (0–1, default 0),
`gaintonal`/`gaintransient`/`gainnoise` (dB, −70 = −inf … +12, ~10 ms ramps),
`solo` (none/tonal/transient/noise), `sumout`, `bypass` (delay-matched),
`identitymode` (debug, see below), `classifier` (median only in Stage A).

**Deviation from the brief — `masksoftness` direction.** The brief's table
described p→0 as binary. The mask is `M = 1/(1 + (βv/h)^p)`, which *hardens*
as p grows and degenerates to 0.5 everywhere as p→0. Implemented accordingly:
range 0.25–8, p = 2 Wiener-like default, larger = harder. Flagged in the
attribute description as well.

### Latency

Exact, derived from the buffering (not estimated):

```
D = fftsize + K · hop,   K = (harmfiltersize − 1)/2 (symmetric) or 0 (causal)
```

Defaults: 4096 + 8·1024 = **12288 samples** (278 ms @ 44.1k). Causal mode:
4096 samples. All four signal outlets are aligned to D.

### Null test patch (acceptance criterion 1)

```
[adc~ / playback] → [stn.split~ @sumout 1]
                      outlet 4 → [-~] ← [delay~ 12288] ← same source
                                   ↓
                              [meter~] / [snapshot~] → expect ≤ −120 dB
```

Use the info outlet (`latency` message) to set the `delay~` time after any
attribute change. `@identitymode 1` runs the same test through outlet 1 with
masks forced to (1, 0), isolating the STFT chain from the classifier.

## Verification results (headless suite, this codebase)

| test | result |
|---|---|
| RealFft vs naive DFT (n=64, 256); round-trip (to 4096) | error < 1e−12 |
| COLA, all fftsize × overlap combinations | deviation ≈ 0 (exact) |
| Identity null (masks 1,0), 5 configs incl. causal, N=256…8192, L_h to 101 | −295 dB or better; latency formula exact |
| Complementary null-sum, live median masks, mixed material | **−309 dB** (criterion ≤ −120) |
| Silence input | finite, silent outputs |
| Plausibility (sines + clicks + −40 dB floor) | sine→tonal 100 %, sine→noise 0 %, transient energy 98.9 % concentrated at clicks |
| Glue end-to-end (mock kernel: construct, attr clamping, dspsetup, perform) | null-sum **−319 dB** through the wrapped Max object |

The glue itself was additionally compiled error-free against the real
min-api headers and driven end-to-end through min-api's mock kernel (see
`test/glue_smoke_test.cpp`, including a Linux link-order note).

Whole suite (~25 s of audio, several configs) runs in ~1.3 s single-threaded
with the reference FFT and naive `nth_element` medians — comfortable
real-time headroom before any optimisation.

## Stage A assumptions and simplifications

- **Single resolution**: the transient mask is computed at the tonal window
  size, so transients are smeared over up to `fftsize` samples (audible as a
  soft attack on the transient stream at 4096). This is precisely the Stage B
  motivation; the plausibility test quantifies it.
- Both masks come from one median pair on the same frames; for β ≥ 1 they
  provably sum ≤ 1 per bin — but null-sum never depends on this, since noise
  is produced by time-domain subtraction.
- Reference FFT is a portable radix-2 with a real pack/unpack behind the
  `RealFft` interface; swap points for pffft/vDSP are documented in `fft.hpp`
  (keep the stated scaling convention).
- Median startup: pre-roll frames read as zeros, so the first `harmfiltersize`
  hops classify toward noise while the history fills. Harmless, documented in
  `stft_splitter.hpp`.
- Heavy-attribute changes emit one silent signal vector during the engine
  swap rather than crossfading two engines.
- Per-sample `tick()` with an amortised hop branch; chunked memcpy processing
  is a marked optimisation if profiling ever asks for it.

## Roadmap seams

- **Stage B** (`engine.hpp`): second `StftSplitter` at `fftsize_transient`
  (default 256) consuming the pass-1 remainder `r1 = delay(x) − tonal`
  sample-by-sample; `D = D1 + D2`. The `tick()` API was shaped for exactly
  this composition.
- **Stage C** (`mask_estimator.hpp`): structure-tensor (Füg et al. 2016) and
  peak-stability (US 10,430,154 B2) estimators against the same
  `SpectralContext`; per-stream phase hook is marked at the mask-application
  point in `stft_splitter.hpp` (noise decorrelation, transient phase-locking,
  RTPGHI).
