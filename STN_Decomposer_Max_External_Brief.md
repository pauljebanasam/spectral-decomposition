# Development Brief: Real-Time Transient / Tonal / Noise Decomposition Max External

## 0. How to use this brief
You (Claude Fable) are being asked to produce the **initial C++ codebase** for a Max/MSP audio external that decomposes an incoming audio stream into three independently-routable output signals — **tonal**, **transient**, and **noise** — for further processing downstream. This code will then be taken into Claude Code to compile the Max device and iterate. Prioritise a **correct, well-structured, incrementally-buildable** codebase over feature completeness. Where a design choice is open, make a sensible expert decision and leave a clearly-commented hook. Assume the reader is an expert DSP engineer and C++ programmer; do not over-explain standard DSP.

---

## 1. Project context and goal

Commercial tools (Eventide SplitEQ, Oeksound Soothe2, Newfangled Elevate) and research toolkits (FluCoMa `fluid.hpss~`, IRCAM SuperVP) can split audio into structural components. The purpose of building a **custom external** rather than using those packages is to escape their fixed architecture and gain three things they cannot offer together:

1. **Multiresolution analysis** — a long analysis window for tonal content (fine frequency resolution) and a short window for transients (fine time resolution) in a single engine, which is the property underlying "transparent" commercial separation.
2. **Three separately-routable, sample-accurate output streams** that **null-sum** back to the input, so downstream processing is unconstrained.
3. **An open representation we compute in and reconstruct from** — a swappable classifier (what counts as "tonal"), independent per-stream phase/time handling hooks, and recursive/feedback extensions — rather than a closed gain-masked black box.

The target algorithm is an **STFT-domain median-filtering Harmonic/Percussive/Residual (HPR) decomposition**, extended to three components with a separation factor, and implemented as a **two-pass multiresolution scheme with complementary (subtractive) reconstruction** so the three streams sum exactly to the delayed input.

---

## 2. Recommended overall strategy and design approach

### 2.1 Architectural principles
- **SDK-agnostic DSP core.** Implement all signal processing in a pure C++ class (e.g. `StnEngine`) with **no dependency on the Max SDK**. The Max object is a thin glue layer that owns an `StnEngine` instance, forwards audio buffers, and maps attributes to engine parameters. This lets the same core later compile to a JUCE VST/AU (the planned v3) without a rewrite.
- **Complementary reconstruction for guaranteed null-sum.** Do **not** rely on three masks summing to unity in the spectral domain across two different FFT sizes (they can't — the resolutions differ). Instead extract streams by **time-domain subtraction**:
  - `tonal = ISTFT( M_tonal ⊙ X_large )`
  - `remainder = delay(input) − tonal`
  - `transient = ISTFT( M_transient ⊙ STFT_small(remainder) )`
  - `noise = delay(remainder) − transient`
  - Therefore `tonal + transient + noise ≡ delay(input)` by construction, regardless of mask imperfections. This mirrors Eventide's patented complementary approach and Driedger's two-pass ordering.
- **Swappable classifier interface.** The function that turns spectrogram frames into a soft mask must be an abstract strategy (e.g. `IMaskEstimator`), with **median-filter ratio** as the first concrete implementation and clearly-marked seams for **structure-tensor** and **spectral-peak-stability** estimators later.
- **Preserve phase by construction; leave phase hooks.** Masks are **real-valued gains** applied to the complex STFT, so original phase is preserved and recombination is transparent. Leave commented hooks for future per-stream phase manipulation (noise decorrelation, transient phase-locking, PGHI reconstruction of transformed magnitude).
- **Build incrementally.** Structure the code so it can be brought up in stages (see §3.6): Stage A single-resolution three-way with verified null-sum; Stage B multiresolution two-pass; Stage C classifier/phase extension hooks.

### 2.2 What this buys us over FluCoMa / SuperVP
Multiresolution matched-reconstruction split; user-definable classification metric; three raw streams (not a remix) for arbitrary downstream routing; independent per-stream phase/time hooks; recursive "onion-peel" residual extraction. These are the justification for the custom build — keep them designed-for even if not all are implemented in v1.

---

## 3. Algorithm design specification (for an expert DSP programmer)

### 3.1 Overview of signal flow
Real-time streaming, block-based STFT processing inside an MSP perform routine. One signal inlet; **three signal outlets** (tonal, transient, noise) plus optionally a fourth (reconstructed sum, for null-testing). Two internal STFT pipelines run at different FFT sizes and are delay-aligned to a common output latency.

```
                 ┌────────────────── PASS 1 (large window, N_h) ──────────────────┐
 x[n] ──┬──────► STFT_large ─► |X| ─► med_time (L_h) = Ỹh
        │                          └─► med_freq (L_p1) = Ỹp1
        │                          ─► tonal soft-mask M_t = f(Ỹh, Ỹp1, β_h, p)
        │                          ─► Tonal_spec = M_t ⊙ X_large ─► ISTFT+OLA ─► tonal[n]
        │
        ├─ delay(D1) ─► r1[n] = x[n−D1] − tonal[n]  (remainder after tonal removed)
        │
        │        ┌────────────────── PASS 2 (small window, N_p) ──────────────────┐
        │        r1[n] ─► STFT_small ─► |R| ─► med_time (L_h2) = R̃h
        │                                  └─► med_freq (L_p2) = R̃p
        │                                  ─► transient soft-mask M_tr = f(R̃p, R̃h, β_p, p)
        │                                  ─► Trans_spec = M_tr ⊙ R_small ─► ISTFT+OLA ─► transient[n]
        │
        └─ delay(D1+D2) alignment; noise[n] = r1[n−D2] − transient[n]
                                    tonal[n]     ─► outlet 1  (aligned to D1+D2)
                                    transient[n] ─► outlet 2
                                    noise[n]     ─► outlet 3
                                    (sum)        ─► outlet 4 (optional, == x[n−(D1+D2)])
```

### 3.2 STFT analysis / synthesis stage
- **Windowing / perfect reconstruction:** use a **√Hann window on both analysis and synthesis** (weighted overlap-add), so the analysis·synthesis product is a Hann window satisfying the Constant-Overlap-Add (COLA) condition. Default **overlap factor 4** (hop = FFT/4). Verify COLA numerically at init for the chosen window/hop and normalise the synthesis gain accordingly.
- **Real FFT** of real input; process the `N/2 + 1` non-negative-frequency bins; enforce Hermitian symmetry on synthesis. Recommended FFT: **pffft** (BSD, portable) or Apple **vDSP/Accelerate** on macOS; keep the FFT behind a small interface so it can be swapped. Do not depend on any FFT bundled with the Max SDK.
- **Streaming buffer management:** maintain, per pipeline, (a) an input accumulation ring buffer; when a hop of new samples has arrived, window the most recent `N` samples, FFT, process, IFFT, and (b) overlap-add into an output ring buffer from which the perform routine reads. Handle arbitrary Max signal-vector sizes (do not assume vector size == hop). Guard against denormals (flush-to-zero or add tiny dither).

### 3.3 Median-filter mask estimation (first classifier)
Given magnitude `Y = |X|` (a spectrogram buffered as a small rolling set of recent frames):
- **Horizontal (time-direction) median** along each frequency bin over `L_h` frames → `Ỹh` (tonal-enhanced: stable partials survive, impulsive spikes removed).
- **Vertical (frequency-direction) median** along each frame over `L_p` bins → `Ỹp` (transient-enhanced: broadband events survive, narrow peaks removed).
- Use an **odd** filter length. The horizontal median requires lookahead of `(L_h−1)/2` frames → this dominates latency (see §3.5). Provide a **causal mode** (backward-only median, zero added lookahead, slightly lower quality) and a **symmetric mode** (centered, higher quality, higher latency).

**Soft (Wiener-style) mask with separation factor β and power p:**
```
M_tonal    = (Ỹh^p) / ( Ỹh^p + (β · Ỹp)^p )          // pass 1
M_transient= (R̃p^p) / ( R̃p^p + (β · R̃h)^p )          // pass 2, computed on remainder
```
- `p` controls softness: `p = 2` ≈ Wiener (default, fewest artifacts); large `p` → binary; expose as a "mask softness" control.
- `β ≥ 1` (separation factor) sets how dominant one orientation must be to claim a bin; `β = 2` is a robust default. Larger β pushes more energy into the residual/noise.
- Because reconstruction is **complementary/subtractive** (§2.1), the transient and noise streams are derived from the remainder, so only the tonal mask (pass 1) and transient mask (pass 2) need be computed — everything not claimed becomes noise automatically. This is what guarantees null-sum.

### 3.4 Multiresolution two-pass parameters
- **Pass 1 (tonal):** large window **N_h = 4096** (default), `β_h = 2`, horizontal median ~ tens of frames.
- **Pass 2 (transient):** small window **N_p = 256** (default), `β_p = 2`, computed on the pass-1 remainder.
- Rationale (state explicitly in comments): large N over-assigns energy to tonal and smears transients; small N blurs partials and over-assigns to transient — so tonal is best extracted at large N and transient at small N, exactly the two-pass ordering. Residual/noise is the leftover after both passes.

### 3.5 Latency accounting
- Report total latency `D = D1 + D2` (in samples) as a read-only attribute and/or a dedicated outlet, and delay-align all outputs (and the optional sum outlet) to `D`.
- `D1 ≈ N_h + (lookahead frames of pass-1 horizontal median) · hop_h`; `D2 ≈ N_p + (lookahead frames of pass-2 median) · hop_p`. Compute precisely from the actual buffering, not by guessing.
- Provide the causal-median mode to minimise latency for near-real-time monitoring; document the quality trade-off.

### 3.6 Incremental build stages (implement in this order)
- **Stage A — single-resolution three-way, null-sum verified.** One FFT size, complementary reconstruction, three outlets. Acceptance: `sum − delay(input)` ≤ ~ −120 dB (float epsilon). Get this rock-solid first.
- **Stage B — multiresolution two-pass.** Add the second pipeline at N_p with time-domain remainder subtraction and delay alignment. Re-verify null-sum.
- **Stage C — extensibility hooks.** Abstract the mask estimator (`IMaskEstimator`); add commented stubs for structure-tensor and peak-stability estimators; add per-stream gain/solo and per-stream phase-hook stubs.

### 3.7 Max SDK glue requirements
- MSP external: 1 signal inlet, 3 (or 4) signal outlets. Implement `dsp64`/`perform64`; support 64-bit signal processing; no allocations in the perform routine (pre-allocate on `dsp64`). Reallocate internal buffers when FFT sizes / sample rate change, outside the audio thread where possible.
- Expose parameters as **attributes** (with ranges, defaults, and clamping) so they are automatable and saved with the patch; also accept them as messages. Changing an FFT size or filter length must rebuild buffers safely (avoid clicks; ramp gains).
- You may target either the **classic Max C SDK (max-sdk / max-api)** or the **Min-DevKit (min-api, C++)**. Recommend **Min-DevKit** for cleaner modern C++, but keep the `StnEngine` core independent of whichever is chosen. State your choice and the folder/project structure at the top of the code.
- Provide a minimal help patch description or comments indicating inlet/outlet semantics.

### 3.8 Numerical and robustness requirements
- Denormal protection throughout; NaN/Inf guards on mask division (add small ε to denominators).
- Correct handling when input is silent (masks well-defined, no divide-by-zero).
- Deterministic, allocation-free audio thread; thread-safe parameter updates.

---

## 4. Recommended papers (underlying concepts and equations)

Implementable primary sources — cite/consult these for exact procedures and equations:

1. **Fitzgerald, D. (2010). "Harmonic/Percussive Separation using Median Filtering." DAFx-10.** — The core median-filtering method (horizontal/vertical median, binary and soft masks). *Foundational for the classifier.*
2. **Driedger, J., Müller, M., Disch, S. (2014). "Extending Harmonic-Percussive Separation of Audio Signals." ISMIR 2014.** — The **three-way** extension: separation factor β, the residual component, frame-size dependence, and the **iterative two-pass multiresolution** procedure (robust defaults β=2, N_h=4096, N_p=256). *This is the primary spec reference for the whole algorithm.*
3. **Serra, X., Smith, J. O. (1990). "Spectral Modeling Synthesis: A Sound Analysis/Synthesis System Based on a Deterministic Plus Stochastic Decomposition." Computer Music Journal 14(4).** — Conceptual origin of sinusoidal-plus-residual thinking. *Context.*
4. **Ono, N., Miyamoto, K., Le Roux, J., Kameoka, H., Sagayama, S. (2008). "Separation of a Monaural Audio Signal into Harmonic/Percussive Components by Complementary Diffusion on Spectrogram." EUSIPCO 2008.** — Origin of the horizontal=harmonic / vertical=percussive observation. *Context.*
5. **Füg, R., Niedermeier, A., Driedger, J., Disch, S., Müller, M. (2016). "Harmonic-Percussive-Residual Sound Separation Using the Structure Tensor on Spectrograms." ICASSP 2016.** — Orientation/anisotropy classifier that rescues vibrato/FM partials from the noise stream. *For the swappable-classifier extension (Stage C).*
6. **Fierro, L., Välimäki, V. (2021). "SiTraNo: A MATLAB App for Sines-Transients-Noise Decomposition." DAFx 2021**, and **Fierro & Välimäki (2023), "Enhanced Fuzzy Decomposition of Sound Into Sines, Transients, and Noise," JAES.** — Fuzzy tripartite membership (soft continuous class membership summing to unity) and artifact analysis. *For a fuzzy classifier and quality tuning.*
7. **Bello, J. P., Duxbury, C., Davies, M., Sandler, M. (2004). "On the Use of Phase and Energy for Musical Onset Detection in the Complex Domain." IEEE Signal Processing Letters 11(6).** — Complex-domain detection function capturing both magnitude jumps and phase deviations. *Optional alternative transient detector.*
8. **Průša, Z., Balazs, P., Søndergaard, P. (2017). "A Noniterative Method for Reconstruction of Phase from STFT Magnitude (PGHI/RTPGHI)." IEEE/ACM TASLP 25(5).** — Real-time phase reconstruction. *For future per-stream phase manipulation of transformed magnitudes.*
9. **US Patent 10,430,154 B2, "Tonal/transient structural separation for audio effects" (Eventide Inc.).** — Discloses spectral-peak **stability** thresholding and Morphological Component Analysis, plus complementary subtractive reconstruction. *For the peak-stability classifier alternative and confirmation of the complementary-reconstruction approach.*

Reference implementations to consult for correctness (not to copy verbatim; mind licences): `librosa.decompose.hpss` (ISC), FluCoMa (BSD-3-Clause), SiTraNo (MIT).

---

## 5. Suggested Max external parameters

Expose the following as attributes (name / type / range / default / description). Group as noted.

### Core analysis
| Name | Range | Default | Description |
|---|---|---|---|
| `fftsize_tonal` | 1024–8192 (pow2) | 4096 | Pass-1 (tonal) FFT/window size — large for frequency resolution. |
| `fftsize_transient` | 128–1024 (pow2) | 256 | Pass-2 (transient) FFT/window size — small for time resolution. |
| `overlap` | 2 / 4 / 8 | 4 | Overlap factor (hop = FFT/overlap) for both passes. |
| `harmfiltersize` | 3–101 (odd) | 17 | Horizontal (time) median length in frames → tonal stability. |
| `percfiltersize` | 3–101 (odd) | 31 | Vertical (frequency) median length in bins → transient breadth. |
| `latency_mode` | causal / symmetric | symmetric | Backward-only vs centered median (latency vs quality). |

### Classification / masking
| Name | Range | Default | Description |
|---|---|---|---|
| `beta_tonal` | 1.0–5.0 | 2.0 | Separation factor for the tonal/remainder split (pass 1). |
| `beta_transient` | 1.0–5.0 | 2.0 | Separation factor for the transient/noise split (pass 2). |
| `mask_softness` (p) | 0.0–2.0 | 2.0 | Mask exponent: 2 ≈ Wiener (smooth), →0 ≈ binary (tight). |
| `mask_smoothing` | 0.0–1.0 | 0.0 | Temporal smoothing of masks to reduce musical-noise/birdies (cf. Eventide "Smoothing"). |
| `classifier` | median / structuretensor / peakstability | median | Which mask estimator to use (Stage C; only `median` in v1). |

### Sonic macro controls (map onto the above)
| Name | Range | Default | Description |
|---|---|---|---|
| `tonalness_bias` | −1.0–+1.0 | 0.0 | Reallocates energy between tonal and noise by biasing `beta_tonal` (negative = more noise, positive = more tonal). |
| `transient_sharpness` | 0.0–1.0 | 0.5 | Maps to `fftsize_transient`/`percfiltersize` — crisper vs softer transient stream. |

### Output / routing / utility
| Name | Range | Default | Description |
|---|---|---|---|
| `gain_tonal` | −∞–+12 dB | 0 | Output gain for the tonal stream. |
| `gain_transient` | −∞–+12 dB | 0 | Output gain for the transient stream. |
| `gain_noise` | −∞–+12 dB | 0 | Output gain for the noise stream. |
| `solo` | none/tonal/transient/noise | none | Solo one stream for auditioning. |
| `sum_outlet` | on / off | off | Enable 4th outlet emitting the summed reconstruction for null-testing. |
| `report_latency` | (read-only, samples) | — | Total processing latency `D1+D2`. |
| `bypass` | on / off | off | True bypass (delay-matched). |

### Future / hooks (stub in v1, commented)
`noise_decorrelation` (0–1, phase decorrelation for stereo width of the noise stream), `phase_lock_transient` (on/off), `oversample_nonlinear` (1×/2×/4×, for any per-stream nonlinear insert), `recursive_passes` (1–4, onion-peel residual extraction).

---

## 6. Acceptance criteria for the initial codebase
1. **Null-sum test passes:** with `sum_outlet` on, `sum − delay(input)` measures ≤ ~ −120 dB across varied program material (Stage A and Stage B).
2. **Three plausible streams:** tonal carries sustained partials, transient carries onsets/attacks, noise carries breath/texture/residual; audibly sensible on drums, sustained synths, and vocal material.
3. **Latency reported accurately** and all outlets delay-aligned (no inter-stream time offset — verify by summing).
4. **Stable, allocation-free audio thread;** no denormal CPU spikes; safe re-init on FFT-size/sample-rate change.
5. **DSP core is Max-SDK-independent** (a pure C++ class), so it can later be wrapped in JUCE for VST/AU.
6. Clear README/comment header stating SDK choice, project/folder structure, build steps, and which stages (A/B/C) are implemented.

Deliver clean, commented, modular C++ with the `StnEngine` core cleanly separated from the Max glue, and note explicitly any assumptions or simplifications made in this first version.
