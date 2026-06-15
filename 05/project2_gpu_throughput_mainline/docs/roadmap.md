# Roadmap

Frozen on `2026-04-18`.

This roadmap is preserved as a historical planning document for the throughput-mainline exploration.
The active disposition of this route is now defined by [freeze_snapshot.md](freeze_snapshot.md), not by the unfinished exit criteria below.

## Stage 0

Scaffold and primitive benchmark.

Exit criteria:

- workspace builds cleanly
- throughput plan is documented
- packed pointwise-add benchmark runs and logs results

## Stage 1

Device pack/unpack kernel path.

Exit criteria:

- host-driven D2D packing loops are removed from the benchmark hot path
- same-shape batches can be packed and unpacked with dedicated kernels
- persistent multi-level pack+reduce survives across level buffers without node-major reconstruction

## Stage 2

Batched convolution backbone.

Exit criteria:

- first batched cuFFT or equivalent convolution-backbone benchmark runs and logs results
- first device-side residue-to-FFT bridge benchmark runs and logs results
- first grouped multi-modulus residue-to-FFT bridge benchmark runs and logs results
- same-shape grouped convolutions execute as true batched FFT work
- workspace has a persistent FFT buffer strategy

## Stage 3

Binary-splitting scheduler.

Current status:

- exact `mod 2^32` grouped planner path is working
- first exact multi-modulus grouped planner smoke benchmark is working
- first leaf-backed Chudnovsky `P-factor` grouped planner benchmark is working
- first shared-plan leaf-backed Chudnovsky `P/Q` grouped planner benchmark is working
- first shared-plan leaf-backed Chudnovsky `P/Q/T` grouped planner benchmark is working
- Stage 3 is no longer blocked on the coupled `T` stream or on basic exact grouped accumulation

Exit criteria:

- level planner exists
- grouped merges survive across levels without falling back to single-node scheduling
- planner projection remains numerically stable at the target packing range, not only at conservative masks
- exact grouped accumulation is promoted beyond a single ring into a true multi-modulus path
- the exact grouped path is attached to real binary-splitting scheduler inputs rather than staying synthetic, single-stream, or product-only forever

## Stage 4

Attach real `pi` semantics.

Current status:

- leaf-backed grouped `P/Q/T` scheduler path is working on the throughput-first backbone
- end-to-end closure now passes at `16-term/10-moduli`, `32-term/10-moduli`, `64-term/10-moduli`, and `128-term/10-moduli` no-wrap smoke points
- the staged closure route now uses per-level balanced normalization, so exact closure no longer needs linearly more moduli as the smoke scale rises
- the common `2`-modulus staged-normalization case now has a device-resident fast path, which removes the earlier host-side normalization bottleneck from these smoke points
- the device-resident staged-normalization route has now been widened again into a validated small-`N` closure backend: forced `3`-moduli and forced `4`-moduli end-to-end smoke points both pass
- the final host `sqrt(10005)` plus division tail now correctly uses `working_digits` guard precision, and the new limb-wise host division path has collapsed that tail from the old `10^2 ms` band into the low single-digit `ms` band at `2500d`
- the end-to-end smoke path is no longer hard-capped by the embedded `2500`-digit reference prefix, because it now validates against the available built-in prefix and can truncate reported decimal output with `--report-decimal-digits`
- the measured device-closure path now reuses the immutable leaf-digit buffer directly at level `0` instead of replaying a full `d_leaf_digits -> d_digits_current` reset every measured iteration, but the observed gain is still small relative to the remaining closure wall
- the new closure headroom metrics show that the auto-selected closure window still stays at `2` moduli with `17-19` bits of half-range margin on the current scaled smokes
- the new timing split shows that the remaining Stage 4 wall is now a mix of device closure cost and one-time per-process CUDA startup rather than modulus dynamic range, basic host normalization throughput, a hard `2`-modulus specialization wall, or a broken/dominant high-digit tail
- the current `512-term / 4096-slot / 5000-digit` smoke reaches about `4.34e4 digits/s` in the new steady-state single-result metric but only about `6.49e3 digits/s` as a cold whole-process run, which sharpens the next optimization target

Exit criteria:

- Chudnovsky leaves are generated on the new backbone
- end-to-end route exists without destroying the throughput-oriented execution model
- final exact closure and reconstruction remain isolated from the throughput hot path
- end-to-end closure scales beyond the tiny smoke regime without immediately exhausting modulus dynamic range
- closure no longer needs a near-linear increase in CRT modulus count just to advance one modest scale step
- balanced normalization and digit repack no longer dominate the staged closure route for the current `2`-modulus fast path
- the device-resident staged-normalization route covers the practical small-`N` closure window instead of stopping at `2` moduli
- larger-scale end-to-end runs keep closure headroom positive while identifying whether the remaining blocker lives in the device closure wall rather than the host tail
- larger-scale end-to-end runs are no longer blocked by the built-in reference-prefix cap or by dumping full decimal expansions into the logs
- the final host tail no longer loses correctness at higher target digits because guard digits are actually applied

## Kill Criteria

This new mainline should be abandoned or reset again if:

- the hot path becomes host-driven
- grouped batches are immediately unpacked back to single-node tensors every level
- the benchmark focus drifts back to tiny exactness windows instead of throughput primitives
- T4 work escapes modulus-growth failure only by leaning on a toy closure window that cannot survive beyond the current small-`N` device backend
- the high-digit route remains "correct" only because smoke validation was weakened instead of fixing the final tail properly
