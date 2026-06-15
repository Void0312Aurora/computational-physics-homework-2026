# Architecture

## Goal

This mainline is designed for throughput-first exploration, not for preserving the semantics of the previous exact prototype.

The working assumption is simple: if the hot path is still dominated by host reconstruction, per-node scheduling, or tiny-window exactness checks, it is not a serious path toward `10M+ digits/s`.

## Core Principles

1. The hot path must stay on the GPU.
2. The execution model must be batched by default.
3. The primary layout must be persistent across levels.
4. Primitive throughput must be measured before end-to-end `pi` claims are made.

## Data Layout

The base layout is:

`residues[modulus][batch][slot]`

This layout is chosen because:

- same-shape groups can be fused into a single batched kernel launch
- the NTT backend can directly treat `batch` as the transform batch dimension
- the scheduler can avoid falling back to `value_count = 1` tensors

## Throughput Stages

### T0

Primitive throughput scaffold.

- persistent packed residue layout
- device-side initialization
- batched pointwise primitive benchmark
- stable CLI and result logging

### T1

Device pack/unpack and shape-group batching.

- group same-shape merges per level
- pack child tensors on-device
- avoid host-side D2D copy loops in the merge hot path
- keep multi-level parent buffers on-device during synthetic reduction passes

### T2

Batched FFT backbone.

- cuFFT-backed or equivalent large-FFT convolution path
- persistent FFT work buffers
- same-shape grouped convolution batches
- first synthetic batched FFT convolution pipeline benchmark exists before semantic binding
- first device-side residue-buffer to FFT-workbuffer bridge exists for same-shape level groups
- grouped multi-modulus residue-to-FFT bridge exists for same-shape group batches

### T3

Binary-splitting tree scheduler on the batched backbone.

- level planner
- persistent level buffers
- no per-node host scheduling in the hot loop
- first grouped level planner smoke benchmark exists with per-level cuFFT plan reuse
- current T3 scaffold also includes a generic `4-bit limb` grouped planner that stabilizes contiguous low-bit masks such as `packing_mask=63` and `packing_mask=4095`
- current T3 scaffold now also reaches exact `mod 2^32` reconstruction through the same generic multi-limb planner
- current T3 scaffold also includes an exact multi-modulus grouped planner smoke path that performs per-modulus accumulation fully on GPU with `2 x 16-bit` limbs and `4` ordered `fp64/Z2Z` passes
- current T3 scaffold now also includes a leaf-backed Chudnovsky `P-factor` path where GPU-generated base-`2^16` leaves feed the same exact grouped planner
- current T3 scaffold now also includes a shared-plan `P/Q` dual-stream path, which proves the grouped planner can carry more than one real binary-splitting product stream without leaving the GPU-resident execution model
- current T3 scaffold now also includes a shared-plan `P/Q/T` grouped path, where the `T` stream executes the real coupled merge rule `T_left*Q_right + P_left*T_right` on the same exact-moduli backbone
- current T3 semantic limit is no longer the lack of a coupled `T` stream; it is the lack of final exact closure and reconstruction on top of this leaf-backed exact grouped planner backbone

### T4

`pi` semantics attach to the fast backbone.

- Chudnovsky leaf generation
- batched merge schedule
- final exact closure isolated from the throughput hot path
- current entry point into T4 is already in place because the leaf-backed grouped scheduler now carries real `P/Q/T` semantics
- a first exact closure smoke now exists and reconstructs exact GPU roots into a correct `pi` prefix on the host through at least the current `128-term` no-wrap smoke point
- the end-to-end smoke path now reuses its last measured grouped-planner execution to capture roots instead of issuing a second full grouped-planner replay for closure
- the current staged closure route now also applies balanced carry normalization after every merge level, which holds the exact coefficient range to a tiny fixed-modulus window instead of letting it explode at the root
- the common `2`-modulus staged-normalization case now also has a device-resident normalization and repack path between levels
- the staged closure backend now also has a more general small-`N` device normalization path that has been validated in forced `3`-moduli and forced `4`-moduli end-to-end runs
- the final host tail now also uses real guard digits and a faster host long-division path, so higher-digit smoke points no longer fail for a fake truncation reason and the tail no longer dominates the hot path
- the end-to-end smoke path now also validates against the available built-in `pi` prefix and can cap emitted decimal output with `--report-decimal-digits`, so larger smoke points are no longer blocked by the embedded `2500`-digit reference or by log blow-up alone
- the measured device-closure path now treats the leaf-digit buffer as immutable first-level input instead of replaying a full device-to-device reset into a work buffer before every measured iteration; that removes a small piece of avoidable traffic, but it does not change the main bottleneck ranking
- the current T4 limit is no longer "invent a closure tail", "reduce closure-tail modulus pressure", "move balanced normalization off the host", "escape the current two-moduli special case", "repair a broken high-digit tail", or even "reduce a dominant host tail", but "reduce the remaining closure wall around the device backbone and the one-time cold-start cost now that the arithmetic tail has fallen away"

## Explicit Differences From `project2_gpu_native_rns`

- No attempt to benchmark a tiny exact window as a throughput proxy.
- No requirement that every iteration rebuild or validate the full root on the host.
- No design centered on single-node tensors as the primary execution grain.

## What Counts As Progress

The main metric here is not yet `digits/s`. Before the real `pi` scheduler is attached, the meaningful metrics are:

- residue values per second
- coefficients per second
- device bytes per second
- kernel launch count per level
- pack/unpack cost relative to arithmetic cost

If those primitives do not improve materially, the final `digits/s` target will not move either.
