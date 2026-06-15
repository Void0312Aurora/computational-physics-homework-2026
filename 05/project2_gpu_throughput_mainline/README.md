# Project2 GPU Throughput Mainline

## Frozen Status

This workspace is frozen as of `2026-04-18`.

It remains useful as a research record for the GPU exact grouped scheduler / staged closure line, but it is no longer the recommended performance mainline for full `pi` computation on this machine.

Use [docs/freeze_snapshot.md](docs/freeze_snapshot.md) as the source of truth for:

- the frozen result set
- the final blocker diagnosis
- the reason this route was stopped
- the conditions under which it would be worth unfreezing

No further incremental optimization is planned on this branch.

This workspace is a fresh throughput-first reboot for `HW/05`.

The previous `project2_gpu_native_rns` line proved that a GPU-native exact prototype can work on a small window, but it also showed that its optimization target is wrong for a `10M/s -> 30M/s` chase. This new workspace starts from a different priority order:

1. Keep the hot path device-resident.
2. Keep the data layout persistent and batched.
3. Measure throughput primitives first.
4. Reintroduce full `pi` semantics only after the backbone is fast enough.

## Non-Goals

- Do not preserve the execution model of `project2_gpu_native_rns`.
- Do not benchmark tiny exact windows and call them throughput.
- Do not keep host-side reconstruction or per-node host scheduling in the hot path.

## Current Status

The current scaffold provides:

- A throughput plan printer: `--print-plan`
- A packed batched pointwise-add microbenchmark: `--pointwise-add-smoke`
- A device pair pack/unpack microbenchmark: `--pair-pack-smoke`
- A persistent multi-level pack+reduce benchmark: `--persistent-level-reduce-smoke`
- A batched cuFFT convolution-backbone benchmark: `--batched-fft-smoke`
- A residue-buffer to cuFFT-workbuffer bridge benchmark: `--residue-fft-bridge-smoke`
- A grouped multi-modulus residue-to-cuFFT bridge benchmark: `--grouped-residue-fft-bridge-smoke`
- A first grouped level planner benchmark: `--grouped-level-planner-smoke`
- A generic multi-limb grouped level planner benchmark: `--grouped-level-planner-multilimb-smoke`
- A compatibility alias for the `mask=63` stability case: `--grouped-level-planner-split-mask63-smoke`
- A `32-bit exact modulo` grouped level planner path via the generic multi-limb runner
- A first exact multi-modulus grouped level planner smoke path: `--grouped-level-planner-exact-moduli-smoke`
- A first leaf-backed exact multi-modulus Chudnovsky `P-factor` path: `--grouped-level-planner-exact-moduli-pfactor-smoke`
- A first shared-plan exact multi-modulus Chudnovsky `P/Q` dual-stream path: `--grouped-level-planner-exact-moduli-pq-smoke`
- A first shared-plan exact multi-modulus Chudnovsky `P/Q/T` grouped scheduler path: `--grouped-level-planner-exact-moduli-pqt-smoke`
- A first end-to-end `pi` closure smoke path with expandable exact-moduli closure capacity: `--pi-end-to-end-smoke`
- A clean CUDA/C++ workspace with docs, build rules, and a benchmark-oriented CLI

This microbenchmark is not `pi digits/s`. It is the first primitive meter for the new mainline.
The residue-to-FFT bridge currently packs masked low-bit residue values into floating-point FFT inputs.
That makes it a throughput bridge for execution structure, not a final exact RNS convolution path.
The grouped bridge extends that idea across the full modulus batch, but it is still a masked floating bridge rather than a final exact multi-modulus convolution path.
The grouped level planner now keeps the level buffers on GPU and reuses one cuFFT plan per level shape, but its projection semantics are only numerically stable for conservative masks such as `packing_mask=15` at `4096 x 4096 x 8`.
At `packing_mask=63`, the benchmark still runs and reports throughput, but the report will flag projection mismatches because single-precision cuFFT error is large enough to flip low bits after rounding.
The new generic multi-limb planner addresses that blocker by splitting each masked coefficient into auto-generated `4-bit` limbs and reconstructing the requested low-bit window from grouped FFT passes whose shifts still contribute below the mask width.
For `packing_mask=63`, that reduces to the old `4+2` / `3-pass` route.
For `packing_mask=4095`, it expands to a `4+4+4` / `6-pass` route.
For `packing_mask=4294967295`, it now runs as an exact `32-bit modulo` route, switches to a numerically safer `fp64/Z2Z` backbone, and uses a much wider `16-bit` limb split.
That exact32 route is no longer a `mask=63` special case or a low-bit-only toy.
On top of it, the new exact-moduli smoke path performs exact per-modulus accumulation on GPU with `2 x 16-bit` limbs and `4` ordered `fp64/Z2Z` passes (`low*low`, `low*high`, `high*low`, `high*high`), then validates sampled parent residues against exact modular references.
The next step is no longer purely synthetic either: the new `pfactor` variant initializes leaf nodes on GPU as base-`2^16` coefficient vectors for the absolute Chudnovsky `P` factors `|(6k-5)(2k-1)(6k-1)|`, then runs the same exact-moduli tree on those real binary-splitting inputs.
That path has now been extended again into a shared-plan `P/Q/T` grouped scheduler benchmark: `P`, `Q`, and `T` leaves are all generated on GPU, the level schedule and cuFFT plan set are shared, and the `T` stream now executes the real coupled merge rule `T_left*Q_right + P_left*T_right` instead of pretending to be another independent product stream.
That grouped scheduler benchmark by itself is still not end-to-end `pi`, because it stops at modular root buffers rather than performing final closure.
A first isolated closure tail now exists as a separate smoke path and can reconstruct and emit correct `pi` prefixes through at least the current `128-term` no-wrap smoke regime by expanding the exact-moduli pool when needed.
The end-to-end smoke path now also reuses the last measured grouped-planner execution to capture the final roots instead of replaying the whole merge tree a second time just for closure.
That route has now been replaced again by a stronger staged closure path: every GPU merge level is followed by host-side balanced carry normalization back into base-`2^16` digits, which keeps the next level's exact coefficient range small enough that only a tiny fixed modulus subset is needed.
That staged route now has two device-resident closure modes: a specialized fast path for `effective_closure_modulus_count=2`, and a more general small-`N` device normalization path that has now been validated through forced `3`-moduli and `4`-moduli end-to-end smoke runs.
The final host tail has also been corrected and then accelerated: the `sqrt(10005)` plus division route now really uses `working_digits` guard precision, the generic bit-by-bit host division has been replaced with a limb-wise long division path, and the post-division decimal truncation now drops `10^k` in `1e9` chunks instead of issuing another generic big division.
The end-to-end smoke path is no longer capped by the embedded `2500`-digit reference prefix either: it now validates against the available built-in prefix, can continue past that bound, and can cap the emitted decimal string with `--report-decimal-digits` so larger throughput probes do not drown in their own logs.
The measured closure path now also reuses the immutable device leaf-digit buffer as the first-level input instead of copying it into a fresh working buffer every measured iteration; that cleanup is structurally correct, but its measured gain is only marginal next to the remaining closure wall.
The remaining issue is no longer "can we move that balanced normalization and repack work off the host?", "is this still stuck at a narrow two-moduli special case?", "is the high-digit smoke failing because the tail lost guard digits?", or even "does the final host tail dominate once the smoke reaches the low-thousands-digit range?".
It is now "how much of the remaining end-to-end wall time sits in the device closure path and one-time process overhead, now that the closure window stays at `2` moduli with comfortable headroom and the host tail has fallen to a small fraction of the hot path?"

## Latest Benchmark Snapshot

Measured on the current workstation with `4096 x 4096 x 8` grouped planner smoke:

- `packing_mask=15`: `avg_pipeline_ms ~= 20.13`, `packed_residue_values_per_second ~= 1.33e10`, `verification_mismatch_count = 0`
- `packing_mask=63`: `avg_pipeline_ms ~= 20.11`, `packed_residue_values_per_second ~= 1.33e10`, `verification_mismatch_count = 4`, `max_projection_real_error = 0.5`
- `multilimb-mask63`: `avg_pipeline_ms ~= 62.74`, `packed_residue_values_per_second ~= 4.28e9`, `split_limb_count = 2`, `split_pass_count = 3`, `verification_mismatch_count = 0`
- `multilimb-mask4095`: `avg_pipeline_ms ~= 124.67`, `packed_residue_values_per_second ~= 2.15e9`, `split_limb_count = 3`, `split_pass_count = 6`, `verification_mismatch_count = 0`
- `multilimb-mask32full`: `avg_pipeline_ms ~= 283.31`, `packed_residue_values_per_second ~= 9.47e8`, `split_limb_count = 2`, `split_pass_count = 3`, `verification_mismatch_count = 0`
- `exact-moduli-32full`: `avg_pipeline_ms ~= 375.87`, `packed_residue_values_per_second ~= 7.14e8`, `split_limb_count = 2`, `split_pass_count = 4`, `verification_mismatch_count = 0`
- `exact-moduli-pfactor-32full`: `avg_pipeline_ms ~= 375.00`, `packed_residue_values_per_second ~= 7.16e8`, `split_limb_count = 2`, `split_pass_count = 4`, `verification_mismatch_count = 0`
- `exact-moduli-pq-32full`: `avg_pipeline_ms ~= 756.01`, `packed_residue_values_per_second ~= 7.10e8`, `split_limb_count = 2`, `split_pass_count = 4`, `verification_mismatch_count = 0`
- `exact-moduli-pqt-32full`: `avg_pipeline_ms ~= 1507.89`, `packed_residue_values_per_second ~= 7.12e8`, `split_limb_count = 2`, `split_pass_count = 4`, `verification_mismatch_count = 0`
- `pi-end-to-end-smoke-16x256x10-80d`: `planner_avg_pipeline_ms ~= 2.19`, `effective_closure_modulus_count = 2`, `crt_product_bits = 62`, `root_reconstruction_match = 1`, `prefix_match = 1`
- `pi-end-to-end-smoke-32x1024x10-80d`: `planner_avg_pipeline_ms ~= 8.33`, `effective_closure_modulus_count = 2`, `crt_product_bits = 62`, `root_reconstruction_match = 1`, `prefix_match = 1`
- `pi-end-to-end-smoke-64x1024x10-80d`: `planner_avg_pipeline_ms ~= 10.09`, `effective_closure_modulus_count = 2`, `crt_product_bits = 62`, `root_reconstruction_match = 1`, `prefix_match = 1`
- `pi-end-to-end-smoke-128x1024x10-80d`: `planner_avg_pipeline_ms ~= 12.82`, `effective_closure_modulus_count = 2`, `crt_product_bits = 62`, `root_reconstruction_match = 1`, `prefix_match = 1`
- `pi-end-to-end-smoke-128x1024x10-80d-forced3`: `planner_avg_pipeline_ms ~= 93.58`, `effective_closure_modulus_count = 3`, `crt_product_bits = 91`, `root_reconstruction_match = 1`, `prefix_match = 1`
- `pi-end-to-end-smoke-128x1024x10-80d-forced4`: `planner_avg_pipeline_ms ~= 131.20`, `effective_closure_modulus_count = 4`, `crt_product_bits = 122`, `root_reconstruction_match = 1`, `prefix_match = 1`
- `pi-end-to-end-smoke-128x1024x10-1000d`: `effective_closure_modulus_count = 2`, `required_closure_half_range_bits = 42`, `closure_modulus_headroom_bits = 19`, `planner_avg_pipeline_ms ~= 12.94`, `closure_wall_ms ~= 268.89`, `final_host_tail_ms ~= 0.55`, `prefix_match = 1`
- `pi-end-to-end-smoke-256x2048x10-2000d`: `effective_closure_modulus_count = 2`, `required_closure_half_range_bits = 43`, `closure_modulus_headroom_bits = 18`, `planner_avg_pipeline_ms ~= 34.78`, `closure_wall_ms ~= 408.06`, `final_host_tail_ms ~= 2.55`, `prefix_match = 1`
- `pi-end-to-end-smoke-256x2048x10-2500d-breakdown`: `effective_closure_modulus_count = 2`, `required_closure_half_range_bits = 43`, `closure_modulus_headroom_bits = 18`, `cuda_runtime_init_ms ~= 210.90`, `closure_setup_ms ~= 14.99`, `closure_measured_total_ms ~= 102.92`, `closure_wall_ms ~= 190.28`, `final_host_tail_ms ~= 3.12`, `prefix_match = 1`
- `pi-end-to-end-smoke-512x4096x10-5000d-prefix128`: `effective_closure_modulus_count = 2`, `required_closure_half_range_bits = 44`, `closure_modulus_headroom_bits = 17`, `planner_avg_pipeline_ms ~= 90.44`, `root_rebuild_ms ~= 14.32`, `final_host_tail_ms ~= 10.38`, `steady_state_pi_result_ms ~= 115.15`, `steady_state_pi_digits_per_second ~= 4.34e4`, `cold_process_pi_digits_per_second ~= 6.49e3`, `reference_prefix_digits_checked = 2500`, `prefix_match = 1`

That result is important: the throughput backbone is no longer the immediate blocker for T3, and we now have a numerically stable generic multi-limb route that reaches exact `32-bit modulo` semantics.
The new exact-moduli benchmark matters for a different reason: it shows that moving from exact `mod 2^32` to true per-modulus exact accumulation costs roughly the expected extra pass, not an accidental collapse into host-bound execution.
The new `pfactor` result matters because it shows that attaching a real Chudnovsky leaf generator for one binary-splitting stream does not materially damage throughput relative to the synthetic exact-moduli scaffold.
The new `pq` result matters because it shows the backbone scales to multiple real product streams with shared planning and without a throughput collapse; doubling the work roughly doubles the wall time instead of triggering a worse structural penalty.
The new `pqt` result matters because it replaces the fake "third product stream" shortcut with the real coupled `T` semantics and still lands at roughly the same residue-throughput band as the `P/Q` run; adding the mixed `TQ + PT` work roughly doubles wall time again, but does not trigger a structural collapse back toward the old host-driven prototype regime.
The new `pi-end-to-end-smoke` result matters because it proves the new mainline can now produce a real `pi_decimal` from its own GPU `P/Q/T` roots instead of stopping at grouped planner metrics.
The old failed `32x1024x10` closure smoke still matters because it isolates the original failure mode cleanly: root-level coefficient reconstruction with no intermediate normalization forces modulus growth.
The new staged balanced-normalization closure smokes matter for the opposite reason: they show that exact closure no longer needs `18` or `36` moduli just to cross `32-term` and `64-term`.
The new forced-`3` and forced-`4` closure runs matter because they prove the staged exact closure backend is no longer locked to the `2`-modulus case.
The new `1000d`, `2000d`, `2500d`, and `5000d` runs matter for a different reason again: they show that the auto-selected closure window is still only `2` moduli with `17-19` bits of headroom, so the next scaling wall is not closure modulus pressure.
The host-division rewrite changes the picture sharply again: fixed-modulus exact closure is still structurally ahead of modulus pressure, but the final host tail is no longer the dominant hot cost either.
The new timing split makes the next blocker harder to misread: at `2500` digits, roughly `210.90 ms` is one-time CUDA runtime startup, about `102.92 ms` is the three measured closure passes, and only about `3.12 ms` is the final host `sqrt/div` tail.
At `5000` digits, the steady-state single-result path is about `115.15 ms` or `4.34e4 digits/s`, while the cold whole-process route is still only about `6.49e3 digits/s`.
That means the remaining work is back on the closure backbone and the surrounding one-time process overhead, not on the arithmetic tail or the old embedded-prefix smoke limit.

## Workspace Layout

- `docs/architecture.md`: throughput-first architecture and invariants
- `docs/roadmap.md`: staged execution plan
- `include/project2_gpu_throughput_mainline/runtime.cuh`: public runtime declarations
- `src/runtime.cu`: CUDA runtime and throughput microbenchmark
- `src/main.cu`: CLI entry point

## Build

```bash
make -C HW/05/project2_gpu_throughput_mainline build
```

## Example Commands

Print the mainline plan:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline --print-plan
```

Run the packed pointwise-add smoke benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --pointwise-add-smoke \
  --batch-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --iterations 20
```

Run the device pair pack/unpack benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --pair-pack-smoke \
  --merge-count 2048 \
  --slot-count 4096 \
  --modulus-count 8 \
  --iterations 20
```

Run the persistent multi-level pack+reduce benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --persistent-level-reduce-smoke \
  --node-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --iterations 20
```

Run the batched cuFFT backbone benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --batched-fft-smoke \
  --fft-batch-count 2048 \
  --fft-length 4096 \
  --iterations 20
```

Run the residue-buffer to cuFFT bridge benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --residue-fft-bridge-smoke \
  --node-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --bridge-modulus-index 0 \
  --packing-mask 63 \
  --iterations 20
```

Run the grouped multi-modulus residue-to-cuFFT bridge benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --grouped-residue-fft-bridge-smoke \
  --node-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --packing-mask 63 \
  --iterations 20
```

Run the first grouped level planner benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --grouped-level-planner-smoke \
  --node-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --packing-mask 15 \
  --iterations 20
```

The default `make groupedlevelplanner` target uses `packing_mask=15` because that configuration is currently stable enough to pass the planner's first-level projection check.
If you run the same benchmark with `packing_mask=63`, expect the log to include `verification_mismatch_count > 0`; that is the current numerical blocker, not a hidden host fallback.

Run the generic multi-limb grouped level planner benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --grouped-level-planner-multilimb-smoke \
  --node-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --packing-mask 4095 \
  --iterations 20
```

The generic multi-limb path uses auto-generated `4-bit` limbs and includes every ordered limb pair whose shift still contributes below the target low-bit width.
That means:

- `packing_mask=63` becomes `2 limbs / 3 passes`
- `packing_mask=4095` becomes `3 limbs / 6 passes`
- `packing_mask=4294967295` becomes `2 limbs / 3 passes` on the current benchmark because the runner switches to `fp64` and widens the limb size to `16 bits`

Then it reconstructs the requested low-bit parent window on GPU.
The old `--grouped-level-planner-split-mask63-smoke` command is still available as a compatibility alias, but it now routes through the same generic multi-limb runner.
For the `4294967295` case, the same runner now gives exact `mod 2^32` reconstruction on the benchmarked grouped planner path.
This is much more stable than the single-pass masked float planner, but it is not yet the final exact multi-modulus RNS path.

Run the current exact `32-bit modulo` grouped level planner benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --grouped-level-planner-multilimb-smoke \
  --node-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --packing-mask 4294967295 \
  --iterations 3
```

Run the first exact multi-modulus grouped level planner smoke benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --grouped-level-planner-exact-moduli-smoke \
  --node-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --packing-mask 4294967295 \
  --iterations 3
```

This path keeps exact per-modulus accumulation on GPU and is the current best approximation of the future exact RNS grouped merge backbone.
It is still fed by synthetic grouped level values, so its throughput should be interpreted as a T3 backbone metric rather than final `pi digits/s`.

Run the leaf-backed Chudnovsky `P-factor` exact multi-modulus grouped level planner benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --grouped-level-planner-exact-moduli-pfactor-smoke \
  --node-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --packing-mask 4294967295 \
  --iterations 3
```

This path generates the absolute Chudnovsky `P` factors on GPU, encodes them as base-`2^16` leaf vectors, and then reduces them with the same exact-moduli grouped planner.
At large `4096 x 4096` sizes the tree is intentionally treated as a throughput benchmark rather than a no-wrap root reconstruction benchmark, because the cyclic FFT length is smaller than the full unreduced polynomial support.

Run the shared-plan Chudnovsky `P/Q` dual-stream exact multi-modulus grouped level planner benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --grouped-level-planner-exact-moduli-pq-smoke \
  --node-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --packing-mask 4294967295 \
  --iterations 3
```

This path keeps two real binary-splitting product streams resident on GPU: `P` and `Q`.
The current implementation shares the same plan set and level schedule across both streams, but still executes the two streams sequentially inside each level; that is deliberate and keeps the new scheduler step honest about what has and has not been fused yet.

Run the shared-plan Chudnovsky `P/Q/T` exact multi-modulus grouped level planner benchmark:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --grouped-level-planner-exact-moduli-pqt-smoke \
  --node-count 4096 \
  --slot-count 4096 \
  --modulus-count 8 \
  --packing-mask 4294967295 \
  --iterations 3
```

This path keeps three real binary-splitting streams resident on GPU and uses the real `T_left*Q_right + P_left*T_right` merge semantics for the `T` stream.
`P` and `Q` remain pure product streams, while `T` performs two mixed convolutions per pass and accumulates both into the same modular parent buffer.
The current implementation is still honest about fusion boundaries: the `P`, `Q`, and coupled `T` work share one per-level plan set and one grouped schedule, but the mixed convolutions still execute sequentially inside each level.

Run the quick end-to-end `pi` closure regression smoke:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --pi-end-to-end-smoke \
  --term-count 16 \
  --slot-count 256 \
  --modulus-count 10 \
  --target-digits 80 \
  --warmup 1 \
  --iterations 3
```

This path reuses the GPU-resident `P/Q/T` grouped scheduler, captures the final root digit buffers once, rebuilds the exact roots on the host from balanced base-`2^16` digits, and then performs the isolated final `sqrt(10005)` and division tail.
It is still a closure smoke rather than a throughput claim.
At `term-count=16`, `slot-count=256`, and `modulus-count=10`, it reconstructs the roots exactly and emits the correct `80`-digit `pi` prefix.
At `term-count=32`, `64`, and even `128` with the current `slot-count=1024`, the same closure path now still passes with the requested `modulus-count=10`, but the staged balanced-normalization route only needs an effective exact subset of `2` moduli at each level because it renormalizes the digits after every merge.
The current default route keeps that normalization and repack work on the GPU between levels and only copies the final root digits back for the last host `sqrt(10005)` and division tail.
That removes the old linear-modulus-growth blocker and most of the intermediate host bottleneck for the default small-window regime.

Run the scaled end-to-end `pi` closure smoke at the current `128-term` checkpoint:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --pi-end-to-end-smoke \
  --term-count 128 \
  --slot-count 1024 \
  --modulus-count 10 \
  --target-digits 80 \
  --warmup 1 \
  --iterations 3
```

Run the forced `3`-moduli device-closure validation pass:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --pi-end-to-end-smoke \
  --term-count 128 \
  --slot-count 1024 \
  --modulus-count 10 \
  --force-closure-modulus-count 3 \
  --target-digits 80 \
  --warmup 1 \
  --iterations 3
```

The `--force-closure-modulus-count` flag is a validation knob, not a new default throughput mode.
It is there so the small-`N` device closure path can be exercised honestly before the auto-selected closure window naturally grows past `2`.

Run the current high-digit closure-wall probe:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --pi-end-to-end-smoke \
  --term-count 256 \
  --slot-count 2048 \
  --modulus-count 10 \
  --target-digits 2500 \
  --report-decimal-digits 256 \
  --warmup 1 \
  --iterations 3
```

This run is useful because it exposes the current structural crossover after the host-division rewrite:
the device closure still stays at `2` moduli with comfortable half-range headroom, the final host `sqrt/div` tail is now only a few milliseconds, and the remaining end-to-end wall time sits elsewhere in the closure path.

Run a larger smoke beyond the embedded reference window while keeping logs short:

```bash
./HW/05/project2_gpu_throughput_mainline/bin/project2_gpu_throughput_mainline \
  --pi-end-to-end-smoke \
  --term-count 512 \
  --slot-count 4096 \
  --modulus-count 10 \
  --target-digits 5000 \
  --report-decimal-digits 128 \
  --warmup 1 \
  --iterations 3
```

This run only checks the built-in `2500`-digit prefix for correctness, but it still exercises the full higher-digit closure and host tail.

## Immediate Next Steps

1. Reduce the remaining device closure wall and identify how much of it is true hot-path planner cost versus one-time process overhead.
2. Keep the closure headroom and timing breakdown metrics in the loop so we do not regress back into fake bottleneck attribution.
3. Only after that compare the resulting end-to-end route honestly against the CPU and hybrid baselines.
