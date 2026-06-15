# Plan

## Scope

- Target folder: `HW/13`
- Problem source: `docs/problems/计算物理lecture(IsingModel)20261.pdf`, pages 62--72.
- Source heading: Homework 14, dated 06/10/2026.

## Problem Summary

- Problem 1 asks for a one-dimensional Ising model with the Metropolis method:
  - compare ordered and random initial states for `N=20`, `T=1.0`;
  - estimate equilibration time;
  - scan `T=0.5` to `5.0` and compare mean energy with `-N tanh(J/kBT)`;
  - discuss magnetization and acceptance ratio.
- Problem 2 asks for a one-dimensional demon-algorithm simulation:
  - use `N=100`, `J=1`, `h=0`, desired total energy `E_ini=-20`;
  - measure running and equilibrium values of demon energy, magnetization, and squared magnetization;
  - infer temperature from the demon-energy formula;
  - repeat for `E_ini=-40,-60,-80` and discuss finite-size and run-length effects.
- Problem 3 asks for a two-dimensional square-lattice Ising model:
  - thermal bath, no external field, spin-flip dynamics, PBC, `kB=J=1`;
  - study trajectories at `L=30`, `T=2` and `T=4`;
  - compare initial configurations, temperature scans, `L=4`, open boundaries, random/order spin selection, and different sampling cadences.
- Problem 4 asks for MCMC magnetization of the 2D Ising model:
  - plot magnetization versus temperature for different system sizes;
  - discuss phase transition, finite-size effects, numerical accuracy, and critical-exponent fitting from `ln(<m^2>)` versus `ln(L)`;
  - provide representative `100x100` spin-pattern snapshots.

## Approach

- Use Python because the task emphasizes Monte Carlo experiments, plots, tables, and document export.
- Implement exact incremental single-spin updates for the 1D Metropolis and demon algorithms.
- Implement the main 2D equilibrium scans with a vectorized red-black Metropolis sweep. This keeps the local spin-flip rule while making `L=30` and `L=100` runs practical.
- Use separate small sequential-update runs for the ordered-versus-random update and single-attempt-versus-sweep sampling comparisons.
- Store generated CSV, JSON, and figures under `result/`.
- Render the final report from `docs/answer/answer.md` to `answer.docx` and `answer.pdf`.

## Testing

- Run `make run` to regenerate numerical artifacts.
- Run `make render` to regenerate `answer.docx` and `answer.pdf`.
- Run `make docs` for the full path.
- Check the following expected behavior:
  - 1D Metropolis energy follows `-N tanh(1/T)`.
  - Acceptance ratio increases with temperature.
  - Demon runs conserve total energy and give lower temperature for more negative total energy.
  - 2D magnetization drops near `T_c≈2.269`.
  - Specific-heat estimates peak near the finite-size transition region.
  - Open boundaries and small `L` show stronger finite-size rounding.

## Risks

- Low-temperature and critical-region Metropolis simulations decorrelate slowly, so the reported values are finite-run estimates rather than exact thermodynamic-limit values.
- Signed magnetization in zero field can average to nearly zero in finite simulations; the report therefore distinguishes signed `m` from `|m|`.
- Vectorized red-black sweeps and random sequential sweeps have the same intended equilibrium distribution but different temporal correlations, so trajectory-level comparisons are interpreted cautiously.
- The PDF text extraction omits some formulas; page images were used to recover the missing `E_ini=-40,-60,-80` and open-boundary-condition statements.
