from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
SEED = 20260610
J = 1.0
KB = 1.0


def ensure_result_dir() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)


def to_builtin(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_builtin(row.get(key, "")) for key in fieldnames})


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=to_builtin)


def estimate_equilibration(values, window=40, hold=25, rel_tol=0.04, abs_tol=0.0) -> int:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window + hold + 2:
        return max(0, len(arr) - 1)
    tail = arr[int(0.75 * len(arr)) :]
    target = float(np.mean(tail))
    tail_std = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
    tol = max(abs(target) * rel_tol, 1.8 * tail_std, abs_tol)
    rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
    for idx in range(0, len(rolling) - hold):
        if np.all(np.abs(rolling[idx : idx + hold] - target) <= tol):
            return int(idx + window)
    return int(len(arr) - 1)


def estimate_to_target(values, target, window=30, hold=25, tol=1.0) -> int:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window + hold + 2:
        return max(0, len(arr) - 1)
    rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
    for idx in range(0, len(rolling) - hold):
        if np.all(np.abs(rolling[idx : idx + hold] - target) <= tol):
            return int(idx + window)
    return int(len(arr) - 1)


def exact_1d_energy(N: int, T: float) -> float:
    return -N * math.tanh(J / (KB * T))


def init_1d(N: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "up":
        return np.ones(N, dtype=np.int8)
    if mode == "down":
        return -np.ones(N, dtype=np.int8)
    if mode == "random":
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    raise ValueError(f"unknown 1D init mode: {mode}")


def energy_1d(spins: np.ndarray, h: float = 0.0) -> float:
    return float(-J * np.sum(spins * np.roll(spins, -1)) - h * np.sum(spins))


def run_metropolis_1d(
    N: int,
    T: float,
    sweeps: int,
    init_mode: str,
    rng: np.random.Generator,
    h: float = 0.0,
) -> dict:
    spins = init_1d(N, init_mode, rng)
    energy = energy_1d(spins, h=h)
    magnetization = int(np.sum(spins))
    rows = [
        {
            "sweep": 0,
            "energy": energy,
            "magnetization": magnetization,
            "acceptance": 0.0,
        }
    ]
    total_accepted = 0
    total_attempts = 0
    for sweep in range(1, sweeps + 1):
        accepted = 0
        for _ in range(N):
            i = int(rng.integers(N))
            spin = int(spins[i])
            nn = int(spins[(i - 1) % N] + spins[(i + 1) % N])
            delta_e = 2.0 * spin * (J * nn + h)
            if delta_e <= 0.0 or rng.random() < math.exp(-delta_e / (KB * T)):
                spins[i] = -spin
                energy += delta_e
                magnetization -= 2 * spin
                accepted += 1
        total_accepted += accepted
        total_attempts += N
        rows.append(
            {
                "sweep": sweep,
                "energy": energy,
                "magnetization": magnetization,
                "acceptance": accepted / N,
            }
        )
    return {
        "rows": rows,
        "spins": spins.copy(),
        "mean_acceptance": total_accepted / total_attempts,
    }


def run_problem1(rng: np.random.Generator) -> dict:
    N = 20
    sweeps = 2500
    traj_rows: list[dict] = []
    summary: dict[str, dict] = {}
    exact_t1 = exact_1d_energy(N, 1.0)
    for mode in ["up", "random"]:
        run = run_metropolis_1d(N, 1.0, sweeps, mode, rng)
        energies = np.array([row["energy"] for row in run["rows"]], dtype=float)
        mags = np.array([row["magnetization"] for row in run["rows"]], dtype=float)
        eq_step = estimate_to_target(energies, exact_t1, tol=1.2)
        summary[mode] = {
            "equilibration_sweeps": eq_step,
            "tail_energy_mean": float(np.mean(energies[-500:])),
            "tail_abs_m_mean": float(np.mean(np.abs(mags[-500:]))),
            "mean_acceptance": run["mean_acceptance"],
        }
        for row in run["rows"]:
            traj_rows.append({"init": mode, **row})

    scan_rows: list[dict] = []
    temps = np.round(np.arange(0.5, 5.0 + 0.001, 0.5), 3)
    for mode in ["up", "random"]:
        for T in temps:
            run = run_metropolis_1d(N, float(T), 1300, mode, rng)
            rows = run["rows"][101:]
            energies = np.array([row["energy"] for row in rows], dtype=float)
            mags = np.array([row["magnetization"] for row in rows], dtype=float)
            acceptances = np.array([row["acceptance"] for row in rows], dtype=float)
            scan_rows.append(
                {
                    "init": mode,
                    "T": float(T),
                    "mean_E": float(np.mean(energies)),
                    "mean_M": float(np.mean(mags)),
                    "mean_abs_M": float(np.mean(np.abs(mags))),
                    "acceptance": float(np.mean(acceptances)),
                    "exact_E": exact_1d_energy(N, float(T)),
                }
            )

    write_csv(
        RESULT / "problem1_equilibration.csv",
        traj_rows,
        ["init", "sweep", "energy", "magnetization", "acceptance"],
    )
    write_csv(
        RESULT / "problem1_temperature_scan.csv",
        scan_rows,
        ["init", "T", "mean_E", "mean_M", "mean_abs_M", "acceptance", "exact_E"],
    )

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for mode, color in [("up", "tab:blue"), ("random", "tab:orange")]:
        rows = [row for row in traj_rows if row["init"] == mode]
        x = [row["sweep"] for row in rows]
        y = [row["energy"] for row in rows]
        ax.plot(x, y, lw=1.0, alpha=0.82, color=color, label=f"init={mode}")
        ax.axvline(summary[mode]["equilibration_sweeps"], color=color, ls="--", lw=1.0)
    ax.axhline(exact_t1, color="black", ls=":", lw=1.4, label="exact mean")
    ax.set_xlabel("MC sweeps")
    ax.set_ylabel("Energy")
    ax.set_title("1D Ising Metropolis at T=1, N=20")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem1_equilibration.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for mode, color in [("up", "tab:blue"), ("random", "tab:orange")]:
        rows = [row for row in scan_rows if row["init"] == mode]
        axes[0].plot(
            [row["T"] for row in rows],
            [row["mean_E"] for row in rows],
            marker="o",
            color=color,
            label=f"init={mode}",
        )
        axes[1].plot(
            [row["T"] for row in rows],
            [row["mean_abs_M"] for row in rows],
            marker="o",
            color=color,
            label=f"init={mode}",
        )
    exact_rows = [row for row in scan_rows if row["init"] == "up"]
    axes[0].plot(
        [row["T"] for row in exact_rows],
        [row["exact_E"] for row in exact_rows],
        color="black",
        ls="--",
        label=r"$-N\tanh(1/T)$",
    )
    axes[0].set_xlabel("T")
    axes[0].set_ylabel(r"$\langle E\rangle$")
    axes[1].set_xlabel("T")
    axes[1].set_ylabel(r"$\langle |M|\rangle$")
    axes[0].set_title("Mean energy")
    axes[1].set_title("Absolute magnetization")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem1_temperature_scan.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for mode, color in [("up", "tab:blue"), ("random", "tab:orange")]:
        rows = [row for row in scan_rows if row["init"] == mode]
        ax.plot(
            [row["T"] for row in rows],
            [row["acceptance"] for row in rows],
            marker="o",
            color=color,
            label=f"init={mode}",
        )
    ax.set_xlabel("T")
    ax.set_ylabel("Acceptance ratio")
    ax.set_title("1D Metropolis acceptance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem1_acceptance.png", dpi=180)
    plt.close(fig)

    summary["exact_energy_T1"] = exact_t1
    return {"summary": summary, "scan_rows": scan_rows}


def demon_temperature(mean_demon_energy: float) -> float:
    return 4.0 * J / math.log(1.0 + 4.0 * J / mean_demon_energy)


def run_demon_1d(
    N: int,
    total_energy: float,
    sweeps: int,
    rng: np.random.Generator,
) -> dict:
    spins = np.ones(N, dtype=np.int8)
    system_energy = energy_1d(spins)
    demon_energy = float(total_energy - system_energy)
    if demon_energy < 0:
        raise ValueError("initial demon energy is negative")
    magnetization = int(np.sum(spins))
    rows = [
        {
            "sweep": 0,
            "system_E": system_energy,
            "demon_E": demon_energy,
            "magnetization": magnetization,
            "M2": magnetization * magnetization,
            "acceptance": 0.0,
        }
    ]
    total_accepted = 0
    total_attempts = 0
    for sweep in range(1, sweeps + 1):
        accepted = 0
        for _ in range(N):
            i = int(rng.integers(N))
            spin = int(spins[i])
            nn = int(spins[(i - 1) % N] + spins[(i + 1) % N])
            delta_e = 2.0 * J * spin * nn
            if delta_e <= demon_energy:
                spins[i] = -spin
                system_energy += delta_e
                demon_energy -= delta_e
                magnetization -= 2 * spin
                accepted += 1
        total_accepted += accepted
        total_attempts += N
        rows.append(
            {
                "sweep": sweep,
                "system_E": system_energy,
                "demon_E": demon_energy,
                "magnetization": magnetization,
                "M2": magnetization * magnetization,
                "acceptance": accepted / N,
            }
        )
    return {
        "rows": rows,
        "spins": spins.copy(),
        "mean_acceptance": total_accepted / total_attempts,
    }


def run_problem2(rng: np.random.Generator) -> dict:
    rows_main: list[dict] = []
    main = run_demon_1d(100, -20.0, 3500, rng)
    for row in main["rows"]:
        rows_main.append(row)
    write_csv(
        RESULT / "problem2_demon_Eini_minus20.csv",
        rows_main,
        ["sweep", "system_E", "demon_E", "magnetization", "M2", "acceptance"],
    )

    demon_values = np.array([row["demon_E"] for row in rows_main], dtype=float)
    eq_step = estimate_equilibration(demon_values, window=60, hold=30, abs_tol=3.0)
    post = rows_main[max(eq_step, 1000) :]
    main_summary = {
        "E_total": -20.0,
        "equilibration_sweeps": eq_step,
        "mean_demon_E": float(np.mean([row["demon_E"] for row in post])),
        "mean_system_E": float(np.mean([row["system_E"] for row in post])),
        "mean_M": float(np.mean([row["magnetization"] for row in post])),
        "mean_M2": float(np.mean([row["M2"] for row in post])),
        "acceptance": float(np.mean([row["acceptance"] for row in post])),
    }
    main_summary["T_from_demon"] = demon_temperature(main_summary["mean_demon_E"])
    main_summary["exact_E_per_spin_at_T"] = -math.tanh(1.0 / main_summary["T_from_demon"])

    case_rows: list[dict] = []
    for total_energy in [-40.0, -60.0, -80.0]:
        run = run_demon_1d(100, total_energy, 5000, rng)
        post = run["rows"][1000:]
        mean_demon = float(np.mean([row["demon_E"] for row in post]))
        temp = demon_temperature(mean_demon)
        mean_system = float(np.mean([row["system_E"] for row in post]))
        case_rows.append(
            {
                "N": 100,
                "sweeps": 5000,
                "E_total": total_energy,
                "mean_demon_E": mean_demon,
                "T": temp,
                "mean_system_E": mean_system,
                "mean_E_per_spin": mean_system / 100.0,
                "exact_E_per_spin": -math.tanh(1.0 / temp),
                "mean_M": float(np.mean([row["magnetization"] for row in post])),
                "mean_M2": float(np.mean([row["M2"] for row in post])),
                "acceptance": float(np.mean([row["acceptance"] for row in post])),
            }
        )

    dependency_rows: list[dict] = []
    for N, total_energy in [(64, -40.0), (100, -60.0), (200, -120.0)]:
        for sweeps in [1000, 5000]:
            run = run_demon_1d(N, total_energy, sweeps, rng)
            warmup = min(800, sweeps // 2)
            post = run["rows"][warmup:]
            mean_demon = float(np.mean([row["demon_E"] for row in post]))
            temp = demon_temperature(mean_demon)
            mean_system = float(np.mean([row["system_E"] for row in post]))
            dependency_rows.append(
                {
                    "N": N,
                    "sweeps": sweeps,
                    "E_total": total_energy,
                    "E_total_per_spin": total_energy / N,
                    "T": temp,
                    "mean_E_per_spin": mean_system / N,
                    "exact_E_per_spin": -math.tanh(1.0 / temp),
                    "mean_demon_E": mean_demon,
                    "mean_M2_per_N2": float(np.mean([row["M2"] for row in post])) / (N * N),
                }
            )

    write_csv(
        RESULT / "problem2_demon_cases.csv",
        case_rows,
        [
            "N",
            "sweeps",
            "E_total",
            "mean_demon_E",
            "T",
            "mean_system_E",
            "mean_E_per_spin",
            "exact_E_per_spin",
            "mean_M",
            "mean_M2",
            "acceptance",
        ],
    )
    write_csv(
        RESULT / "problem2_demon_size_steps.csv",
        dependency_rows,
        [
            "N",
            "sweeps",
            "E_total",
            "E_total_per_spin",
            "T",
            "mean_E_per_spin",
            "exact_E_per_spin",
            "mean_demon_E",
            "mean_M2_per_N2",
        ],
    )

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True)
    sweeps = [row["sweep"] for row in rows_main]
    for ax, key, ylabel in [
        (axes[0], "demon_E", r"$E_d$"),
        (axes[1], "system_E", r"$E$"),
        (axes[2], "M2", r"$M^2$"),
    ]:
        values = np.array([row[key] for row in rows_main], dtype=float)
        running = np.cumsum(values) / (np.arange(len(values)) + 1)
        ax.plot(sweeps, running, lw=1.1)
        ax.axvline(eq_step, color="tab:red", ls="--", lw=1.0)
        ax.set_ylabel(ylabel)
    axes[2].set_xlabel("MC sweeps per spin")
    axes[0].set_title("1D demon running averages, E_total=-20")
    fig.tight_layout()
    fig.savefig(RESULT / "problem2_demon_running.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(
        [row["T"] for row in case_rows],
        [row["mean_E_per_spin"] for row in case_rows],
        marker="o",
        label="demon",
    )
    axes[0].plot(
        [row["T"] for row in case_rows],
        [row["exact_E_per_spin"] for row in case_rows],
        marker="s",
        ls="--",
        label="exact infinite chain",
    )
    axes[1].plot(
        [row["T"] for row in case_rows],
        [row["mean_M2"] / (100 * 100) for row in case_rows],
        marker="o",
        color="tab:purple",
    )
    axes[0].set_xlabel("T from demon")
    axes[0].set_ylabel(r"$E/N$")
    axes[1].set_xlabel("T from demon")
    axes[1].set_ylabel(r"$\langle M^2\rangle/N^2$")
    axes[0].legend()
    axes[0].set_title("Energy comparison")
    axes[1].set_title("Magnetization fluctuation")
    fig.tight_layout()
    fig.savefig(RESULT / "problem2_demon_cases.png", dpi=180)
    plt.close(fig)

    return {
        "main_summary": main_summary,
        "case_rows": case_rows,
        "dependency_rows": dependency_rows,
    }


def init_2d(L: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "up":
        return np.ones((L, L), dtype=np.int8)
    if mode == "down":
        return -np.ones((L, L), dtype=np.int8)
    if mode == "random":
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    if mode == "checkerboard":
        i, j = np.indices((L, L))
        return np.where((i + j) % 2 == 0, 1, -1).astype(np.int8)
    if mode == "half":
        spins = -np.ones((L, L), dtype=np.int8)
        spins[:, : L // 2] = 1
        return spins
    raise ValueError(f"unknown 2D init mode: {mode}")


def neighbor_sum_2d(spins: np.ndarray, pbc: bool) -> np.ndarray:
    if pbc:
        return (
            np.roll(spins, 1, axis=0)
            + np.roll(spins, -1, axis=0)
            + np.roll(spins, 1, axis=1)
            + np.roll(spins, -1, axis=1)
        )
    nn = np.zeros_like(spins, dtype=np.int16)
    nn[1:, :] += spins[:-1, :]
    nn[:-1, :] += spins[1:, :]
    nn[:, 1:] += spins[:, :-1]
    nn[:, :-1] += spins[:, 1:]
    return nn


def energy_2d(spins: np.ndarray, pbc: bool = True) -> float:
    if pbc:
        bonds = np.sum(spins * np.roll(spins, -1, axis=0)) + np.sum(
            spins * np.roll(spins, -1, axis=1)
        )
    else:
        bonds = np.sum(spins[:-1, :] * spins[1:, :]) + np.sum(
            spins[:, :-1] * spins[:, 1:]
        )
    return float(-J * bonds)


def sweep_checkerboard(
    spins: np.ndarray,
    T: float,
    rng: np.random.Generator,
    pbc: bool = True,
) -> tuple[int, int]:
    L = spins.shape[0]
    i, j = np.indices(spins.shape)
    accepted = 0
    attempts = 0
    for parity in (0, 1):
        mask = (i + j) % 2 == parity
        nn = neighbor_sum_2d(spins, pbc=pbc)
        delta_e = 2.0 * J * spins * nn
        randoms = rng.random(spins.shape)
        accept = mask & ((delta_e <= 0.0) | (randoms < np.exp(-delta_e / T)))
        count = int(np.count_nonzero(accept))
        spins[accept] *= -1
        accepted += count
        attempts += L * L // 2 + (1 if (L * L) % 2 and parity == 0 else 0)
    return accepted, attempts


def run_2d_checkerboard(
    L: int,
    T: float,
    total_sweeps: int,
    equil_sweeps: int,
    init_mode: str,
    rng: np.random.Generator,
    pbc: bool = True,
    initial_spins: np.ndarray | None = None,
) -> dict:
    spins = initial_spins.copy() if initial_spins is not None else init_2d(L, init_mode, rng)
    records: list[dict] = []
    acceptance_values: list[float] = []
    for sweep in range(total_sweeps + 1):
        e = energy_2d(spins, pbc=pbc)
        m = int(np.sum(spins))
        records.append(
            {
                "sweep": sweep,
                "T": T,
                "E": e,
                "E_per_spin": e / (L * L),
                "M": m,
                "m": m / (L * L),
                "abs_m": abs(m) / (L * L),
                "m2": (m / (L * L)) ** 2,
                "acceptance": acceptance_values[-1] if acceptance_values else 0.0,
            }
        )
        if sweep == total_sweeps:
            break
        accepted, attempts = sweep_checkerboard(spins, T, rng, pbc=pbc)
        acceptance_values.append(accepted / attempts)

    post = [row for row in records if row["sweep"] > equil_sweeps]
    energies = np.array([row["E"] for row in post], dtype=float)
    e_per_spin = np.array([row["E_per_spin"] for row in post], dtype=float)
    mags = np.array([row["m"] for row in post], dtype=float)
    abs_mags = np.array([row["abs_m"] for row in post], dtype=float)
    m2 = np.array([row["m2"] for row in post], dtype=float)
    stats = {
        "L": L,
        "T": T,
        "pbc": pbc,
        "equil_sweeps": equil_sweeps,
        "measure_sweeps": len(post),
        "mean_E": float(np.mean(energies)),
        "mean_E_per_spin": float(np.mean(e_per_spin)),
        "mean_m": float(np.mean(mags)),
        "mean_abs_m": float(np.mean(abs_mags)),
        "mean_m2": float(np.mean(m2)),
        "specific_heat_fluct": float(np.var(energies, ddof=1) / (L * L * T * T))
        if len(energies) > 1
        else 0.0,
        "acceptance": float(np.mean([row["acceptance"] for row in post])),
    }
    return {"records": records, "stats": stats, "spins": spins.copy()}


def single_site_delta_2d(spins: np.ndarray, i: int, j: int, pbc: bool = True) -> float:
    L = spins.shape[0]
    if pbc:
        nn = (
            spins[(i - 1) % L, j]
            + spins[(i + 1) % L, j]
            + spins[i, (j - 1) % L]
            + spins[i, (j + 1) % L]
        )
    else:
        nn = 0
        if i > 0:
            nn += spins[i - 1, j]
        if i < L - 1:
            nn += spins[i + 1, j]
        if j > 0:
            nn += spins[i, j - 1]
        if j < L - 1:
            nn += spins[i, j + 1]
    return float(2.0 * J * spins[i, j] * nn)


def run_2d_sequential(
    L: int,
    T: float,
    sweeps: int,
    init_mode: str,
    rng: np.random.Generator,
    order: str = "random",
    pbc: bool = True,
    equil_sweeps: int = 0,
) -> dict:
    spins = init_2d(L, init_mode, rng)
    records: list[dict] = []
    accepted_total = 0
    attempts_total = 0
    sites_ordered = [(i, j) for i in range(L) for j in range(L)]
    for sweep in range(sweeps + 1):
        e = energy_2d(spins, pbc=pbc)
        m = int(np.sum(spins))
        records.append(
            {
                "sweep": sweep,
                "E_per_spin": e / (L * L),
                "m": m / (L * L),
                "abs_m": abs(m) / (L * L),
            }
        )
        if sweep == sweeps:
            break
        if order == "ordered":
            sites = sites_ordered
        elif order == "random":
            idx_i = rng.integers(0, L, size=L * L)
            idx_j = rng.integers(0, L, size=L * L)
            sites = zip(idx_i, idx_j)
        else:
            raise ValueError(f"unknown update order: {order}")
        for i, j in sites:
            i = int(i)
            j = int(j)
            delta_e = single_site_delta_2d(spins, i, j, pbc=pbc)
            if delta_e <= 0.0 or rng.random() < math.exp(-delta_e / T):
                spins[i, j] *= -1
                accepted_total += 1
            attempts_total += 1
    post = records[equil_sweeps + 1 :]
    stats = {
        "order": order,
        "mean_E_per_spin": float(np.mean([row["E_per_spin"] for row in post])),
        "mean_abs_m": float(np.mean([row["abs_m"] for row in post])),
        "acceptance": accepted_total / attempts_total,
    }
    return {"records": records, "stats": stats, "spins": spins.copy()}


def run_single_attempt_sampling(
    L: int,
    T: float,
    rng: np.random.Generator,
    equil_sweeps: int = 250,
    measure_sweeps: int = 300,
) -> dict:
    spins = init_2d(L, "random", rng)
    for _ in range(equil_sweeps):
        for _ in range(L * L):
            i = int(rng.integers(L))
            j = int(rng.integers(L))
            delta_e = single_site_delta_2d(spins, i, j, pbc=True)
            if delta_e <= 0.0 or rng.random() < math.exp(-delta_e / T):
                spins[i, j] *= -1

    attempt_rows: list[dict] = []
    sweep_rows: list[dict] = []
    for sweep in range(1, measure_sweeps + 1):
        for attempt in range(1, L * L + 1):
            i = int(rng.integers(L))
            j = int(rng.integers(L))
            delta_e = single_site_delta_2d(spins, i, j, pbc=True)
            if delta_e <= 0.0 or rng.random() < math.exp(-delta_e / T):
                spins[i, j] *= -1
            if attempt % max(1, (L * L) // 20) == 0:
                e = energy_2d(spins, pbc=True) / (L * L)
                m = abs(np.sum(spins)) / (L * L)
                attempt_rows.append(
                    {
                        "time_sweeps": sweep - 1 + attempt / (L * L),
                        "E_per_spin": e,
                        "abs_m": m,
                    }
                )
        e = energy_2d(spins, pbc=True) / (L * L)
        m = abs(np.sum(spins)) / (L * L)
        sweep_rows.append({"sweep": sweep, "E_per_spin": e, "abs_m": m})
    return {
        "attempt_rows": attempt_rows,
        "sweep_rows": sweep_rows,
        "summary": {
            "attempt_mean_E_per_spin": float(np.mean([row["E_per_spin"] for row in attempt_rows])),
            "sweep_mean_E_per_spin": float(np.mean([row["E_per_spin"] for row in sweep_rows])),
            "attempt_mean_abs_m": float(np.mean([row["abs_m"] for row in attempt_rows])),
            "sweep_mean_abs_m": float(np.mean([row["abs_m"] for row in sweep_rows])),
        },
    }


def exact_2d_spontaneous_magnetization(T: float) -> float:
    tc = 2.0 / math.log(1.0 + math.sqrt(2.0))
    if T >= tc:
        return 0.0
    value = 1.0 - math.sinh(2.0 / T) ** -4
    return value ** 0.125


def add_specific_heat_derivative(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, bool], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["L"]), bool(row["pbc"])), []).append(row)
    output = []
    for _, group in grouped.items():
        group = sorted(group, key=lambda row: row["T"])
        temps = np.array([row["T"] for row in group], dtype=float)
        energy = np.array([row["mean_E_per_spin"] for row in group], dtype=float)
        deriv = np.gradient(energy, temps)
        for row, c_deriv in zip(group, deriv):
            copied = dict(row)
            copied["specific_heat_derivative"] = float(c_deriv)
            output.append(copied)
    return sorted(output, key=lambda row: (row["L"], row["pbc"], row["T"]))


def plot_spin(ax, spins: np.ndarray, title: str) -> None:
    cmap = ListedColormap(["#2a9d8f", "#f8f4ef"])
    ax.imshow(spins, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def run_problem3_and_4(rng: np.random.Generator) -> dict:
    trajectory_rows: list[dict] = []
    trajectory_summary: list[dict] = []
    final_spins: dict[float, np.ndarray] = {}
    for T, sweeps in [(2.0, 900), (4.0, 700)]:
        run = run_2d_checkerboard(30, T, sweeps, 0, "random", rng, pbc=True)
        final_spins[T] = run["spins"]
        e_series = [row["E_per_spin"] for row in run["records"]]
        eq = estimate_equilibration(e_series, window=35, hold=25, abs_tol=0.03)
        for row in run["records"]:
            trajectory_rows.append({"case": f"L30_T{T:g}", **row})
        trajectory_summary.append(
            {
                "case": f"L30_T{T:g}",
                "T": T,
                "estimated_equil_sweeps": eq,
                "final_E_per_spin": run["records"][-1]["E_per_spin"],
                "final_abs_m": run["records"][-1]["abs_m"],
                "tail_E_per_spin": float(np.mean([r["E_per_spin"] for r in run["records"][-120:]])),
                "tail_abs_m": float(np.mean([r["abs_m"] for r in run["records"][-120:]])),
            }
        )

    write_csv(
        RESULT / "problem3_trajectories.csv",
        trajectory_rows,
        ["case", "sweep", "T", "E", "E_per_spin", "M", "m", "abs_m", "m2", "acceptance"],
    )
    write_csv(
        RESULT / "problem3_trajectory_summary.csv",
        trajectory_summary,
        [
            "case",
            "T",
            "estimated_equil_sweeps",
            "final_E_per_spin",
            "final_abs_m",
            "tail_E_per_spin",
            "tail_abs_m",
        ],
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8), sharex="col")
    for col, T in enumerate([2.0, 4.0]):
        rows = [row for row in trajectory_rows if row["case"] == f"L30_T{T:g}"]
        x = [row["sweep"] for row in rows]
        axes[0, col].plot(x, [row["E_per_spin"] for row in rows], lw=1.0)
        axes[1, col].plot(x, [row["abs_m"] for row in rows], lw=1.0, color="tab:orange")
        axes[0, col].set_title(f"L=30, T={T:g}")
        axes[1, col].set_xlabel("MC sweeps")
        axes[0, col].set_ylabel("E/N")
        axes[1, col].set_ylabel("|M|/N")
    fig.tight_layout()
    fig.savefig(RESULT / "problem3_T2_T4_trajectories.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8))
    plot_spin(axes[0], final_spins[2.0], "L=30, T=2 final")
    plot_spin(axes[1], final_spins[4.0], "L=30, T=4 final")
    fig.tight_layout()
    fig.savefig(RESULT / "problem3_T2_T4_snapshots.png", dpi=180)
    plt.close(fig)

    init_rows: list[dict] = []
    for mode in ["random", "up", "down", "checkerboard", "half"]:
        run = run_2d_checkerboard(30, 2.0, 900, 0, mode, rng, pbc=True)
        e_series = [row["E_per_spin"] for row in run["records"]]
        eq = estimate_equilibration(e_series, window=35, hold=25, abs_tol=0.03)
        init_rows.append(
            {
                "init": mode,
                "estimated_equil_sweeps": eq,
                "tail_E_per_spin": float(np.mean([r["E_per_spin"] for r in run["records"][-150:]])),
                "tail_abs_m": float(np.mean([r["abs_m"] for r in run["records"][-150:]])),
                "final_abs_m": run["records"][-1]["abs_m"],
            }
        )
    write_csv(
        RESULT / "problem3_initial_conditions.csv",
        init_rows,
        ["init", "estimated_equil_sweeps", "tail_E_per_spin", "tail_abs_m", "final_abs_m"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    labels = [row["init"] for row in init_rows]
    axes[0].bar(labels, [row["estimated_equil_sweeps"] for row in init_rows], color="tab:blue")
    axes[1].bar(labels, [row["tail_abs_m"] for row in init_rows], color="tab:orange")
    axes[0].set_ylabel("estimated equil sweeps")
    axes[1].set_ylabel("tail |M|/N")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].tick_params(axis="x", rotation=25)
    axes[0].set_title("Equilibration time")
    axes[1].set_title("Final orderedness")
    fig.tight_layout()
    fig.savefig(RESULT / "problem3_initial_conditions.png", dpi=180)
    plt.close(fig)

    scan_rows: list[dict] = []
    temps_p3 = np.round(np.arange(1.0, 4.0 + 0.001, 0.5), 3)
    for L, equil, measure in [(30, 1400, 1800), (4, 500, 1800)]:
        for T in temps_p3:
            run = run_2d_checkerboard(L, float(T), equil + measure, equil, "random", rng, pbc=True)
            scan_rows.append(run["stats"])
    scan_rows = add_specific_heat_derivative(scan_rows)
    write_csv(
        RESULT / "problem3_temperature_scan.csv",
        scan_rows,
        [
            "L",
            "T",
            "pbc",
            "equil_sweeps",
            "measure_sweeps",
            "mean_E",
            "mean_E_per_spin",
            "mean_m",
            "mean_abs_m",
            "mean_m2",
            "specific_heat_fluct",
            "acceptance",
            "specific_heat_derivative",
        ],
    )

    for L, filename in [(30, "problem3_L30_temperature_scan.png"), (4, "problem3_L4_temperature_scan.png")]:
        rows = [row for row in scan_rows if row["L"] == L]
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
        axes[0].plot([r["T"] for r in rows], [r["mean_E_per_spin"] for r in rows], marker="o")
        axes[1].plot([r["T"] for r in rows], [r["mean_abs_m"] for r in rows], marker="o", color="tab:orange")
        axes[2].plot(
            [r["T"] for r in rows],
            [r["specific_heat_fluct"] for r in rows],
            marker="o",
            label="fluct.",
        )
        axes[2].plot(
            [r["T"] for r in rows],
            [r["specific_heat_derivative"] for r in rows],
            marker="s",
            ls="--",
            label="dE/dT",
        )
        axes[0].set_ylabel("E/N")
        axes[1].set_ylabel("|M|/N")
        axes[2].set_ylabel("C/N")
        for ax in axes:
            ax.set_xlabel("T")
            ax.axvline(2.269, color="black", ls=":", lw=1.0)
        axes[2].legend()
        fig.suptitle(f"PBC temperature scan, L={L}")
        fig.tight_layout()
        fig.savefig(RESULT / filename, dpi=180)
        plt.close(fig)

    obc_rows: list[dict] = []
    for L, equil, measure in [(30, 1200, 1400), (4, 500, 1400)]:
        for T in [2.0, 2.5, 4.0]:
            for pbc in [True, False]:
                run = run_2d_checkerboard(L, T, equil + measure, equil, "random", rng, pbc=pbc)
                row = dict(run["stats"])
                row["boundary"] = "PBC" if pbc else "OBC"
                obc_rows.append(row)
    write_csv(
        RESULT / "problem3_obc_comparison.csv",
        obc_rows,
        [
            "L",
            "T",
            "boundary",
            "pbc",
            "mean_E_per_spin",
            "mean_abs_m",
            "specific_heat_fluct",
            "acceptance",
            "equil_sweeps",
            "measure_sweeps",
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for L, marker in [(30, "o"), (4, "s")]:
        for boundary, ls in [("PBC", "-"), ("OBC", "--")]:
            rows = [r for r in obc_rows if r["L"] == L and r["boundary"] == boundary]
            axes[0].plot(
                [r["T"] for r in rows],
                [r["mean_E_per_spin"] for r in rows],
                marker=marker,
                ls=ls,
                label=f"L={L} {boundary}",
            )
            axes[1].plot(
                [r["T"] for r in rows],
                [r["mean_abs_m"] for r in rows],
                marker=marker,
                ls=ls,
                label=f"L={L} {boundary}",
            )
    axes[0].set_ylabel("E/N")
    axes[1].set_ylabel("|M|/N")
    for ax in axes:
        ax.set_xlabel("T")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT / "problem3_obc_comparison.png", dpi=180)
    plt.close(fig)

    update_rows: list[dict] = []
    update_traj_rows: list[dict] = []
    for order in ["random", "ordered"]:
        run = run_2d_sequential(30, 2.3, 650, "random", rng, order=order, pbc=True, equil_sweeps=250)
        update_rows.append({"order": order, **run["stats"]})
        for row in run["records"]:
            update_traj_rows.append({"order": order, **row})
    write_csv(
        RESULT / "problem3_update_order.csv",
        update_rows,
        ["order", "mean_E_per_spin", "mean_abs_m", "acceptance"],
    )
    write_csv(
        RESULT / "problem3_update_order_trajectories.csv",
        update_traj_rows,
        ["order", "sweep", "E_per_spin", "m", "abs_m"],
    )

    sampling = run_single_attempt_sampling(30, 2.3, rng, equil_sweeps=250, measure_sweeps=260)
    write_csv(
        RESULT / "problem3_sampling_attempts.csv",
        sampling["attempt_rows"],
        ["time_sweeps", "E_per_spin", "abs_m"],
    )
    write_csv(
        RESULT / "problem3_sampling_sweeps.csv",
        sampling["sweep_rows"],
        ["sweep", "E_per_spin", "abs_m"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for order, color in [("random", "tab:blue"), ("ordered", "tab:orange")]:
        rows = [row for row in update_traj_rows if row["order"] == order]
        axes[0].plot([r["sweep"] for r in rows], [r["E_per_spin"] for r in rows], lw=1.0, color=color, label=order)
    axes[1].plot(
        [r["time_sweeps"] for r in sampling["attempt_rows"]],
        [r["E_per_spin"] for r in sampling["attempt_rows"]],
        lw=0.7,
        alpha=0.65,
        label="single-attempt samples",
    )
    axes[1].plot(
        [r["sweep"] for r in sampling["sweep_rows"]],
        [r["E_per_spin"] for r in sampling["sweep_rows"]],
        lw=1.1,
        label="end-of-sweep samples",
    )
    axes[0].set_title("Random vs ordered spin choice")
    axes[1].set_title("Sampling cadence")
    for ax in axes:
        ax.set_xlabel("MC sweeps")
        ax.set_ylabel("E/N")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem3_update_sampling_comparison.png", dpi=180)
    plt.close(fig)

    finite_rows: list[dict] = []
    temps_fss = [1.0, 1.25, 1.5, 1.75, 2.0, 2.1, 2.2, 2.269, 2.35, 2.5, 2.75, 3.0, 3.5, 4.0]
    for L in [4, 8, 16, 32]:
        equil = 650 if L < 32 else 900
        measure = 950 if L < 32 else 1200
        for T in temps_fss:
            run = run_2d_checkerboard(L, T, equil + measure, equil, "random", rng, pbc=True)
            row = dict(run["stats"])
            row["exact_m_infinite"] = exact_2d_spontaneous_magnetization(T)
            finite_rows.append(row)
    write_csv(
        RESULT / "problem4_finite_size_scan.csv",
        finite_rows,
        [
            "L",
            "T",
            "pbc",
            "equil_sweeps",
            "measure_sweeps",
            "mean_E_per_spin",
            "mean_m",
            "mean_abs_m",
            "mean_m2",
            "specific_heat_fluct",
            "acceptance",
            "exact_m_infinite",
        ],
    )

    fig, ax = plt.subplots(figsize=(8, 5.0))
    for L, marker in [(4, "o"), (8, "s"), (16, "^"), (32, "D")]:
        rows = [r for r in finite_rows if r["L"] == L]
        ax.errorbar(
            [r["T"] for r in rows],
            [r["mean_abs_m"] for r in rows],
            marker=marker,
            capsize=2,
            lw=1.0,
            label=f"L={L}",
        )
    exact_t = np.linspace(1.0, 4.0, 220)
    ax.plot(
        exact_t,
        [exact_2d_spontaneous_magnetization(float(T)) for T in exact_t],
        color="black",
        ls="--",
        label="exact L=∞",
    )
    ax.axvline(2.269, color="black", ls=":", lw=1.0)
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle |m| \rangle$")
    ax.set_title("2D Ising magnetization vs temperature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem4_magnetization_vs_T.png", dpi=180)
    plt.close(fig)

    fit_rows: list[dict] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for T, color in [(2.0, "tab:green"), (2.269, "tab:blue"), (2.5, "tab:red")]:
        rows = sorted([r for r in finite_rows if abs(r["T"] - T) < 1e-9], key=lambda r: r["L"])
        x = np.log([r["L"] for r in rows])
        y = np.log([r["mean_m2"] for r in rows])
        slope, intercept = np.polyfit(x, y, 1)
        fit_rows.append({"T": T, "slope": float(slope), "intercept": float(intercept)})
        ax.plot(x, y, marker="o", color=color, label=f"T={T:g}, slope={slope:.3f}")
        ax.plot(x, slope * x + intercept, ls="--", color=color, alpha=0.75)
    ax.set_xlabel(r"$\ln L$")
    ax.set_ylabel(r"$\ln \langle m^2\rangle$")
    ax.set_title("Finite-size scaling of magnetization squared")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT / "problem4_fss_logfit.png", dpi=180)
    plt.close(fig)
    write_csv(RESULT / "problem4_critical_exponent_fit.csv", fit_rows, ["T", "slope", "intercept"])

    snapshot_sweeps = [0, 100, 200, 300]
    snapshots = []
    for T in [2.269, 2.0 * 2.269]:
        spins = init_2d(100, "random", rng)
        frame_map = {0: spins.copy()}
        for sweep in range(1, max(snapshot_sweeps) + 1):
            sweep_checkerboard(spins, T, rng, pbc=True)
            if sweep in snapshot_sweeps:
                frame_map[sweep] = spins.copy()
        snapshots.append((T, frame_map))
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.7))
    for row_idx, (T, frame_map) in enumerate(snapshots):
        for col_idx, sweep in enumerate(snapshot_sweeps):
            plot_spin(axes[row_idx, col_idx], frame_map[sweep], f"T={T:.3g}, t={sweep}")
    fig.tight_layout()
    fig.savefig(RESULT / "problem4_L100_snapshots.png", dpi=180)
    plt.close(fig)

    return {
        "trajectory_summary": trajectory_summary,
        "scan_rows": scan_rows,
        "obc_rows": obc_rows,
        "update_rows": update_rows,
        "sampling_summary": sampling["summary"],
        "finite_rows": finite_rows,
        "fit_rows": fit_rows,
    }


def main() -> None:
    ensure_result_dir()
    start = time.time()
    rng = np.random.default_rng(SEED)
    summary = {
        "seed": SEED,
        "problem1": run_problem1(rng),
        "problem2": run_problem2(rng),
        "problem3_4": run_problem3_and_4(rng),
    }
    summary["runtime_seconds"] = time.time() - start
    write_json(RESULT / "summary.json", summary)
    with (RESULT / "temp-01.log").open("w", encoding="utf-8") as f:
        f.write(f"HW/13 Ising simulations completed in {summary['runtime_seconds']:.2f} seconds\n")
        f.write(f"Seed: {SEED}\n")
        f.write("Generated CSV, JSON, and PNG artifacts under result/.\n")
    print(f"Generated HW/13 Ising artifacts in {summary['runtime_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
