from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRUE_PI = math.pi
SEED = 20260513


def save_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def format_float(value: float, digits: int = 10) -> str:
    return f"{value:.{digits}g}"


def buffon_factor(length_ratio: float) -> float:
    """Return c(x) such that P(crossing)=c(x)/pi for parallel-line Buffon needle."""
    x = float(length_ratio)
    if x <= 1.0:
        return 2.0 * x
    return 2.0 * (x - math.sqrt(x * x - 1.0) + math.acos(1.0 / x))


def buffon_probability(length_ratio: float) -> float:
    return buffon_factor(length_ratio) / TRUE_PI


def buffon_pi_std(length_ratio: float, n_samples: int) -> float:
    p = buffon_probability(length_ratio)
    return TRUE_PI * math.sqrt((1.0 - p) / (p * n_samples))


def simulate_buffon_hits(rng: np.random.Generator, n_samples: int, length_ratio: float) -> int:
    distance_to_nearest_line = 0.5 * rng.random(n_samples)
    theta = 0.5 * math.pi * rng.random(n_samples)
    reach = 0.5 * length_ratio * np.sin(theta)
    hits = distance_to_nearest_line <= np.minimum(0.5, reach)
    return int(np.count_nonzero(hits))


def estimate_pi_from_hits(factor: float, hits: int, n_samples: int) -> float:
    if hits == 0:
        return float("nan")
    return factor * n_samples / hits


def problem1(rng: np.random.Generator, result_dir: Path, log: io.StringIO) -> dict[str, object]:
    print("Problem 1: Buffon's needle simulation", file=log)
    n_fixed = 1_000_000
    x_values = np.round(np.linspace(0.1, 1.0, 10), 1)
    rows: list[list[object]] = []
    fixed_results: list[dict[str, float]] = []

    for x in x_values:
        hits = simulate_buffon_hits(rng, n_fixed, float(x))
        factor = buffon_factor(float(x))
        p_hat = hits / n_fixed
        pi_est = estimate_pi_from_hits(factor, hits, n_fixed)
        std = buffon_pi_std(float(x), n_fixed)
        result = {
            "x": float(x),
            "n": n_fixed,
            "hits": hits,
            "p_hat": p_hat,
            "pi_est": pi_est,
            "abs_error": abs(pi_est - TRUE_PI),
            "theory_std": std,
        }
        fixed_results.append(result)
        rows.append(
            [
                f"{x:.1f}",
                n_fixed,
                hits,
                f"{p_hat:.10f}",
                f"{pi_est:.10f}",
                f"{abs(pi_est - TRUE_PI):.10f}",
                f"{std:.10f}",
            ]
        )
        print(
            f"  x={x:.1f}, N={n_fixed}, hits={hits}, pi={pi_est:.10f}, abs_error={abs(pi_est - TRUE_PI):.10f}",
            file=log,
        )

    save_csv(
        result_dir / "problem1_buffon_x_fixed_n.csv",
        ["x", "N", "hits", "p_hat", "pi_est", "abs_error", "theory_std"],
        rows,
    )
    plot_problem1_x(result_dir / "problem1_buffon_x_fixed_n.png", fixed_results)

    n_series = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]
    max_n = max(n_series)
    theta = 0.5 * math.pi * rng.random(max_n)
    distance_to_nearest_line = 0.5 * rng.random(max_n)
    hits_cumulative = np.cumsum(distance_to_nearest_line <= 0.5 * np.sin(theta))
    rows = []
    n_results: list[dict[str, float]] = []
    for n in n_series:
        hits = int(hits_cumulative[n - 1])
        factor = buffon_factor(1.0)
        pi_est = estimate_pi_from_hits(factor, hits, n)
        p_hat = hits / n
        std = buffon_pi_std(1.0, n)
        result = {
            "n": n,
            "hits": hits,
            "p_hat": p_hat,
            "pi_est": pi_est,
            "abs_error": abs(pi_est - TRUE_PI),
            "theory_std": std,
        }
        n_results.append(result)
        rows.append([n, hits, f"{p_hat:.10f}", f"{pi_est:.10f}", f"{abs(pi_est - TRUE_PI):.10f}", f"{std:.10f}"])
        print(
            f"  x=1.0, N={n}, hits={hits}, pi={pi_est:.10f}, abs_error={abs(pi_est - TRUE_PI):.10f}",
            file=log,
        )

    save_csv(
        result_dir / "problem1_buffon_n_convergence.csv",
        ["N", "hits", "p_hat", "pi_est", "abs_error", "theory_std"],
        rows,
    )
    plot_problem1_n(result_dir / "problem1_buffon_n_convergence.png", n_results)
    del theta, distance_to_nearest_line, hits_cumulative

    long_x_values = list(range(1, 15))
    rows = []
    long_results: list[dict[str, float]] = []
    for x in long_x_values:
        hits = simulate_buffon_hits(rng, n_fixed, float(x))
        factor = buffon_factor(float(x))
        p_hat = hits / n_fixed
        pi_est = estimate_pi_from_hits(factor, hits, n_fixed)
        std = buffon_pi_std(float(x), n_fixed)
        result = {
            "x": float(x),
            "n": n_fixed,
            "hits": hits,
            "p_hat": p_hat,
            "pi_est": pi_est,
            "abs_error": abs(pi_est - TRUE_PI),
            "theory_std": std,
            "theory_probability": buffon_probability(float(x)),
        }
        long_results.append(result)
        rows.append(
            [
                x,
                n_fixed,
                hits,
                f"{p_hat:.10f}",
                f"{buffon_probability(float(x)):.10f}",
                f"{pi_est:.10f}",
                f"{abs(pi_est - TRUE_PI):.10f}",
                f"{std:.10f}",
            ]
        )
    save_csv(
        result_dir / "problem1_buffon_long_needles.csv",
        ["x", "N", "hits", "p_hat", "theory_probability", "pi_est", "abs_error", "theory_std"],
        rows,
    )
    plot_problem1_long_x(result_dir / "problem1_buffon_long_needles.png", long_results)

    best_fixed = min(fixed_results, key=lambda item: item["abs_error"])
    best_long = min(long_results, key=lambda item: item["abs_error"])
    print(
        f"  best fixed-N short-needle estimate: x={best_fixed['x']:.1f}, pi={best_fixed['pi_est']:.10f}",
        file=log,
    )
    print(
        f"  best optional long-needle estimate: x={best_long['x']:.0f}, pi={best_long['pi_est']:.10f}",
        file=log,
    )

    return {
        "fixed_n": fixed_results,
        "n_convergence": n_results,
        "long_needles": long_results,
        "n_fixed": n_fixed,
    }


def plot_problem1_x(path: Path, results: list[dict[str, float]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.array([item["x"] for item in results])
    pi_est = np.array([item["pi_est"] for item in results])
    std = np.array([item["theory_std"] for item in results])
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    ax.errorbar(x, pi_est, yerr=std, marker="o", capsize=3, color="#1f77b4", label="MC estimate")
    ax.axhline(TRUE_PI, color="#d62728", linewidth=1.8, label="true pi")
    ax.set_xlabel("needle length ratio x = b/a")
    ax.set_ylabel("estimated pi")
    ax.set_title("Problem 1(a): fixed N, varying needle length")
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_problem1_n(path: Path, results: list[dict[str, float]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    n = np.array([item["n"] for item in results], dtype=float)
    pi_est = np.array([item["pi_est"] for item in results])
    std = np.array([item["theory_std"] for item in results])
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    ax.errorbar(n, pi_est, yerr=std, marker="o", capsize=3, color="#2ca02c", label="x=1 MC estimate")
    ax.axhline(TRUE_PI, color="#d62728", linewidth=1.8, label="true pi")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("estimated pi")
    ax.set_title("Problem 1(b): Buffon estimate versus sample size")
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_problem1_long_x(path: Path, results: list[dict[str, float]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.array([item["x"] for item in results])
    pi_est = np.array([item["pi_est"] for item in results])
    std = np.array([item["theory_std"] for item in results])
    prob = np.array([item["theory_probability"] for item in results])
    fig, ax1 = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    ax1.errorbar(x, pi_est, yerr=std, marker="o", capsize=3, color="#1f77b4", label="estimated pi")
    ax1.axhline(TRUE_PI, color="#d62728", linewidth=1.8, label="true pi")
    ax1.set_xlabel("needle length ratio x")
    ax1.set_ylabel("estimated pi")
    ax2 = ax1.twinx()
    ax2.plot(x, prob, color="#9467bd", marker="s", alpha=0.72, label="crossing probability")
    ax2.set_ylabel("crossing probability")
    ax1.set_title("Problem 1(d): long-needle Buffon estimates")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def problem2(rng: np.random.Generator, result_dir: Path, log: io.StringIO) -> dict[str, object]:
    print("Problem 2: acceptance-rejection pi simulation", file=log)
    n_series = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]
    max_n = max(n_series)
    x = rng.random(max_n)
    y = rng.random(max_n)
    inside_cumulative = np.cumsum(x * x + y * y <= 1.0)
    rows = []
    results: list[dict[str, float]] = []
    p = TRUE_PI / 4.0
    for n in n_series:
        hits = int(inside_cumulative[n - 1])
        p_hat = hits / n
        pi_est = 4.0 * p_hat
        std = 4.0 * math.sqrt(p * (1.0 - p) / n)
        result = {
            "n": n,
            "inside": hits,
            "p_hat": p_hat,
            "pi_est": pi_est,
            "abs_error": abs(pi_est - TRUE_PI),
            "theory_std": std,
        }
        results.append(result)
        rows.append([n, hits, f"{p_hat:.10f}", f"{pi_est:.10f}", f"{abs(pi_est - TRUE_PI):.10f}", f"{std:.10f}"])
        print(
            f"  N={n}, inside={hits}, pi={pi_est:.10f}, abs_error={abs(pi_est - TRUE_PI):.10f}",
            file=log,
        )
    save_csv(
        result_dir / "problem2_acceptance_rejection.csv",
        ["N", "inside", "p_hat", "pi_est", "abs_error", "theory_std"],
        rows,
    )
    plot_problem2(result_dir / "problem2_acceptance_rejection.png", results)
    del x, y, inside_cumulative
    return {"n_convergence": results}


def plot_problem2(path: Path, results: list[dict[str, float]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    n = np.array([item["n"] for item in results], dtype=float)
    pi_est = np.array([item["pi_est"] for item in results])
    std = np.array([item["theory_std"] for item in results])
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    ax.errorbar(n, pi_est, yerr=std, marker="o", capsize=3, color="#ff7f0e", label="MC estimate")
    ax.axhline(TRUE_PI, color="#d62728", linewidth=1.8, label="true pi")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("estimated pi")
    ax.set_title("Problem 2: acceptance-rejection estimate versus sample size")
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def problem3(result_dir: Path, log: io.StringIO) -> dict[str, object]:
    print("Problem 3: university-transition Markov chain", file=log)
    states = ["Harvard", "Yale", "Dartmouth"]
    transition = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.3, 0.4, 0.3],
            [0.2, 0.1, 0.7],
        ],
        dtype=float,
    )
    transition_squared = transition @ transition
    modified = transition.copy()
    modified[0] = np.array([1.0, 0.0, 0.0])
    modified_squared = modified @ modified
    answer_a = float(transition_squared[0, 0])
    answer_b = float(modified_squared[0, 0])

    save_csv(
        result_dir / "problem3_transition_matrix.csv",
        ["from_state", *states],
        [[states[i], *[f"{transition[i, j]:.10f}" for j in range(3)]] for i in range(3)],
    )
    save_csv(
        result_dir / "problem3_two_step_matrix.csv",
        ["from_state", *states],
        [[states[i], *[f"{transition_squared[i, j]:.10f}" for j in range(3)]] for i in range(3)],
    )
    save_csv(
        result_dir / "problem3_answers.csv",
        ["part", "probability"],
        [["a", f"{answer_a:.10f}"], ["b", f"{answer_b:.10f}"]],
    )
    print(f"  part (a) P(Harvard grandson | Harvard grandfather) = {answer_a:.10f}", file=log)
    print(f"  part (b) modified probability = {answer_b:.10f}", file=log)
    return {
        "transition": transition.tolist(),
        "transition_squared": transition_squared.tolist(),
        "modified_transition": modified.tolist(),
        "modified_transition_squared": modified_squared.tolist(),
        "answer_a": answer_a,
        "answer_b": answer_b,
    }


def solve_stationary_distribution(transition: np.ndarray) -> np.ndarray:
    n = transition.shape[0]
    lhs = transition.T - np.eye(n)
    lhs[-1] = np.ones(n)
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    return np.linalg.solve(lhs, rhs)


def solve_hitting_times(transition: np.ndarray, target_index: int) -> np.ndarray:
    n = transition.shape[0]
    unknowns = [idx for idx in range(n) if idx != target_index]
    lhs = np.eye(len(unknowns))
    rhs = np.ones(len(unknowns))
    for row_index, state in enumerate(unknowns):
        for col_index, next_state in enumerate(unknowns):
            lhs[row_index, col_index] -= transition[state, next_state]
    solution = np.linalg.solve(lhs, rhs)
    hitting = np.zeros(n)
    for idx, state in enumerate(unknowns):
        hitting[state] = solution[idx]
    return hitting


def simulate_first_hitting_time(
    rng: np.random.Generator, transition: np.ndarray, start: int, target: int, trials: int
) -> float:
    values = []
    cumulative = np.cumsum(transition, axis=1)
    for _ in range(trials):
        state = start
        steps = 0
        while state != target:
            steps += 1
            state = int(np.searchsorted(cumulative[state], rng.random(), side="right"))
        values.append(steps)
    return float(np.mean(values))


def simulate_first_return_time(rng: np.random.Generator, transition: np.ndarray, start: int, trials: int) -> float:
    values = []
    cumulative = np.cumsum(transition, axis=1)
    for _ in range(trials):
        steps = 0
        state = start
        while True:
            steps += 1
            state = int(np.searchsorted(cumulative[state], rng.random(), side="right"))
            if state == start:
                values.append(steps)
                break
    return float(np.mean(values))


def problem4(rng: np.random.Generator, result_dir: Path, log: io.StringIO) -> dict[str, object]:
    print("Problem 4: rat-maze Markov chain", file=log)
    states = [1, 2, 3, 4, 5, 6]
    edges = [(1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
    transition = np.zeros((6, 6), dtype=float)
    for a, b in edges:
        transition[a - 1, b - 1] = 1.0
        transition[b - 1, a - 1] = 1.0
    degrees = transition.sum(axis=1)
    transition = transition / degrees[:, None]
    stationary = solve_stationary_distribution(transition)
    hitting_to_5 = solve_hitting_times(transition, target_index=4)
    return_time_room_1 = 1.0 / stationary[0]
    trials = 200_000
    hit_sim = simulate_first_hitting_time(rng, transition, start=0, target=4, trials=trials)
    return_sim = simulate_first_return_time(rng, transition, start=0, trials=trials)

    save_csv(
        result_dir / "problem4_transition_matrix.csv",
        ["from_room", *[f"to_{state}" for state in states]],
        [[states[i], *[f"{transition[i, j]:.10f}" for j in range(6)]] for i in range(6)],
    )
    save_csv(
        result_dir / "problem4_stationary_hitting.csv",
        ["room", "stationary_probability", "hitting_time_to_room_5"],
        [[states[i], f"{stationary[i]:.10f}", f"{hitting_to_5[i]:.10f}"] for i in range(6)],
    )
    save_csv(
        result_dir / "problem4_summary.csv",
        ["quantity", "value"],
        [
            ["expected_steps_room_1_to_room_5", f"{hitting_to_5[0]:.10f}"],
            ["expected_return_time_room_1", f"{return_time_room_1:.10f}"],
            ["simulation_trials", trials],
            ["simulated_steps_room_1_to_room_5", f"{hit_sim:.10f}"],
            ["simulated_return_time_room_1", f"{return_sim:.10f}"],
        ],
    )
    plot_problem4_graph(result_dir / "problem4_maze_graph.png", edges)
    print(f"  stationary distribution = {stationary}", file=log)
    print(f"  E_1[T_5] = {hitting_to_5[0]:.10f}; simulation mean = {hit_sim:.10f}", file=log)
    print(f"  expected return time to room 1 = {return_time_room_1:.10f}; simulation mean = {return_sim:.10f}", file=log)
    return {
        "transition": transition.tolist(),
        "stationary": stationary.tolist(),
        "hitting_to_5": hitting_to_5.tolist(),
        "return_time_room_1": return_time_room_1,
        "simulated_hitting_to_5": hit_sim,
        "simulated_return_time_room_1": return_sim,
    }


def plot_problem4_graph(path: Path, edges: list[tuple[int, int]]) -> None:
    positions = {
        1: (1.0, 2.0),
        2: (0.0, 1.0),
        3: (1.0, 1.0),
        4: (2.0, 1.0),
        5: (1.0, 0.0),
        6: (2.0, 0.0),
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    for a, b in edges:
        xa, ya = positions[a]
        xb, yb = positions[b]
        ax.plot([xa, xb], [ya, yb], color="#444444", linewidth=2.0, zorder=1)
    for node, (x, y) in positions.items():
        ax.scatter([x], [y], s=900, color="#e8f4ff", edgecolor="#1f77b4", linewidth=2.0, zorder=2)
        ax.text(x, y, str(node), ha="center", va="center", fontsize=15, fontweight="bold", zorder=3)
    ax.set_title("Problem 4: graph induced by the maze doors")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def problem5(rng: np.random.Generator, result_dir: Path, log: io.StringIO) -> dict[str, object]:
    print("Problem 5: rectangular-grid Buffon simulation", file=log)
    a = 1.0
    b = 1.5
    ell = 0.8
    n_series = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]
    max_n = max(n_series)

    dx = 0.5 * a * rng.random(max_n)
    dy = 0.5 * b * rng.random(max_n)
    theta = 0.5 * math.pi * rng.random(max_n)
    crosses_vertical = dx <= 0.5 * ell * np.cos(theta)
    crosses_horizontal = dy <= 0.5 * ell * np.sin(theta)
    crosses_both = crosses_vertical & crosses_horizontal
    vertical_cumulative = np.cumsum(crosses_vertical)
    horizontal_cumulative = np.cumsum(crosses_horizontal)
    both_cumulative = np.cumsum(crosses_both)

    p_a = 2.0 * ell / (TRUE_PI * a)
    p_b = 2.0 * ell / (TRUE_PI * b)
    p_ab = ell * ell / (TRUE_PI * a * b)
    rows = []
    results: list[dict[str, float]] = []
    for n in n_series:
        count_a = int(vertical_cumulative[n - 1])
        count_b = int(horizontal_cumulative[n - 1])
        count_ab = int(both_cumulative[n - 1])
        phat_a = count_a / n
        phat_b = count_b / n
        phat_ab = count_ab / n
        pi_a = 2.0 * ell / (a * phat_a)
        pi_b = 2.0 * ell / (b * phat_b)
        pi_ab = ell * ell / (a * b * phat_ab)
        std_a = TRUE_PI * math.sqrt((1.0 - p_a) / (p_a * n))
        std_b = TRUE_PI * math.sqrt((1.0 - p_b) / (p_b * n))
        std_ab = TRUE_PI * math.sqrt((1.0 - p_ab) / (p_ab * n))
        result = {
            "n": n,
            "count_a": count_a,
            "count_b": count_b,
            "count_ab": count_ab,
            "pi_a": pi_a,
            "pi_b": pi_b,
            "pi_ab": pi_ab,
            "abs_error_a": abs(pi_a - TRUE_PI),
            "abs_error_b": abs(pi_b - TRUE_PI),
            "abs_error_ab": abs(pi_ab - TRUE_PI),
            "std_a": std_a,
            "std_b": std_b,
            "std_ab": std_ab,
        }
        results.append(result)
        rows.append(
            [
                n,
                count_a,
                count_b,
                count_ab,
                f"{phat_a:.10f}",
                f"{phat_b:.10f}",
                f"{phat_ab:.10f}",
                f"{pi_a:.10f}",
                f"{pi_b:.10f}",
                f"{pi_ab:.10f}",
                f"{abs(pi_a - TRUE_PI):.10f}",
                f"{abs(pi_b - TRUE_PI):.10f}",
                f"{abs(pi_ab - TRUE_PI):.10f}",
            ]
        )
        print(
            (
                f"  N={n}, pi_A={pi_a:.10f}, pi_B={pi_b:.10f}, "
                f"pi_AB={pi_ab:.10f}, counts=({count_a}, {count_b}, {count_ab})"
            ),
            file=log,
        )

    save_csv(
        result_dir / "problem5_grid_buffon.csv",
        [
            "N",
            "count_A_vertical",
            "count_B_horizontal",
            "count_A_intersection_B",
            "p_hat_A",
            "p_hat_B",
            "p_hat_AB",
            "pi_from_A",
            "pi_from_B",
            "pi_from_AB",
            "abs_error_A",
            "abs_error_B",
            "abs_error_AB",
        ],
        rows,
    )
    save_csv(
        result_dir / "problem5_grid_parameters.csv",
        ["parameter", "value"],
        [
            ["a", f"{a:.10f}"],
            ["b", f"{b:.10f}"],
            ["ell", f"{ell:.10f}"],
            ["theory_P_A", f"{p_a:.10f}"],
            ["theory_P_B", f"{p_b:.10f}"],
            ["theory_P_AB", f"{p_ab:.10f}"],
        ],
    )
    plot_problem5_methods(result_dir / "problem5_grid_buffon_methods.png", results)
    plot_problem5_intersection(result_dir / "problem5_grid_buffon_intersection.png", results)

    del dx, dy, theta, crosses_vertical, crosses_horizontal, crosses_both
    del vertical_cumulative, horizontal_cumulative, both_cumulative

    return {
        "parameters": {"a": a, "b": b, "ell": ell, "p_a": p_a, "p_b": p_b, "p_ab": p_ab},
        "n_convergence": results,
    }


def plot_problem5_methods(path: Path, results: list[dict[str, float]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    n = np.array([item["n"] for item in results], dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    ax.plot(n, [item["pi_a"] for item in results], marker="o", color="#1f77b4", label="from P(A)")
    ax.plot(n, [item["pi_b"] for item in results], marker="s", color="#2ca02c", label="from P(B)")
    ax.plot(n, [item["pi_ab"] for item in results], marker="^", color="#9467bd", label="from P(A and B)")
    ax.axhline(TRUE_PI, color="#d62728", linewidth=1.8, label="true pi")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("estimated pi")
    ax.set_title("Problem 5: rectangular-grid Buffon estimates")
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_problem5_intersection(path: Path, results: list[dict[str, float]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    n = np.array([item["n"] for item in results], dtype=float)
    pi_ab = np.array([item["pi_ab"] for item in results])
    std_ab = np.array([item["std_ab"] for item in results])
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    ax.errorbar(n, pi_ab, yerr=std_ab, marker="o", capsize=3, color="#9467bd", label="from P(A and B)")
    ax.axhline(TRUE_PI, color="#d62728", linewidth=1.8, label="true pi")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("estimated pi")
    ax.set_title("Problem 5(c): intersection-method convergence")
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    log = io.StringIO()
    print(f"HW/09 analysis with numpy random seed {SEED}", file=log)
    print(f"true pi = {TRUE_PI:.15f}", file=log)

    summary = {
        "seed": SEED,
        "true_pi": TRUE_PI,
        "problem1": problem1(rng, result_dir, log),
        "problem2": problem2(rng, result_dir, log),
        "problem3": problem3(result_dir, log),
        "problem4": problem4(rng, result_dir, log),
        "problem5": problem5(rng, result_dir, log),
    }
    write_summary_json(result_dir / "hw09_summary.json", summary)

    log_text = log.getvalue()
    (result_dir / "temp-01.log").write_text(log_text, encoding="utf-8")
    print(log_text, end="")


if __name__ == "__main__":
    main()
