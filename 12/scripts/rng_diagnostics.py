from __future__ import annotations

from hw12.config import PROBLEM_SEEDS
from hw12.io_utils import ensure_dirs
from hw12.rng_diagnostics import rng_diagnostics


def main() -> None:
    ensure_dirs()
    log: list[str] = []
    result = rng_diagnostics(PROBLEM_SEEDS["rng_diagnostics"], log)
    print("\n".join(log))
    print(f"RNG diagnostic keys: {', '.join(sorted(result))}")


if __name__ == "__main__":
    main()
