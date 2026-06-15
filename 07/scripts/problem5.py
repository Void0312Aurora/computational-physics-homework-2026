from __future__ import annotations

from hw07_integrals import RESULT_DIR, problem5


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    problem5(log)
    print("\n".join(log))


if __name__ == "__main__":
    main()
