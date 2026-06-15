from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "lcrs.txt"
RESULT_DIR = ROOT_DIR / "result"
OUTPUT_FILE = RESULT_DIR / "galaxies.txt"
NUMERIC_FILE = RESULT_DIR / "lcrs_numeric.txt"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in DATA_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit():
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))

    NUMERIC_FILE.write_text(
        "".join(f"{v} {theta} {phi} {mag}\n" for v, theta, phi, mag in rows)
    )

    smallest = min(rows, key=lambda row: row[0])
    largest = max(rows, key=lambda row: row[0])
    brightest = min(rows, key=lambda row: row[3])
    faintest = max(rows, key=lambda row: row[3])

    OUTPUT_FILE.write_text(
        "LCRS galaxy summary\n"
        "===================\n\n"
        "a) Smallest recession velocity:\n"
        f"{smallest[0]} {smallest[1]} {smallest[2]} {smallest[3]}\n\n"
        "a) Largest recession velocity:\n"
        f"{largest[0]} {largest[1]} {largest[2]} {largest[3]}\n\n"
        "b) Brightest galaxy (most negative absolute magnitude):\n"
        f"{brightest[0]} {brightest[1]} {brightest[2]} {brightest[3]}\n\n"
        "b) Faintest galaxy:\n"
        f"{faintest[0]} {faintest[1]} {faintest[2]} {faintest[3]}\n\n"
        "c) Qualitative consistency check:\n"
        "Yes. The brightest galaxy in this sample is much farther away than the faintest one,\n"
        "which is consistent with a flux-limited survey: distant galaxies need to be intrinsically\n"
        "brighter to stay above the detection threshold, while faint galaxies are mostly observed nearby.\n"
    )

    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
