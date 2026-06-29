"""Build a reproducible manifest for existing high-digit pi output files.

This module is intentionally read-only with respect to pi computation. It
hashes and samples an existing decimal output file, then records historical
y-cruncher validation artifacts when they are present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TARGET = Path("results/project2/output/project2_pi_100000000_digits.txt")
DEFAULT_OUTPUT = Path("results/project2/manifest/project2_pi_100000000_digits.txt.manifest.json")
DEFAULT_YCRUNCHER_CSV = Path("results/project2/references/ycruncher/project2_ycruncher_benchmark.csv")
DEFAULT_YCRUNCHER_VALIDATION_DIR = Path("results/project2/references/ycruncher/validation")
WINDOW_BYTES = 80
PI_PREFIX = "3.1415926535897932384626433832795028841971693993751058209749445923078"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def decode_ascii(data: bytes) -> str:
    return data.decode("ascii", errors="replace")


def relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def byte_window(path: Path, offset: int, length: int = WINDOW_BYTES) -> dict[str, Any]:
    size = path.stat().st_size
    safe_offset = max(0, min(offset, max(size - 1, 0)))
    with path.open("rb") as handle:
        handle.seek(safe_offset)
        data = handle.read(length)
    text = decode_ascii(data)
    decimal_start = safe_offset - 1 if safe_offset >= 2 else None
    digits_only = "".join(ch for ch in text if ch.isdigit())
    return {
        "byte_offset": safe_offset,
        "length_bytes": len(data),
        "decimal_digit_start_1based": decimal_start,
        "text": text,
        "digits_only": digits_only,
    }


def stable_offsets(size: int, digest: str, count: int = 4) -> list[int]:
    max_start = max(size - WINDOW_BYTES, 0)
    if max_start == 0:
        return [0]
    seed = hashlib.sha256(f"{size}:{digest}".encode("ascii")).digest()
    offsets: list[int] = []
    for index in range(count):
        chunk = seed[index * 4 : index * 4 + 8]
        value = int.from_bytes(chunk, "big")
        offsets.append(value % (max_start + 1))
    return sorted(set(offsets))


def scan_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    digit_count = 0
    first_bytes = b""
    last_bytes = b""
    ends_with_newline = False
    chunk_size = 1024 * 1024

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            if not first_bytes:
                first_bytes = chunk[:WINDOW_BYTES]
            last_bytes = (last_bytes + chunk)[-WINDOW_BYTES:]
            digit_count += sum(48 <= byte <= 57 for byte in chunk)
            digest.update(chunk)
            ends_with_newline = chunk.endswith(b"\n")

    size = path.stat().st_size
    has_decimal_prefix = first_bytes.startswith(b"3.")
    decimal_digits = (
        digit_count - 1 if has_decimal_prefix and digit_count else digit_count
    )
    sha256 = digest.hexdigest()
    fixed_offsets = [0]
    if size > WINDOW_BYTES:
        fixed_offsets.extend(
            [
                max(0, size // 4),
                max(0, size // 2),
                max(0, (size * 3) // 4),
                max(0, size - WINDOW_BYTES),
            ]
        )

    return {
        "path_exists": True,
        "size_bytes": size,
        "modified_at_utc": utc_from_timestamp(path.stat().st_mtime),
        "sha256": sha256,
        "format": {
            "has_decimal_prefix": has_decimal_prefix,
            "ends_with_newline": ends_with_newline,
            "total_ascii_digits": digit_count,
            "decimal_digits_after_point": decimal_digits,
            "prefix_matches_embedded_reference": decode_ascii(first_bytes).startswith(
                PI_PREFIX[: len(first_bytes)]
            ),
            "embedded_reference_prefix_length": len(PI_PREFIX),
        },
        "windows": {
            "first": byte_window(path, 0),
            "last": byte_window(path, max(0, size - WINDOW_BYTES)),
            "fixed_offsets": [
                byte_window(path, offset) for offset in sorted(set(fixed_offsets))
            ],
            "deterministic_sample_offsets": [
                byte_window(path, offset) for offset in stable_offsets(size, sha256)
            ],
        },
    }


def read_decimal_digits(path: Path, start_1based: int, length: int) -> str:
    if start_1based < 1:
        raise ValueError("decimal digit positions are 1-based")
    byte_offset = start_1based + 1
    with path.open("rb") as handle:
        handle.seek(byte_offset)
        data = handle.read(length)
    return decode_ascii(data)


def parse_ycruncher_benchmark(csv_path: Path, base: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "label": row.get("label", ""),
                    "decimal_digits": row.get("decimal_digits", ""),
                    "pi_seconds": row.get("pi_seconds", ""),
                    "total_seconds": row.get("total_seconds", ""),
                    "wall_seconds": row.get("wall_seconds", ""),
                    "digits_per_second_wall": row.get("digits_per_second_wall", ""),
                    "spot_check": row.get("spot_check", ""),
                    "no_smt": row.get("no_smt", ""),
                    "log_file": row.get("log_file", ""),
                    "csv_path": relpath(csv_path, base),
                }
            )
    return rows


def parse_validation_file(path: Path, base: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    fields: dict[str, Any] = {"path": relpath(path, base)}

    patterns = {
        "program": r"Program:\s+(.+)",
        "decimal_digits": r"Decimal Digits:\s+([\d,]+)",
        "start_date": r"Start Date:\s+(.+)",
        "end_date": r"End Date:\s+(.+)",
        "total_computation_time": r"Total Computation Time:\s+(.+)",
        "wall_time": r"Start-to-End Wall Time:\s+(.+)",
        "dec_hash": r"Dec Hash:\s+(.+)",
        "hex_hash": r"Hex Hash:\s+(.+)",
        "spot_check": r"Spot Check:\s+(.+)",
        "checksum0": r"Checksum0:\s+([0-9a-f]+)",
        "checksum1": r"Checksum1:\s+([0-9a-f]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            fields[key] = match.group(1).strip()

    checkpoints: list[dict[str, Any]] = []
    in_decimal_section = False
    for line in text.splitlines():
        if line.strip() == "Decimal Digits:":
            in_decimal_section = True
            continue
        if in_decimal_section and line.strip() == "Hexadecimal Digits:":
            break
        if not in_decimal_section:
            continue
        match = re.match(r"\s*([0-9 ]+)\s+:\s+([\d,]+)\s*$", line)
        if not match:
            continue
        digits = match.group(1).replace(" ", "")
        end_position = int(match.group(2).replace(",", ""))
        checkpoints.append(
            {
                "end_decimal_digit_1based": end_position,
                "start_decimal_digit_1based": end_position - len(digits) + 1,
                "digits": digits,
            }
        )
    fields["decimal_checkpoints"] = checkpoints
    return fields


def compare_checkpoints(target: Path, validation: dict[str, Any]) -> dict[str, Any]:
    compared = 0
    matched = 0
    mismatches: list[dict[str, Any]] = []
    for checkpoint in validation.get("decimal_checkpoints", []):
        start = int(checkpoint["start_decimal_digit_1based"])
        expected = str(checkpoint["digits"])
        if start < 1:
            continue
        actual = read_decimal_digits(target, start, len(expected))
        compared += 1
        if actual == expected:
            matched += 1
        else:
            mismatches.append(
                {
                    "start_decimal_digit_1based": start,
                    "end_decimal_digit_1based": checkpoint["end_decimal_digit_1based"],
                    "expected": expected,
                    "actual": actual,
                }
            )
    return {
        "validation_path": validation.get("path", ""),
        "checkpoints_compared": compared,
        "checkpoints_matched": matched,
        "mismatches": mismatches,
    }


def collect_ycruncher_references(
    base: Path,
    target: Path,
    target_decimal_digits: int | None,
    csv_path: Path,
    validation_dir: Path,
) -> dict[str, Any]:
    validations = []
    checkpoint_comparisons = []
    if validation_dir.exists():
        for validation_path in sorted(validation_dir.glob("Pi - *.txt")):
            parsed = parse_validation_file(validation_path, base)
            validations.append(
                {
                    key: value
                    for key, value in parsed.items()
                    if key != "decimal_checkpoints"
                }
                | {"checkpoint_count": len(parsed.get("decimal_checkpoints", []))}
            )
            parsed_digits = str(parsed.get("decimal_digits", "")).replace(",", "")
            if (
                target.exists()
                and target_decimal_digits is not None
                and parsed_digits == str(target_decimal_digits)
            ):
                checkpoint_comparisons.append(compare_checkpoints(target, parsed))

    return {
        "note": (
            "Historical y-cruncher logs and validation files are recorded only; "
            "this manifest generator does not run y-cruncher or recompute pi."
        ),
        "benchmark_rows": parse_ycruncher_benchmark(csv_path, base),
        "validation_files": validations,
        "checkpoint_comparisons_against_target": checkpoint_comparisons,
    }


def missing_stub(target: Path, base: Path) -> dict[str, Any]:
    return {
        "path_exists": False,
        "expected_path": relpath(target, base),
        "missing_note": (
            "The high-digit decimal output file is not present. This stub is "
            "kept so the review trail records that no large recomputation was "
            "started while generating the manifest."
        ),
        "generation_guidance": [
            "Do not use make run for 100M/2.5B reproduction; it is the default homework route.",
            "Use an explicit Project 2 high-digit command or a retained historical artifact.",
            "After producing the output, rerun make project2_manifest to hash and sample it.",
            "For independent reference, compare against retained y-cruncher validation files or rerun y-cruncher explicitly outside the default workflow.",
        ],
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    base = Path(args.base).resolve()
    target = (base / args.target).resolve()
    output = (base / args.output).resolve()
    ycruncher_csv = (base / args.ycruncher_csv).resolve()
    validation_dir = (base / args.ycruncher_validation_dir).resolve()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "generator": "project2_pi.manifest_pi_output",
        "output_path": relpath(output, base),
        "target": {
            "path": relpath(target, base),
            "absolute_path": str(target),
        },
        "reproducibility_scope": {
            "default_make_run_triggers_large_recompute": False,
            "large_digit_recompute_policy": (
                "100M/140M/2.5B scale runs are explicit-only historical or manual "
                "experiments, not part of make run or this manifest target."
            ),
        },
    }

    target_decimal_digits = None
    if target.exists():
        scan = scan_file(target)
        target_decimal_digits = scan["format"]["decimal_digits_after_point"]
        manifest["target"].update(scan)
    else:
        manifest["target"].update(missing_stub(target, base))

    manifest["reference_validation"] = {
        "embedded_prefix_reference": {
            "source": "known leading decimal expansion of pi",
            "prefix": PI_PREFIX,
            "used_for": "lightweight prefix sanity check only",
        },
        "ycruncher": collect_ycruncher_references(
            base,
            target,
            target_decimal_digits,
            ycruncher_csv,
            validation_dir,
        ),
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a sidecar manifest for retained Project 2 pi output."
    )
    parser.add_argument("--base", default=".", help="HW/05 root directory")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help="pi output path")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="manifest JSON path"
    )
    parser.add_argument(
        "--ycruncher-csv",
        default=str(DEFAULT_YCRUNCHER_CSV),
        help="historical y-cruncher benchmark CSV path",
    )
    parser.add_argument(
        "--ycruncher-validation-dir",
        default=str(DEFAULT_YCRUNCHER_VALIDATION_DIR),
        help="historical y-cruncher validation directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base).resolve()
    output = (base / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(relpath(output, base))


if __name__ == "__main__":
    main()
