from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


LATEX_DIR = Path(__file__).resolve().parent
SOURCE = LATEX_DIR.parent / "answer.md"
TEX_SOURCE = LATEX_DIR / "answer.tex"
MANIFEST = LATEX_DIR / "source_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_manifest() -> dict[str, str]:
    return {
        "source_path": "../answer.md",
        "source_sha256": sha256_file(SOURCE),
        "latex_path": "answer.tex",
        "latex_sha256": sha256_file(TEX_SOURCE),
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "note": "LaTeX is a frozen snapshot; refresh only after answer.tex is synchronized with answer.md.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether the frozen LaTeX source matches ../answer.md."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Refresh source_manifest.json with the current answer.md hash.",
    )
    args = parser.parse_args()

    manifest = current_manifest()
    if args.write:
        MANIFEST.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {MANIFEST.name}")
        return

    if not MANIFEST.exists():
        raise SystemExit("missing source_manifest.json; run make refresh-source-manifest")

    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if recorded.get("source_sha256") != manifest["source_sha256"]:
        raise SystemExit(
            "answer.md changed after the LaTeX snapshot; update answer.tex/answer.pdf "
            "and run make refresh-source-manifest"
        )
    if recorded.get("latex_sha256") != manifest["latex_sha256"]:
        raise SystemExit(
            "answer.tex changed after source_manifest.json was recorded; verify it is "
            "synchronized with answer.md, then run make refresh-source-manifest"
        )
    print("source manifest ok")


if __name__ == "__main__":
    main()
