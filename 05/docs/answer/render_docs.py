from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pypandoc


def main() -> None:
    answer_dir = Path(__file__).resolve().parent
    project_root = answer_dir.parents[1]
    source = answer_dir / "answer.md"
    docx_out = answer_dir / "answer.docx"
    pdf_out = answer_dir / "answer.pdf"
    latex_dir = answer_dir / "latex"
    latex_pdf = latex_dir / "answer.pdf"
    md_format = "markdown+tex_math_dollars"
    shared_hw_ref = project_root.parent / "docs" / "ref"
    counter_filter = shared_hw_ref / "fix_unnumbered_group_counters.lua"
    resource_path = (
        f"{answer_dir}:{project_root}:{project_root / 'result'}:{project_root / 'results'}:"
        f"{project_root / 'outputs'}:{shared_hw_ref}"
    )

    os.chdir(answer_dir)
    common_args = [
        "--standalone",
        "--resource-path",
        resource_path,
        "--number-sections",
        "--lua-filter",
        str(counter_filter),
    ]
    pypandoc.convert_file(
        str(source),
        "docx",
        format=md_format,
        outputfile=str(docx_out),
        extra_args=common_args,
    )
    subprocess.run(["make"], cwd=latex_dir, check=True)
    shutil.copy2(latex_pdf, pdf_out)


if __name__ == "__main__":
    main()
