from __future__ import annotations

import os
from pathlib import Path

import pypandoc


def main() -> None:
    answer_dir = Path(__file__).resolve().parent
    project_root = answer_dir.parents[1]
    source = answer_dir / "answer.md"
    docx_out = answer_dir / "answer.docx"
    pdf_out = answer_dir / "answer.pdf"
    md_format = "markdown+tex_math_dollars"
    shared_hw_ref = project_root.parent / "docs" / "ref"
    resource_path = (
        f"{answer_dir}:{project_root}:{project_root / 'result'}:{project_root / 'results'}:"
        f"{project_root / 'outputs'}:{shared_hw_ref}"
    )

    os.chdir(answer_dir)
    common_args = ["--standalone", "--resource-path", resource_path, "--number-sections"]
    pdf_args = common_args + [
        "--pdf-engine=xelatex",
        "-V",
        "mainfont=TeX Gyre Pagella",
        "-V",
        "CJKmainfont=Noto Serif CJK SC",
        "-V",
        "monofont=DejaVu Sans Mono",
        "-V",
        "linestretch=1.12",
    ]

    pypandoc.convert_file(
        str(source),
        "docx",
        format=md_format,
        outputfile=str(docx_out),
        extra_args=common_args,
    )

    pypandoc.convert_file(
        str(source),
        "pdf",
        format=md_format,
        outputfile=str(pdf_out),
        extra_args=pdf_args,
    )


if __name__ == "__main__":
    main()
