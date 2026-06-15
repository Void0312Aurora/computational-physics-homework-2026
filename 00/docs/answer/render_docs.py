from __future__ import annotations

import os
from pathlib import Path

import pypandoc


def ensure_pandoc() -> str:
    try:
        version = str(pypandoc.get_pandoc_version())
    except OSError:
        pypandoc.download_pandoc(delete_installer=True)
        version = str(pypandoc.get_pandoc_version())
    return version


def main() -> None:
    answer_dir = Path(__file__).resolve().parent
    project_root = answer_dir.parents[1]
    source = answer_dir / "answer.md"
    docx_out = answer_dir / "answer.docx"
    pdf_out = answer_dir / "answer.pdf"
    shared_hw_ref = project_root.parent / "docs" / "ref"
    counter_filter = shared_hw_ref / "fix_unnumbered_group_counters.lua"
    resource_path = (
        f"{answer_dir}:{answer_dir / 'assets'}:{project_root}:{project_root / 'result'}:"
        f"{project_root / 'results'}:{shared_hw_ref}"
    )

    ensure_pandoc()
    os.chdir(answer_dir)

    common_args = [
        "--standalone",
        "--resource-path",
        resource_path,
        "--number-sections",
        "--lua-filter",
        str(counter_filter),
    ]
    pdf_args = common_args + [
        "--pdf-engine=xelatex",
        "-V",
        "mainfont=TeX Gyre Pagella",
        "-V",
        "CJKmainfont=Noto Serif CJK SC",
        "-V",
        "monofont=DejaVu Sans Mono",
        "-V",
        "linestretch=1.22",
    ]

    pypandoc.convert_file(
        str(source),
        "docx",
        format="markdown+tex_math_dollars",
        outputfile=str(docx_out),
        extra_args=common_args,
    )
    pypandoc.convert_file(
        str(source),
        "pdf",
        format="markdown+tex_math_dollars",
        outputfile=str(pdf_out),
        extra_args=pdf_args,
    )


if __name__ == "__main__":
    main()
