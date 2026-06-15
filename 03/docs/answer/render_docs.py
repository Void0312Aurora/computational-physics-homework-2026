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
    doc_placeholder = answer_dir / "answer.doc"
    md_format = "markdown+tex_math_dollars"
    shared_hw_ref = project_root.parent / "docs" / "ref"
    counter_filter = shared_hw_ref / "fix_unnumbered_group_counters.lua"
    resource_path = (
        f"{answer_dir}:{project_root}:{project_root / 'result'}:{project_root / 'results'}:"
        f"{project_root / 'outputs'}:{shared_hw_ref}"
    )

    if not source.exists():
        raise FileNotFoundError(f"missing source markdown: {source}")

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

    doc_placeholder.write_text(
        "Compatibility placeholder. Please use answer.docx or answer.pdf for the fully formatted report.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
