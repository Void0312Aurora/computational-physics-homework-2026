from __future__ import annotations

import os
from pathlib import Path

import pypandoc


def ensure_pandoc() -> str:
    try:
        version = str(pypandoc.get_pandoc_version())
    except OSError:
        print("Pandoc not found on PATH; downloading a local copy via pypandoc.")
        pypandoc.download_pandoc(delete_installer=True)
        version = str(pypandoc.get_pandoc_version())
    return version


def main() -> None:
    answer_dir = Path(__file__).resolve().parent
    project_root = answer_dir.parents[1]
    source = answer_dir / "answer.md"
    docx_out = answer_dir / "answer.docx"
    pdf_out = answer_dir / "answer.pdf"
    md_format = "markdown+tex_math_dollars"
    shared_hw_ref = project_root.parent / "docs" / "ref"
    resource_path = (
        f"{answer_dir}:{answer_dir / 'assets'}:{project_root}:{project_root / 'result'}:"
        f"{project_root / 'results'}:{shared_hw_ref}"
    )

    print(f"Rendering documents from {source.relative_to(project_root)}")
    pandoc_version = ensure_pandoc()
    print(f"Using pandoc {pandoc_version}")
    os.chdir(answer_dir)

    common_args = [
        "--standalone",
        "--resource-path",
        resource_path,
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

    print(f"Writing {docx_out.relative_to(project_root)}")
    pypandoc.convert_file(
        str(source),
        "docx",
        format=md_format,
        outputfile=str(docx_out),
        extra_args=common_args,
    )
    print(f"Writing {pdf_out.relative_to(project_root)}")
    pypandoc.convert_file(
        str(source),
        "pdf",
        format=md_format,
        outputfile=str(pdf_out),
        extra_args=pdf_args,
    )
    print("Document rendering completed successfully.")


if __name__ == "__main__":
    main()
