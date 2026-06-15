# Plan

## Scope

- Target folder: `HW/00`
- Purpose: create a reference sample for future HW report structure.
- This is a structure template, not a solved homework problem.

## Problem Summary

- The sample should demonstrate a clean HW report organization.
- The report body should not contain workflow chatter, command transcripts, or deliverable inventories unless a specific homework needs a short file-structure section.
- The sample should keep the problem-by-problem organization used in existing HW reports.

## Approach

- Create `docs/answer/answer.md` as the report-body example.
- Create `docs/answer/section_notes.md` as the editable rule draft for section-level注意事项.
- Reuse the existing portrait asset in `docs/answer/assets/profile.jpg`.
- Provide `docs/answer/render_docs.py` so the sample can be exported when needed.

## Testing

- Render `answer.docx` and `answer.pdf`.
- Check that the PDF contains the portrait and that headings follow the intended structure.

## Risks

- Do not mix the notes document into the final report body.
- Do not turn the report into a file inventory or command log.
