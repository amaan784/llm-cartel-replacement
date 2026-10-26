"""Export report/report.md to report/report.docx laid out like the sprint's submission template.

    uv run python src/export_docx.py

Uses the pandoc bundled with pypandoc-binary (project dependency), so no system install is needed.
report/reference.docx carries the template's typography (Old Standard TT, 20 pt title, 14/13 pt headings,
1-inch margins). The title block (first heading and the plain lines under it) becomes the title and author lines,
the "## Abstract" section becomes the centered abstract block, and p_N / p_M become subscripts. Image paths are
resolved relative to report/ and the repository root.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402


def to_pandoc_markdown(md: str) -> str:
    lines = md.splitlines()
    if not lines or not lines[0].startswith("# "):
        return md
    q = lambda s: '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    title = lines[0][2:].strip()
    i, authors, keep = 1, [], []
    while i < len(lines) and not lines[i].startswith("## "):
        s = lines[i].strip()
        if s.startswith("[^"):
            keep.append(lines[i])  # footnote definitions stay in the body
        elif s:
            authors.append(s)
        i += 1
    body = lines[i:]
    abstract = []
    if body and body[0].strip() == "## Abstract":
        j = 1
        while j < len(body) and not body[j].startswith("## "):
            abstract.append(body[j])
            j += 1
        body = body[j:]
    head = ["---", f"title: {q(title)}", "author:"] + [f"  - {q(a)}" for a in authors]
    if any(a.strip() for a in abstract):
        head += ["abstract: |"] + ["  " + a for a in abstract]
    text = "\n".join(head + ["---", ""] + keep + [""] + body) + "\n"
    return re.sub(r"\bp_([NM])\b", r"p~\1~", text)  # subscripts in the Word output


def main() -> None:
    import pypandoc

    src = REPO / "report" / "report.md"
    dst = REPO / "report" / "report.docx"
    tmp = REPO / "report" / "_export.md"  # next to report.md so relative image paths resolve the same way
    tmp.write_text(to_pandoc_markdown(src.read_text(encoding="utf-8")), encoding="utf-8")
    args = ["--standalone", f"--resource-path={REPO / 'report'};{REPO}"]
    ref = REPO / "report" / "reference.docx"
    if ref.exists():
        args.append(f"--reference-doc={ref}")
    try:
        pypandoc.convert_file(str(tmp), "docx", outputfile=str(dst), extra_args=args)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
