"""Export report/report.md to report/report.docx (for pasting into the sprint's Google Docs template).

    uv run python src/export_docx.py

Uses the pandoc bundled with pypandoc-binary (project dependency), so no system install is needed.
Image paths in the Markdown are resolved relative to report/ and the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402


def main() -> None:
    import pypandoc

    src = REPO / "report" / "report.md"
    dst = REPO / "report" / "report.docx"
    pypandoc.convert_file(
        str(src), "docx", outputfile=str(dst),
        extra_args=["--standalone", f"--resource-path={REPO / 'report'};{REPO}"],
    )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
