#!/usr/bin/env python3
"""Replace the public-repository placeholder in the final DOCX."""

import argparse
from pathlib import Path

from docx import Document


LEGACY_MARKER = "Open-source repository: [Public GitHub " + "URL to be added before submission]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    document = Document(args.docx)
    replacements = 0
    for paragraph in document.paragraphs:
        if LEGACY_MARKER not in paragraph.text:
            continue
        for run in paragraph.runs:
            if LEGACY_MARKER in run.text:
                run.text = run.text.replace(LEGACY_MARKER, f"Open-source repository: {args.url}")
                replacements += 1
                break
        else:
            paragraph.text = paragraph.text.replace(LEGACY_MARKER, f"Open-source repository: {args.url}")
            replacements += 1

    if replacements != 1:
        raise SystemExit(f"expected exactly one paper placeholder, found {replacements}")
    document.save(args.docx)
    print(f"updated {args.docx}")


if __name__ == "__main__":
    main()
