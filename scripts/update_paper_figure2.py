#!/usr/bin/env python3
"""Replace Figure 2 and its caption in the final DOCX."""

import shutil
import tempfile
import zipfile
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "paper" / "Pressing_Structure_MIT_Sloan_Public_Final.docx"
FIGURE = ROOT / "results" / "figures" / "practitioner_pressing_dashboard_t115s.png"
CAPTION = (
    "Figure 2. Tracking-only practitioner view combining reconstructed player positions and pressure contours "
    "with signed pressure dominance, team pressure intensity, and press-outcome counts. Zero is the signed-"
    "dominance balance reference; the separate 0.633 intensity threshold gates low-pressure periods."
)


def update_caption():
    document = Document(DOCX)
    matches = [p for p in document.paragraphs if p.text.startswith("Figure 2.")]
    if len(matches) != 1:
        raise SystemExit(f"expected one Figure 2 caption, found {len(matches)}")
    paragraph = matches[0]
    paragraph.runs[0].text = CAPTION
    for run in paragraph.runs[1:]:
        run.text = ""
    document.save(DOCX)


def replace_media():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        unpacked = temp / "unpacked"
        with zipfile.ZipFile(DOCX) as archive:
            archive.extractall(unpacked)
        media = unpacked / "word" / "media" / "image2.png"
        if not media.exists():
            raise SystemExit("expected Figure 2 media part word/media/image2.png")
        shutil.copy2(FIGURE, media)
        rebuilt = temp / "rebuilt.docx"
        with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(unpacked.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(unpacked).as_posix())
        shutil.copy2(rebuilt, DOCX)


def main():
    update_caption()
    replace_media()
    print(DOCX)


if __name__ == "__main__":
    main()
