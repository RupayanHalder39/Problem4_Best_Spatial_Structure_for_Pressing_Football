#!/usr/bin/env python3
"""Recompose Figure 2 using only its existing derived analytical panels."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "figures" / "practitioner_pressing_dashboard_t115s.png"


def load_font(size: int, bold: bool = False):
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="private original dashboard PNG containing the derived panels")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = Image.open(args.source).convert("RGB")
    if source.size != (2304, 1204):
        raise SystemExit(f"unexpected source size: {source.size}")

    header = source.crop((0, 0, 2304, 60))
    # The radar begins at x=1216 in this render. Cropping from that
    # boundary intentionally excludes the narrow divider and every
    # residual pixel from the broadcast panel.
    radar = source.crop((1216, 60, 2304, 650))
    charts = source.crop((0, 650, 2304, 1204))

    canvas = Image.new("RGB", source.size, (18, 32, 52))
    canvas.paste(header, (0, 0))
    canvas.paste(radar.resize((2304, 520), Image.Resampling.LANCZOS), (0, 60))
    canvas.paste(charts, (0, 650))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 580, 2304, 650), fill=(20, 36, 58))
    draw.line((0, 580, 2304, 580), fill=(32, 205, 225), width=2)
    title_font = load_font(24, bold=True)
    body_font = load_font(20)
    draw.text((28, 590), "TRACKING-ONLY TACTICAL RECONSTRUCTION", font=title_font, fill=(245, 248, 252))
    legend = (
        "Markers = tracked player locations   |   Contours = modeled defensive pressure "
        "(denser/darker = higher)   |   No broadcast pixels"
    )
    draw.text((28, 620), legend, font=body_font, fill=(185, 199, 216))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format="PNG", optimize=True)
    print(args.output)


if __name__ == "__main__":
    main()
