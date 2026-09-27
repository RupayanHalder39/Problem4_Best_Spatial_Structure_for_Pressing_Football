#!/usr/bin/env python3
"""Validate public-package structure without restricted match inputs."""

import csv
import hashlib
import importlib
import re
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/RupayanHalder39/Problem4_Best_Spatial_Structure_for_Pressing_Football"
SOCCERSOLVER = "This research was developed in collaboration with SoccerSolver. SoccerSolver currently works with more than 10 football clubs."
FIGURE_1_MD5 = "176d652899c47abe84667272b34bedd8"


def main():
    required = [
        "README.md",
        "LICENSE",
        "NOTICE.md",
        "CITATION.cff",
        "paper/Pressing_Structure_MIT_Sloan_Public_Final.docx",
        "paper/Pressing_Structure_MIT_Sloan_Public_Final.pdf",
        "results/figures/counterfactual_repositioning_T0-724.png",
        "results/figures/practitioner_pressing_dashboard_t115s.png",
        "assets/RupayanHalder.jpeg",
        "assets/SoccerSolverLogo.png",
        "PUBLIC_RELEASE_AUDIT.md",
    ]
    missing = [item for item in required if not (ROOT / item).is_file()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    markdown = (ROOT / "README.md").read_text(encoding="utf-8")
    if REPOSITORY_URL not in markdown:
        raise SystemExit("README is missing the final repository URL")
    if SOCCERSOLVER not in markdown:
        raise SystemExit("README is missing the exact approved SoccerSolver statement")
    if "withheld" not in markdown.lower() or "broadcast" not in markdown.lower():
        raise SystemExit("README does not document the withheld broadcast-derived videos")
    links = re.findall(r"!?(?:\[[^\]]*\])\(([^)]+)\)", markdown)
    broken = []
    for link in links:
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target = (ROOT / link.split("#", 1)[0]).resolve()
        if not target.exists():
            broken.append(link)
    if broken:
        raise SystemExit(f"broken README links: {broken}")

    public_text = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".py", ".cff", ".txt", ".csv", ".html"}:
            public_text.append(path.read_text(encoding="utf-8", errors="replace"))
    joined = "\n".join(public_text)
    forbidden_terms = [
        "/Users/" + "rupayan/",
        "Public GitHub " + "URL",
        "PLACE" + "HOLDER",
        "ANON" + "YMIZED",
    ]
    for forbidden in forbidden_terms:
        if forbidden in joined:
            raise SystemExit(f"forbidden release text remains: {forbidden}")
    if REPOSITORY_URL not in joined:
        raise SystemExit("repository URL missing from public text")

    included_videos = [path for path in ROOT.rglob("*") if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    if included_videos:
        raise SystemExit(f"restricted videos included: {included_videos}")

    restricted_data = [
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".parquet", ".pkl", ".pickle", ".db", ".sqlite", ".sqlite3"}
    ]
    if restricted_data:
        raise SystemExit(f"restricted row-level or calibration data included: {restricted_data}")

    summary_path = ROOT / "results/tables/verified_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    expected_summary = {
        "detected_press_episodes": "19",
        "features_per_episode": "59",
        "skipped_feature_rows": "0",
        "ball_regain_outcomes": "2",
        "forced_backward_outcomes": "1",
        "uncertain_outcomes": "16",
    }
    if {key: summary.get(key) for key in expected_summary} != expected_summary:
        raise SystemExit("verified summary does not match the validated 19/59/0/2/1/16 values")

    figure_1 = ROOT / "results/figures/counterfactual_repositioning_T0-724.png"
    if hashlib.md5(figure_1.read_bytes()).hexdigest() != FIGURE_1_MD5:
        raise SystemExit("Figure 1 changed from the scientifically approved source")

    figure_2 = ROOT / "results/figures/practitioner_pressing_dashboard_t115s.png"
    docx_path = ROOT / "paper/Pressing_Structure_MIT_Sloan_Public_Final.docx"
    with ZipFile(docx_path) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        paper_text = re.sub(r"<[^>]+>", " ", document_xml)
        paper_text = re.sub(r"\s+", " ", paper_text)
        embedded_figure_1 = archive.read("word/media/image1.png")
        embedded_figure_2 = archive.read("word/media/image2.png")
    if hashlib.md5(embedded_figure_1).hexdigest() != FIGURE_1_MD5:
        raise SystemExit("DOCX Figure 1 differs from the approved scientific figure")
    if embedded_figure_2 != figure_2.read_bytes():
        raise SystemExit("DOCX Figure 2 does not match the public tracking-only figure")
    paper_requirements = [
        REPOSITORY_URL,
        SOCCERSOLVER,
        "detects 19 press episodes",
        "zero skipped cases",
        "59 features",
        "two BALL_REGAIN",
        "one FORCED_BACKWARD",
        "16 remain UNCERTAIN",
        "Tracking-only practitioner view",
        "Zero is the signed-dominance balance reference",
        "0.633 intensity threshold gates low-pressure periods",
        "does not support a reliable correlation with goals",
    ]
    missing_paper_text = [text for text in paper_requirements if text not in paper_text]
    if missing_paper_text:
        raise SystemExit(f"final paper is missing validated text: {missing_paper_text}")
    if "All 19 episodes yield" in paper_text:
        raise SystemExit("superseded 'All 19' results wording remains in the paper")

    for module in [
        "pressing_structure.analytics.pressure_field",
        "pressing_structure.analytics.pressing_v4",
        "pressing_structure.analytics.pressing_features",
        "pressing_structure.analytics.pressing_structure_value",
        "pressing_structure.analytics.counterfactual_pressing",
        "pressing_structure.models.baselines",
    ]:
        importlib.import_module(module)

    symlinks = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_symlink()]
    if symlinks:
        raise SystemExit(f"symlinks are not allowed: {symlinks}")

    print("public package validation passed")


if __name__ == "__main__":
    main()
