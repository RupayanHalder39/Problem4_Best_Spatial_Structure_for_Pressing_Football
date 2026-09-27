# Press Labeling Tool

A single self-contained HTML/JS page (`index.html`) implementing the
schema in `pressing_structure/docs/PRESS_LABELING_PROTOCOL.md`. No
build step, no server, no dependencies — open `index.html` directly in
a browser (double-click it, or `file://` it).

## Usage

1. **Choose file** → pick a local video file to annotate. Use the
   dashboard's own broadcast/tactical-feed panel (team-colored player
   boxes visible), extracted or full — the tool only needs a
   `<video>`-playable file (`.mp4` works directly in Chrome/Safari).
2. Set **FPS** (default 30, matches this project's tracking) so frame
   numbers reported by the tool align with the tracking parquet's own
   `frame` column.
3. Play/pause, or step **< frame** / **frame >** one frame at a time to
   find the exact press start/end per `PRESS_LABELING_PROTOCOL.md` §3.
4. **Mark press_start_frame HERE** / **Mark press_end_frame HERE** at
   the current playback position — fills the form fields automatically
   (still editable by hand if you need to correct them).
5. Fill in `pressing_team`, `ball_carrier_team` (rejected if equal —
   that's always a data-entry error), `pressing_players` (comma-
   separated track ids, optional — leave blank if track ids aren't
   visible in your video source and note who visually in `notes`
   instead), `primary_presser` (optional, leave blank if genuinely
   ambiguous — never guessed), `outcome` (the 9-label taxonomy),
   `confidence`, `notes`.
6. **Add Episode** — appends a row to the table below. `support_pressers`
   is computed automatically as `pressing_players` minus
   `primary_presser`.
7. Repeat for every press in the clip.
8. **Export JSON** or **Export CSV** when done — downloads
   `press_labels.json`/`press_labels.csv` to your browser's normal
   download location. Set your **annotator_id** before exporting (top
   right) so two annotators' files don't get confused.

## What this produces

One row per labeled episode, exactly the Part 2 schema:
`press_start_frame, press_end_frame, pressing_team, ball_carrier_team,
pressing_players, primary_presser, support_pressers, outcome,
confidence, notes` (plus `episode_index`/`annotator_id` for joining).
This is the `human_outcome` tier of the 3-tier outcome design
(`PRESS_LABELING_PROTOCOL.md` §2) — never shown the pipeline's own
`heuristic_outcome` before recording, to avoid anchoring.

## Loading two annotators' exports

Both exported CSV/JSON files can be loaded into
`pressing_structure/evaluation/evaluate_outcomes.py`'s
`inter_annotator_kappa` (after aligning rows by matching
`press_start_frame`/`press_end_frame` windows, or by a shared episode
id if one is assigned externally) to compute Cohen's κ once a second
annotator's pass exists.

## Known limitations (by design, not overlooked)

- No video overlay of track ids — if your source video doesn't already
  burn in track-id labels, `pressing_players`/`primary_presser` should
  be left blank and described in `notes`; a research assistant can
  later cross-reference the noted description against the real
  tracking parquet to fill in exact ids.
- Single-tab, single-session — closing the tab without exporting loses
  unsaved work (no autosave, deliberately — this is a lightweight tool,
  not a database; export often).
- No built-in video trimming/download — bring your own video file.
