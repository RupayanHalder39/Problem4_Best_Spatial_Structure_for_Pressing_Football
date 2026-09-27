# License and Provenance Notice

## Project controlled code

The repository-level MIT license applies only to the original or project-controlled code in:

- `src/pressing_structure/`
- `src/legacy_analytics/`
- `src/tactical_shared/`
- `scripts/`

These families were migrated from the research workspace and have no separate upstream license notice in the source project.

## Third party code

`src/sports/` is an unmodified vendored subset of Roboflow's `sports` project. It remains governed by the upstream MIT license and copyright notice reproduced in `docs/THIRD_PARTY_SPORTS_LICENSE.txt`. The repository-level license does not replace that notice.

Runtime packages listed in `requirements.txt` are not vendored. Each remains governed by its own upstream license.

No modified third-party code or code of unclear provenance was identified in the public source tree during the 2026-09-27 release audit.

## Excluded material

The repository-level MIT license does not grant rights to:

- broadcast footage or video frames;
- source, tracking, calibration, or match-event data;
- row-level derived datasets unless separately licensed;
- the researcher photograph in `assets/RupayanHalder.jpeg`;
- the SoccerSolver logo in `assets/SoccerSolverLogo.png`;
- third-party figures or assets; or
- third-party code governed by its own license.

The public Figure 2 is a project-generated tracking-only reconstruction and contains no broadcast pixels. Figure 1 and the aggregate result table are project-generated research outputs; their inclusion does not license the excluded underlying data.
