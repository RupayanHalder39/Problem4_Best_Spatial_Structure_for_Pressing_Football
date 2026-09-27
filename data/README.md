# Data Availability

No match-level source data are included in this package.

To reproduce the current study, place lawfully obtained inputs under the ignored `data/private/` directory using this structure:

```text
data/private/
├── tracking/testVideo1_120s/
│   ├── tracking.parquet
│   └── metadata.json
├── analytics/testVideo1_120s_v3/
│   ├── ball_trajectory.parquet
│   ├── passes.parquet
│   ├── turnovers.parquet
│   └── homography_transformers.pkl
├── snapshots/
│   └── v4_snapshot.pkl
└── source_video/
    └── testVideo1_120s.mp4
```

The feature-export command needs the tracking parquet, ball trajectory, passes, and turnovers. Dashboard rendering additionally needs homography calibration, the frozen snapshot, and source video.

The original broadcast footage, tracking data, derived event tables, calibration files, frozen snapshot, and row-level episode/features data were excluded because their public redistribution rights have not been confirmed. They are not covered by the scoped project code license. The source video is needed only to reproduce the original broadcast-aligned dashboard; the tracking and aligned derived inputs are needed to regenerate the scientific feature export.
