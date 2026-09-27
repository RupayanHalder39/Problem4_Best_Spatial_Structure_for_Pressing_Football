"""
Part 8: pressing structure archetypes. Scaffolded per the brief's own
instruction ("if current data is insufficient, scaffold it and
explicitly mark it as future-data dependent") -- the extraction and
clustering CODE is real and runnable, but this module's own
`MIN_EPISODES_FOR_CLUSTERING` gate refuses to produce clusters on
today's 19-episode, one-match dataset and says so explicitly, rather
than emitting 8 candidate-archetype labels no reviewer could
meaningfully validate against a handful of points.

Candidate interpretable archetype NAMES (single-pressure,
double-pressure, touchline trap, central funnel, cover-shadow press,
counterpress swarm, delayed containment, overload press) are NOT
hard-coded onto the clustering output -- per the brief, unsupervised
clusters are produced first, and any archetype NAME is a separate,
optional, manually-assigned label attached to a cluster only after
human inspection (`assign_archetype_label`), never inferred
automatically from cluster index.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pressing_structure.models.baselines import full_feature_columns, _to_matrix

MIN_EPISODES_FOR_CLUSTERING = 40  # a structural, disclosed floor (>= ~5x the smallest plausible k=8 archetype count so every cluster could in principle hold >=5 points) -- not tuned to this clip's own 19

CANDIDATE_ARCHETYPE_NAMES = (
    "single-pressure", "double-pressure", "touchline-trap", "central-funnel",
    "cover-shadow-press", "counterpress-swarm", "delayed-containment", "overload-press",
)

ARCHETYPE_FEATURE_COLUMNS = [
    "geom_nearest_presser_distance_cm", "geom_second_presser_distance_cm", "geom_n_within_5m",
    "geom_n_within_8m", "geom_local_numerical_superiority", "geom_compactness_cm",
    "motion_nearest_presser_closing_cm_s", "motion_n_participants", "ctx_touchline_proximity_cm",
]  # a deliberately structural (geometry/motion/context) subset -- outcome/field-pressure columns are excluded so a structure cluster is not circularly defined by its own consequence


def extract_structural_features(rows: Sequence[Dict], columns: Optional[List[str]] = None) -> np.ndarray:
    """Standardized (zero-mean, unit-variance per column, fit on THESE
    rows) structural feature matrix -- `columns` defaults to
    `ARCHETYPE_FEATURE_COLUMNS`, falling back to `full_feature_columns`
    only if none of them are present (defensive, not the normal path)."""
    columns = columns or [c for c in ARCHETYPE_FEATURE_COLUMNS if rows and c in rows[0]]
    if not columns:
        columns = full_feature_columns(rows)
    X = _to_matrix(rows, columns)
    impute = np.nanmean(X, axis=0)
    impute = np.where(np.isnan(impute), 0.0, impute)
    X = np.where(np.isnan(X), impute, X)
    return StandardScaler().fit_transform(X), columns


def cluster_structures(rows: Sequence[Dict], k: int = 4, random_state: int = 0) -> Dict:
    """Unsupervised KMeans over structural features. Refuses (honestly)
    below `MIN_EPISODES_FOR_CLUSTERING` -- see module docstring."""
    if len(rows) < MIN_EPISODES_FOR_CLUSTERING:
        return dict(status="INSUFFICIENT_EPISODES_FOR_CLUSTERING", n_episodes=len(rows),
                     min_required=MIN_EPISODES_FOR_CLUSTERING,
                     detail=f"real dataset today has {len(rows)} episodes from 1 match -- clustering into "
                            f"{k} archetypes on this few points would not be a meaningful/reviewable result; "
                            "awaiting more matches before running this")
    X, columns = extract_structural_features(rows)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    return dict(status="CLUSTERED", n_episodes=len(rows), k=k, columns=columns,
                labels=model.labels_.tolist(), cluster_centers=model.cluster_centers_.tolist(),
                inertia=float(model.inertia_))


def assign_archetype_label(cluster_id: int, archetype_name: str, assignments: Optional[Dict[int, str]] = None) -> Dict[int, str]:
    """Manual, human-reviewed assignment of a `CANDIDATE_ARCHETYPE_NAMES`
    name to a cluster id -- NEVER done automatically by
    `cluster_structures`. `assignments` (a {cluster_id: name} dict) is
    threaded through so a caller can build up the full mapping one
    reviewed cluster at a time; returns the updated dict."""
    if archetype_name not in CANDIDATE_ARCHETYPE_NAMES:
        raise ValueError(f"{archetype_name!r} is not one of CANDIDATE_ARCHETYPE_NAMES -- "
                          "add it there first if a genuinely new archetype is being introduced, "
                          "rather than passing an ad hoc string")
    assignments = dict(assignments or {})
    assignments[cluster_id] = archetype_name
    return assignments


def run_on_real_pipeline_output(rows: List[Dict], k: int = 4) -> Dict:
    """Entry point for the real `pressing_episode_features.parquet`
    rows -- reports the honest `INSUFFICIENT_EPISODES_FOR_CLUSTERING`
    status on today's data (19 episodes)."""
    return cluster_structures(rows, k=k)
