"""
Part 5: simple baselines, defined and implemented BEFORE any advanced
modeling, to test whether the richer geometric feature set
(`pressing_features.py`) actually adds value over trivial signals.

Four baselines, increasing in feature richness only (never in model
complexity beyond what's asked for):

    A. nearest-presser distance only
    B. nearest-presser distance + relative closing speed
    C. hand-crafted "core geometry" set (nearest/second/mean presser
       distance, presser counts within bands, local numerical
       superiority)
    D. the full proposed feature set (every numeric `pressing_features.py`
       column)

Backing estimator is selectable (`"logistic"` or `"forest"`, both
`sklearn` -- a declared project dependency, never reimplemented) and
both natively expose calibrated-enough `predict_proba` for this data
scale; no isotonic/Platt recalibration layer is added on top given how
little real labeled data exists today (see module docstring bottom).

Match-grouped cross-validation (`fit_eval_grouped`) uses
`sklearn.model_selection.GroupKFold`/`LeaveOneGroupOut` on `match_id` --
never a random split (Part 11's own leakage rule). With today's single
match, `LeaveOneGroupOut` degenerates to zero splits (nothing to leave
out) -- `fit_eval_grouped` detects this and returns a status flag
rather than silently doing something else.

HONESTY NOTE: with today's real dataset (19 episodes, one match,
0 human-labeled outcomes, only 3/19 heuristic outcomes non-UNCERTAIN),
there is not enough real labeled data to genuinely train or evaluate
any of these baselines. This module is implemented and unit-tested
against synthetic fixtures (see `tests/test_research_pipeline.py`) --
running it against the real parquet is expected, and made explicit
here, to report `INSUFFICIENT_LABELED_DATA`, never a fabricated score.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

MIN_ROWS_PER_CLASS_TO_FIT = 2  # sklearn's own hard floor for stratified/most classifiers; documented, not arbitrary-feeling

BASELINE_FEATURES = {
    "A": ["geom_nearest_presser_distance_cm"],
    "B": ["geom_nearest_presser_distance_cm", "motion_nearest_presser_closing_cm_s"],
    "C": ["geom_nearest_presser_distance_cm", "geom_second_presser_distance_cm", "geom_mean_presser_distance_cm",
          "geom_n_within_5m", "geom_local_numerical_superiority"],
    # "D" is resolved dynamically at fit time to every numeric pressing_features.py column present in the data
    # (see `full_feature_columns`), so it automatically stays in sync with pressing_features.py rather than
    # duplicating its column list here.
}


def full_feature_columns(rows: Sequence[Dict]) -> List[str]:
    """Every `geom_`/`motion_`/`escape_`/`field_`/`ctx_` numeric column
    present across `rows` -- Baseline D's feature set, derived from
    `pressing_features.py`'s own output rather than hand-duplicated."""
    if not rows:
        return []
    prefixes = ("geom_", "motion_", "escape_", "field_", "ctx_")
    cols = set()
    for r in rows:
        for k, v in r.items():
            if k.startswith(prefixes) and isinstance(v, (int, float)) and not isinstance(v, bool):
                cols.add(k)
    return sorted(cols)


def _to_matrix(rows: Sequence[Dict], columns: List[str]) -> np.ndarray:
    """Missing/None values -> np.nan; caller decides imputation (kept
    explicit, never silently zero-filled, since 0 is a meaningful value
    for several of these features e.g. `local_numerical_superiority`)."""
    return np.array([[float(r.get(c)) if r.get(c) is not None else np.nan for c in columns] for r in rows], dtype=float)


class PressingBaseline:
    """One baseline (A/B/C/D) wrapping a single sklearn classifier over
    a fixed, named feature subset. `impute` (default "mean") fills
    np.nan columns from TRAINING-fold statistics only (never test-fold
    statistics -- would itself be a leakage bug)."""

    def __init__(self, name: str, feature_columns: Optional[List[str]] = None, estimator: str = "logistic", random_state: int = 0):
        if name not in ("A", "B", "C", "D"):
            raise ValueError(f"unknown baseline name {name!r}, expected one of A/B/C/D")
        self.name = name
        self.feature_columns = feature_columns  # resolved at fit() time for "D"
        self.estimator_kind = estimator
        self.random_state = random_state
        self.model = None
        self._impute_values = None
        self._columns = None

    def _make_estimator(self):
        if self.estimator_kind == "logistic":
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
        if self.estimator_kind == "forest":
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        raise ValueError(f"unknown estimator {self.estimator_kind!r}, expected 'logistic' or 'forest'")

    def fit(self, rows: Sequence[Dict], y: Sequence):
        columns = self.feature_columns or (BASELINE_FEATURES.get(self.name) if self.name != "D" else full_feature_columns(rows))
        self._columns = columns
        X = _to_matrix(rows, columns)
        self._impute_values = np.nanmean(X, axis=0)
        self._impute_values = np.where(np.isnan(self._impute_values), 0.0, self._impute_values)  # an all-missing column imputes to 0, not NaN
        X = np.where(np.isnan(X), self._impute_values, X)
        self.model = self._make_estimator()
        self.model.fit(X, list(y))
        return self

    def predict(self, rows: Sequence[Dict]):
        X = _to_matrix(rows, self._columns)
        X = np.where(np.isnan(X), self._impute_values, X)
        return self.model.predict(X)

    def predict_proba(self, rows: Sequence[Dict]):
        X = _to_matrix(rows, self._columns)
        X = np.where(np.isnan(X), self._impute_values, X)
        return self.model.predict_proba(X)


def fit_eval_grouped(name: str, rows: Sequence[Dict], y: Sequence, groups: Sequence, estimator: str = "logistic", n_splits: int = 5) -> Dict:
    """Match-grouped evaluation (Part 11: never a random split). Uses
    `LeaveOneGroupOut` when the number of distinct groups is small
    (<=`n_splits`), else `GroupKFold(n_splits)` -- so with today's ONE
    match this correctly reports it cannot cross-validate at all,
    rather than silently falling back to something leakier."""
    rows, y, groups = list(rows), list(y), list(groups)
    n_groups = len(set(groups))
    class_counts = {c: y.count(c) for c in set(y)}
    if n_groups < 2:
        return dict(status="INSUFFICIENT_MATCH_GROUPS", n_groups=n_groups, n_rows=len(rows),
                     detail="match-grouped CV requires >=2 distinct match_id groups; only 1 present -- "
                            "this is the expected, honestly-reported state until a second match's data exists")
    if any(c < MIN_ROWS_PER_CLASS_TO_FIT for c in class_counts.values()):
        return dict(status="INSUFFICIENT_LABELED_DATA", n_groups=n_groups, n_rows=len(rows),
                     class_counts=class_counts, detail=f"every class needs >={MIN_ROWS_PER_CLASS_TO_FIT} rows to fit/evaluate")

    splitter = LeaveOneGroupOut() if n_groups <= n_splits else GroupKFold(n_splits=n_splits)
    fold_scores = []
    for train_idx, test_idx in splitter.split(rows, y, groups):
        train_rows, test_rows = [rows[i] for i in train_idx], [rows[i] for i in test_idx]
        train_y, test_y = [y[i] for i in train_idx], [y[i] for i in test_idx]
        model = PressingBaseline(name, estimator=estimator).fit(train_rows, train_y)
        pred = model.predict(test_rows)
        fold_scores.append(dict(n_train=len(train_rows), n_test=len(test_rows),
                                 accuracy=float(accuracy_score(test_y, pred)),
                                 macro_f1=float(f1_score(test_y, pred, average="macro", zero_division=0))))
    return dict(status="EVALUATED", n_groups=n_groups, n_folds=len(fold_scores), fold_scores=fold_scores,
                mean_accuracy=float(np.mean([s["accuracy"] for s in fold_scores])),
                mean_macro_f1=float(np.mean([s["macro_f1"] for s in fold_scores])))


def run_on_real_pipeline_output(rows: List[Dict], target_column: str = "heuristic_outcome") -> Dict:
    """Entry point for the real `pressing_episode_features.parquet`
    rows. Reports `INSUFFICIENT_LABELED_DATA` honestly rather than
    training/evaluating a baseline on today's real data -- see this
    module's own docstring for the exact real counts."""
    y = [r.get(target_column) for r in rows]
    groups = [r.get("match_id") for r in rows]
    class_counts = {c: y.count(c) for c in set(y) if c is not None}
    if len(set(groups)) < 2 or any(v < MIN_ROWS_PER_CLASS_TO_FIT for v in class_counts.values()) or len(class_counts) < 2:
        return dict(status="INSUFFICIENT_LABELED_DATA", n_rows=len(rows), n_matches=len(set(groups)),
                     class_counts=class_counts,
                     detail="real dataset today: 1 match, 19 episodes, only 3/19 non-UNCERTAIN heuristic "
                            "outcomes, 0 human-labeled outcomes -- baselines are implemented and unit-tested "
                            "on synthetic fixtures; awaiting more matches / real labels before training here")
    return {"A": fit_eval_grouped("A", rows, y, groups), "B": fit_eval_grouped("B", rows, y, groups),
            "C": fit_eval_grouped("C", rows, y, groups), "D": fit_eval_grouped("D", rows, y, groups)}
