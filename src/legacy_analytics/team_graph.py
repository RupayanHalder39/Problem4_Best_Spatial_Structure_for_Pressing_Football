"""
Reusable, project-level (not demo-only) spatial team-graph analytics.

Builds a simple SPATIAL RELATIONSHIP graph per team/frame from stabilized
player pitch positions -- nodes are visible players (goalkeeper marked
separately), edges connect spatially nearby teammates. EDGES DO NOT MEAN
"passing lane," "communication," or "tactical instruction" -- they mean
only "nearby teammate," a purely geometric relationship. Metrics derived
from the graph are labeled "spatial centrality," never "most important
player" or similar tactical claims this data cannot support.

Topology choice (k-NN vs. Delaunay triangulation), decided empirically on
testVideo1_20s frames 300-400 (measured, not assumed -- see
`team_graph_topology_comparison.md`): frame-to-frame edge churn (Jaccard
distance between consecutive frames' raw edge sets) was actually
comparable for both once node positions are already EMA-smoothed
(k-NN k=3: 0.030, Delaunay: 0.026) -- smoothing the underlying positions
did most of the stability work either way, so churn alone did not
decide it. Edge COUNT did: with 10-11 visible outfield players, k-NN
k=3 produced 18-20 edges and Delaunay 20-23 -- both dense enough to risk
looking cluttered on a small radar panel (the task's explicit "do not
make the graph so dense it becomes unreadable" concern). Dropping to
k-NN **k=2** cut this to 14-15 edges, visibly cleaner, while each edge
keeps a simple, explainable meaning ("this player's 2 nearest
teammates") that a triangulation edge does not have to a casual viewer.
**Chosen: k-nearest-neighbor, k=2.**

Uses networkx (already a project dependency via the venv) for the graph
structure and degree/closeness centrality -- not reimplemented, and not
over-engineered beyond what "spatial centrality" needs.
"""
from collections import deque

import networkx as nx
import numpy as np
import polars as pl
from scipy.spatial import Delaunay

from legacy_analytics.position_fill import build_full_display_frame

DEFAULT_K = 2


def _pairwise_edges_knn(nodes: list[dict], k: int = DEFAULT_K) -> set:
    """nodes: [{track_id, x, y}, ...]. Returns a set of frozenset({id_a,
    id_b}) undirected edges -- the union of each node's k nearest
    neighbors (not just mutual nearest neighbors, so a node's own
    "nearby teammate" relationship is always represented even if the
    neighbor's own nearest teammates are elsewhere)."""
    edges = set()
    n = len(nodes)
    if n < 2:
        return edges
    for i, a in enumerate(nodes):
        dists = []
        for j, b in enumerate(nodes):
            if i == j:
                continue
            d = float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))
            dists.append((d, b["track_id"]))
        dists.sort()
        for d, other_id in dists[:min(k, n - 1)]:
            edges.add(frozenset({a["track_id"], other_id}))
    return edges


def _pairwise_edges_delaunay(nodes: list[dict]) -> set:
    """Delaunay triangulation edges -- alternative topology, kept for
    reference/comparison, not used as the default (see module
    docstring)."""
    edges = set()
    n = len(nodes)
    if n < 3:
        return edges
    pts = np.array([[nd["x"], nd["y"]] for nd in nodes])
    try:
        tri = Delaunay(pts)
    except Exception:
        return edges
    ids = [nd["track_id"] for nd in nodes]
    for simplex in tri.simplices:
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            edges.add(frozenset({ids[simplex[a]], ids[simplex[b]]}))
    return edges


def build_team_graph_for_frame(display_df: pl.DataFrame, team_id: int, frame: int,
                                k: int = DEFAULT_K, include_goalkeeper: bool = True,
                                topology: str = "knn") -> dict:
    """One frame's raw (not temporally stabilized -- see
    compute_team_graph_series for that) spatial graph. Returns
    {nodes: [{track_id, x, y, is_goalkeeper}], edges: set of
    frozenset({id_a, id_b}), method: str}."""
    roles = ["player"] + (["goalkeeper"] if include_goalkeeper else [])
    rows = display_df.filter(
        (pl.col("frame") == frame) & (pl.col("display_team_id") == team_id) &
        (pl.col("display_object_type").is_in(roles)) & pl.col("display_x_pitch").is_not_null()
    )
    nodes = [{"track_id": r["track_id"], "x": r["display_x_pitch"], "y": r["display_y_pitch"],
              "is_goalkeeper": r["display_object_type"] == "goalkeeper"} for r in rows.to_dicts()]

    if topology == "delaunay":
        edges = _pairwise_edges_delaunay(nodes)
        method = "Delaunay triangulation"
    else:
        edges = _pairwise_edges_knn(nodes, k=k)
        method = f"k-nearest-neighbor (k={k})"

    return {"nodes": nodes, "edges": edges, "method": method}


def compute_spatial_centrality(nodes: list[dict], edges: set) -> dict:
    """Degree + closeness centrality via networkx. Returns {track_id:
    {degree_centrality, closeness_centrality}} and a `most_central`
    track_id (highest degree, ties broken by closeness) or None if no
    nodes. Labeled "spatial centrality" everywhere it's surfaced -- this
    is a purely geometric property (well-connected to nearby teammates),
    not a claim about tactical importance."""
    g = nx.Graph()
    g.add_nodes_from(nd["track_id"] for nd in nodes)
    g.add_edges_from(tuple(e) for e in edges if len(e) == 2)

    if g.number_of_nodes() == 0:
        return {"per_node": {}, "most_central_track_id": None}

    degree_c = nx.degree_centrality(g)
    try:
        closeness_c = nx.closeness_centrality(g)
    except Exception:
        closeness_c = {n: 0.0 for n in g.nodes}

    per_node = {n: {"degree_centrality": round(degree_c.get(n, 0.0), 3),
                     "closeness_centrality": round(closeness_c.get(n, 0.0), 3)} for n in g.nodes}
    most_central = None
    if per_node:
        most_central = max(per_node.items(),
                            key=lambda kv: (kv[1]["degree_centrality"], kv[1]["closeness_centrality"]))[0]
    return {"per_node": per_node, "most_central_track_id": most_central}


def compute_team_graph_series(tracking_df: pl.DataFrame, team_id: int, k: int = DEFAULT_K,
                               include_goalkeeper: bool = True, stability_window: int = 5,
                               stability_min_votes: int = 3,
                               display_df: pl.DataFrame | None = None) -> dict:
    """Precomputes a TEMPORALLY STABILIZED graph for every frame in the
    clip -- the renderer only looks this up, it does not recompute
    graphs itself (project architecture rule: analytics logic doesn't
    live in the renderer).

    Stability rule (Part 14): an edge is displayed only if it appeared in
    at least `stability_min_votes` of the last `stability_window` frames'
    RAW graphs (simple majority-style hysteresis -- not a hard "must
    persist N frames before first appearing" rule, which would make the
    graph feel laggy; a genuinely new, immediately-repeated spatial
    relationship still shows up within a couple of frames, while a single
    one-off flicker does not). Node positions already come from
    `display_df`'s EMA-smoothed `display_x_pitch/y_pitch` (alpha=0.35,
    the same smoothing already validated for the radar dots elsewhere in
    the project), which independently reduces a large share of the raw
    per-frame noise before the edge-stability filter even runs.

    Returns {frame: {nodes, edges (stabilized), centrality}} for every
    frame in [tracking_df frame_min, frame_max]."""
    if display_df is None:
        display_df = build_full_display_frame(tracking_df)

    fmin, fmax = int(tracking_df["frame"].min()), int(tracking_df["frame"].max())
    history = deque(maxlen=stability_window)
    out = {}
    for f in range(fmin, fmax + 1):
        raw = build_team_graph_for_frame(display_df, team_id, f, k=k, include_goalkeeper=include_goalkeeper)
        history.append(raw["edges"])

        vote_counts = {}
        for edge_set in history:
            for e in edge_set:
                vote_counts[e] = vote_counts.get(e, 0) + 1
        stable_edges = {e for e, c in vote_counts.items() if c >= min(stability_min_votes, len(history))}
        # only keep stable edges between nodes that are actually visible this frame
        visible_ids = {nd["track_id"] for nd in raw["nodes"]}
        stable_edges = {e for e in stable_edges if set(e).issubset(visible_ids)}

        centrality = compute_spatial_centrality(raw["nodes"], stable_edges)
        out[f] = {"nodes": raw["nodes"], "edges": stable_edges, "method": raw["method"],
                  "centrality": centrality}
    return out


def graph_metrics_for_frame(graph: dict) -> dict:
    """Lightweight reusable metrics from one frame's graph: centroid,
    width, depth, mean edge length, density. Complements (does not
    replace) analytics/team_shape.py's window-based metrics."""
    nodes = graph["nodes"]
    if not nodes:
        return {"n_nodes": 0}
    xs = [nd["x"] for nd in nodes]
    ys = [nd["y"] for nd in nodes]
    n = len(nodes)
    edges = graph["edges"]
    edge_lengths = []
    by_id = {nd["track_id"]: nd for nd in nodes}
    for e in edges:
        a, b = tuple(e)
        if a in by_id and b in by_id:
            edge_lengths.append(float(np.hypot(by_id[a]["x"] - by_id[b]["x"], by_id[a]["y"] - by_id[b]["y"])))
    max_possible_edges = n * (n - 1) / 2
    return {
        "n_nodes": n,
        "centroid_x": round(sum(xs) / n, 1), "centroid_y": round(sum(ys) / n, 1),
        "width_cm": round(max(ys) - min(ys), 1) if n > 1 else 0.0,
        "depth_cm": round(max(xs) - min(xs), 1) if n > 1 else 0.0,
        "mean_edge_length_cm": round(sum(edge_lengths) / len(edge_lengths), 1) if edge_lengths else None,
        "n_edges": len(edges),
        "density": round(len(edges) / max_possible_edges, 3) if max_possible_edges > 0 else None,
    }
