#!/usr/bin/env python3
"""
scripts/04.1_sparsify_effects.py

Evaluate the effect of sparsifying a probabilistic transmission network.

Given a pairwise table of mechanistic edge weights (Prob_Mech), we:
  1) apply multiple min_edge_weight thresholds
  2) build graphs
  3) run Leiden at multiple resolution parameters
  4) compare partitions across thresholds for matching resolutions using ARI:
        part1.compare_to(part2, method="ari")

Inputs
------
A pairwise DataFrame with at least:
  - Case1, Case2 : node identifiers (int/str)
  - Prob_Mech    : edge weight in [0,1]

Outputs
-------
tables/supplementary/
  - sparsify_partition_similarity.csv

Optionally:
data/processed/sparsify/
  - partitions_minw=<...>.parquet   (case_id, gamma, cluster_id)

Notes
-----
- This compares *clusterings* induced by different sparsification levels, not accuracy.
- If sparsification removes nodes, ARI is computed on the intersection of nodes that
  appear in both graphs for the given gamma.

Requirements
------------
pip install python-igraph pandas numpy pyyaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

import igraph as ig

from utils import *


# -----------------------------
# Core utilities
# -----------------------------

def sparsify_edges(df: pd.DataFrame, min_w: float, weight_col: str) -> pd.DataFrame:
    """Filter edges by min edge weight."""
    if min_w <= 0:
        return df
    return df.loc[df[weight_col] >= float(min_w)].copy()


def build_igraph_from_pairwise(df: pd.DataFrame) -> ig.Graph:
    edges = df[["NodeA", "NodeB", "MechProbLinearDist"]].to_records(index=False).tolist()
    return ig.Graph.TupleList(
        edges=edges,
        directed=False,
        vertex_name_attr="case_id",
        edge_attrs="weight"
    )


def leiden_partition(g: ig.Graph, gamma: float, n_restarts: int = 1) -> ig.VertexClustering:
    """
    Run Leiden with optional restarts, returning the best modularity solution.
    """
    best = None
    best_q = -np.inf

    for r in range(max(1, n_restarts)):
        part = g.community_leiden(
            weights="weight",
            resolution=float(gamma),
            n_iterations=-1,
        )
        q = g.modularity(part, weights="weight")
        if q > best_q:
            best_q = q
            best = part

    return best


def compare_partitions_ari(
    part_a: ig.VertexClustering,
    part_b: ig.VertexClustering,
) -> Tuple[float, int]:
    """
    Compare partitions using ARI on the intersection of vertex sets.

    Uses igraph's built-in compare_to(method="ari") after subsetting to the same vertices.
    """
    s_a = set(part_a.graph.vs["case_id"])
    s_b = set(part_b.graph.vs["case_id"])

    common = s_a & s_b
    if len(common) < 2:
        return np.nan, int(len(common))

    ari = part_a.compare_to(part_b, method='ari')
    return ari, int(len(common))


# -----------------------------
# Main analysis
# -----------------------------

@dataclass
class Grid:
    min_ws: List[float]
    gammas: List[float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--scenarios", default="../config/simulate_datasets.yaml")
    parser.add_argument("--clustering", default="../config/clustering.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    scenarios_cfg = load_yaml(Path(args.scenarios))
    clus_cfg = load_yaml(Path(args.clustering))

    processed_dir = Path(deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic"))
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))

    sparsify_dir = processed_dir / "sparsify"
    ensure_dirs(tabs_dir, processed_dir, sparsify_dir)

    gmin = float(deep_get(clus_cfg, ["community_detection", "resolution", "min"], 0.1))
    gmax = float(deep_get(clus_cfg, ["community_detection", "resolution", "max"], 1.0))
    gstep = float(deep_get(clus_cfg, ["community_detection", "resolution", "step"], 0.05))
    gammas = list(np.round(np.arange(gmin, gmax + 1e-9, gstep), 10))
    min_ws = list(deep_get(clus_cfg, ["network", "min_edge_weights"], [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]))
    weight_columns = list(deep_get(clus_cfg, ["community_detection", "weight_columns"], ["MechProbLinearDist"]))
    rng_seed = int(deep_get(clus_cfg, ["community_detection", "rng_seed"], 12345))
    import random
    random.seed(rng_seed)
    n_restarts = int(deep_get(clus_cfg, ["community_detection", "n_restarts"], 10))

    grid = Grid(min_ws=min_ws, gammas=gammas)
    sc_dir = processed_dir / f"scenario=baseline"
    df = pd.read_parquet(sc_dir / "pairwise_eval.parquet")

    # Store partitions: dict[min_w][gamma] -> VertexClustering
    partitions = {}

    for min_w in grid.min_ws:
        df_w = sparsify_edges(df, min_w, weight_col=weight_columns[0])

        # If the threshold is too aggressive, the graph can collapse
        if len(df_w) == 0:
            continue

        g_ig = build_igraph_from_pairwise(df_w)
        partitions[min_w] = {}
        for gamma in grid.gammas:
            if g_ig.vcount() < 2 or g_ig.ecount() < 1:
                continue
            part = leiden_partition(g_ig, gamma=gamma, n_restarts=n_restarts)
            partitions[min_w][gamma] = part

    # Pairwise comparisons across min_edge_weight for matching gamma
    compare_rows = []
    min_ws_sorted = sorted(grid.min_ws)

    for i in range(len(min_ws_sorted)):
        for j in range(i + 1, len(min_ws_sorted)):
            w1 = min_ws_sorted[i]
            w2 = min_ws_sorted[j]

            for gamma in grid.gammas:
                if gamma not in partitions.get(w1, {}) or gamma not in partitions.get(w2, {}):
                    continue

                part1 = partitions[w1][gamma]
                part2 = partitions[w2][gamma]

                ari, n_common = compare_partitions_ari(part1, part2)

                compare_rows.append({
                    "gamma": float(gamma),
                    "minw_1": float(w1),
                    "minw_2": float(w2),
                    "ARI": ari,
                    "N_common_nodes": int(n_common),
                    "n_nodes_w1": int(part1.graph.vcount()),
                    "density_w1": float(part1.graph.density()),
                    "density_w2": float(part2.graph.density()),
                    "n_nodes_w2": int(part2.graph.vcount()),
                    "n_edges_w1": int(part1.graph.ecount()),
                    "n_edges_w2": int(part2.graph.ecount()),
                    "weight_column": weight_columns[0],
                })

    comp_df = pd.DataFrame(compare_rows)
    comp_df.to_csv(tabs_dir / "sparsify_partition_similarity.csv", index=False)

    print(f"Saved: {tabs_dir / 'sparsify_partition_similarity.csv'}")


if __name__ == "__main__":
    main()