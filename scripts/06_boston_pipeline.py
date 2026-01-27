#!/usr/bin/env python3
"""
scripts/06_boston_application.py

Backbone for the Boston empirical application:
  - load precomputed pairwise distances (TN93 -> SNPs or substitutions)
  - compute mechanistic pairwise probabilities (epilink)
  - build weighted network, run Leiden across gamma grid
  - save cluster summaries and persistence across resolution

This script intentionally does NOT attempt to recreate full paper figures—
those should live in a separate plotting script once the pipeline is stable.

Config (suggested)
------------------
config/paths.yaml
config/clustering.yaml
config/inference.yaml
config/boston.yaml:
  input:
    pairwise_distances: "data/raw/empirical/boston/pairwise_distances.parquet"
  output:
    out_dir: "data/processed/boston"
  distances:
    genetic_col: "SNPs"          # or "tn93_subs"
    temporal_col: "DeltaDays"    # sampling-date difference
  inference:
    mutation_model: "deterministic"
    subs_rate: 1e-3
    relax_rate: false
    subs_rate_sigma: 0.0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as e:
    raise ImportError("Please install PyYAML: `pip install pyyaml`.") from e

import igraph as ig

from epilink import TOIT, InfectiousnessParams, estimate_linkage_probabilities


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def hart_default_params() -> InfectiousnessParams:
    return InfectiousnessParams(k_inc=5.807, scale_inc=0.948, k_E=3.38, mu=0.37, k_I=1, alpha=2.29)

def build_igraph(df: pd.DataFrame, min_w: float, wcol: str) -> ig.Graph:
    if min_w > 0:
        df = df[df[wcol] >= min_w].copy()
    nodes = pd.Index(pd.unique(df[["case_i", "case_j"]].values.ravel()))
    idx = {str(n): i for i, n in enumerate(nodes.astype(str).tolist())}
    edges = [(idx[str(a)], idx[str(b)]) for a, b in df[["case_i", "case_j"]].values]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = df[wcol].astype(float).values.tolist()
    g.vs["case_id"] = nodes.astype(str).tolist()
    return g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--clustering", default="config/clustering.yaml")
    parser.add_argument("--inference", default="config/inference.yaml")
    parser.add_argument("--boston", default="config/boston.yaml")
    parser.add_argument("--out-root", default="")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    clus_cfg = load_yaml(Path(args.clustering))
    inf_cfg = load_yaml(Path(args.inference))
    bos_cfg = load_yaml(Path(args.boston))

    out_root = Path(args.out_root) if args.out_root else None

    out_dir = Path(deep_get(bos_cfg, ["output", "out_dir"], "data/processed/boston"))
    if out_root:
        out_dir = out_root / out_dir
    ensure_dirs(out_dir)

    pw_path = Path(deep_get(bos_cfg, ["input", "pairwise_distances"], "data/raw/empirical/boston/pairwise_distances.parquet"))
    if out_root and not pw_path.is_absolute():
        # keep input relative to repo root unless user wants to mirror under out_root
        pass
    if not pw_path.exists():
        raise FileNotFoundError(f"Boston pairwise file not found: {pw_path}")

    df = pd.read_parquet(pw_path)

    gcol = str(deep_get(bos_cfg, ["distances", "genetic_col"], "SNPs"))
    tcol = str(deep_get(bos_cfg, ["distances", "temporal_col"], "DeltaDays"))

    # Expect case identifiers in columns case_i / case_j
    # (Adjust this to your file schema once finalised.)
    required = {"case_i", "case_j", gcol, tcol}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Boston pairwise file: {missing}")

    # Mechanistic inference settings
    subs_rate = float(deep_get(bos_cfg, ["inference", "subs_rate"], 1e-3))
    relax_rate = bool(deep_get(bos_cfg, ["inference", "relax_rate"], False))
    sigma = float(deep_get(bos_cfg, ["inference", "subs_rate_sigma"], 0.0))
    mutation_model = str(deep_get(bos_cfg, ["inference", "mutation_model"], "deterministic"))
    rng_seed = int(deep_get(bos_cfg, ["inference", "rng_seed"], 42))

    params = hart_default_params()
    toit = TOIT(params=params, rng_seed=rng_seed, subs_rate=subs_rate, relax_rate=relax_rate, subs_rate_sigma=sigma)

    # Compute probabilities
    df["P_mech"] = estimate_linkage_probabilities(
        toit=toit,
        genetic_distance=df[gcol].astype(float).values,
        temporal_distance=df[tcol].astype(float).values,
        mutation_model=mutation_model,
    )
    df.to_parquet(out_dir / "boston_pairwise_with_probs.parquet", index=False)

    # Leiden across gamma grid
    gmin = float(deep_get(clus_cfg, ["community_detection", "resolution", "min"], 0.1))
    gmax = float(deep_get(clus_cfg, ["community_detection", "resolution", "max"], 1.0))
    gstep = float(deep_get(clus_cfg, ["community_detection", "resolution", "step"], 0.05))
    gammas = np.round(np.arange(gmin, gmax + 1e-9, gstep), 10)

    min_w = float(deep_get(clus_cfg, ["network", "sparsify", "min_edge_weight"], 0.01))
    sparsify = bool(deep_get(clus_cfg, ["network", "sparsify", "enabled"], True))
    if not sparsify:
        min_w = 0.0

    seed = int(deep_get(clus_cfg, ["community_detection", "random_seed"], 42))
    n_restarts = int(deep_get(clus_cfg, ["community_detection", "n_restarts"], 10))

    g = build_igraph(df[["case_i", "case_j", "P_mech"]].dropna(), min_w=min_w, wcol="P_mech")

    rows = []
    for gamma in gammas:
        best = None
        best_q = -np.inf
        for r in range(n_restarts):
            part = g.community_leiden(weights="weight", resolution_parameter=float(gamma), n_iterations=-1, seed=seed + r)
            q = g.modularity(part, weights="weight")
            if q > best_q:
                best_q, best = q, part
        rows.append(pd.DataFrame({
            "case_id": g.vs["case_id"],
            "gamma": float(gamma),
            "cluster_id": np.array(best.membership, dtype=int),
        }))

    parts = pd.concat(rows, ignore_index=True)
    parts.to_parquet(out_dir / "boston_leiden_partitions.parquet", index=False)

    # Basic cluster summary (counts per gamma)
    summ = (parts.groupby(["gamma", "cluster_id"])
                 .size()
                 .reset_index(name="cluster_size"))
    summ.to_csv(out_dir / "boston_cluster_sizes.csv", index=False)

    print(f"Saved Boston outputs to: {out_dir}")


if __name__ == "__main__":
    main()