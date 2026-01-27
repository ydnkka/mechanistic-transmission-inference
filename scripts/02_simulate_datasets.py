#!/usr/bin/env python3
"""
scripts/02_simulate_datasets.py

Generate synthetic datasets across scenarios and save canonical artefacts:
  - populated tree (with epidemic + sampling times)
  - packed genomes
  - pairwise dataset (including Related, distances, AnySampled, etc.)
  - scenario summary + distribution figures

This script should run ONCE to produce datasets reused by:
  - scripts/03_evaluate_edges.py
  - scripts/04_run_clustering.py

Config
------
config/paths.yaml:
  data:
    processed: data/processed
  outputs:
    figures:
      supplementary: figures/supplementary
    tables:
      supplementary: tables/supplementary

config/baseline.yaml:
  simulate:
    tree_gml: "data/processed/..."
    rng_seed: 42
    gen_length: 29903
    out_dir: "data/processed/synthetic"
    save_formats: ["png","pdf"]
  scenarios:
    ... same mapping as in evaluate_edges.py (truth parameters live here)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import networkx as nx

from epilink import (
    TOIT,
    InfectiousnessParams,
    populate_epidemic_data,
    simulate_genomic_data,
    generate_pairwise_data,
)

from utils import *


def hart_default_params() -> InfectiousnessParams:
    return InfectiousnessParams(
        k_inc=5.807, scale_inc=0.948, k_E=3.38, mu=0.37, k_I=1, alpha=2.29
    )

def plot_hist(x: np.ndarray, title: str, xlabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x[np.isfinite(x)], bins=100, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    return fig

def plot_bar(x: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([0, 1], [int((x == 0).sum()), int((x == 1).sum())])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Unrelated", "Related"])
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{title}: Class balance (Sampled)")
    ax.grid(True, axis="y", alpha=0.3)
    return fig


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    parser.add_argument("--scenarios", default="../config/simulate_datasets.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    scenarios_cfg = load_yaml(Path(args.scenarios))

    processed_dir = Path(deep_get(paths_cfg, ["data", "processed", "synthetic"], "../data/processed/synthetic"))
    figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "../figures/supplementary"))
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))

    figs_dir = figs_dir / "sim_distributions"

    ensure_dirs(processed_dir, figs_dir, tabs_dir)

    tree_path = Path(deep_get(scenarios_cfg, ["backbone", "tree_gml"],
                              "../data/processed/synthetic/scovmod_tree_n5000.gml"))
    rng_seed = int(deep_get(scenarios_cfg, ["backbone", "rng_seed"], 12345))
    gen_length = int(deep_get(scenarios_cfg, ["backbone", "gen_length"], 29903))
    formats = list(deep_get(scenarios_cfg, ["backbone", "save_formats"], ["png", "pdf"]))

    base_tree = nx.read_gml(tree_path)

    scenarios = deep_get(scenarios_cfg, ["scenarios"], None)
    if not isinstance(scenarios, dict) or len(scenarios) == 0:
        raise ValueError("simulate_datasets.yaml must define a `scenarios:` mapping.")

    # Defaults (same philosophy as earlier)
    baseline = scenarios["Baseline"]

    summary_rows = []

    for name, patch in scenarios.items():
        cfg = baseline.copy()
        if isinstance(patch, dict):
            cfg.update(patch)

        print(f"\n>>> Simulating scenario: {name} | {cfg.get('description','')}")
        sc_dir = processed_dir / f"scenario={name}"
        ensure_dirs(sc_dir)

        params = InfectiousnessParams(
            k_inc=float(cfg["k_inc"]),
            scale_inc=float(cfg["scale_inc"]),
            k_E=float(cfg["k_E"]),
            mu=float(cfg["mu"]),
            k_I=float(cfg["k_I"]),
            alpha=float(cfg["alpha"]),
        )

        toit = TOIT(
            params=params,
            rng_seed=rng_seed,
            subs_rate=float(cfg["subs_rate"]),
            relax_rate=bool(cfg["relax_rate"]),
            subs_rate_sigma=float(cfg["subs_rate_sigma"]),
            gen_len=gen_length,
        )

        populated_tree = populate_epidemic_data(
            toit=toit,
            tree=base_tree,
            prop_sampled=float(cfg["prop_sampled"]),
            sampling_shape=float(cfg["sampling_shape"]),
            sampling_scale=float(cfg["sampling_scale"]),
        )

        gen_results = simulate_genomic_data(
            toit=toit,
            tree=populated_tree
        )

        pairwise = generate_pairwise_data(
            packed_genomic_data=gen_results["packed"],
            tree=populated_tree,
        )

        # Save canonical artefacts
        nx.write_gml(populated_tree, sc_dir / "populated_tree.gml")
        pairwise.to_parquet(sc_dir / "pairwise.parquet", index=False)

        # Minimal scenario summary
        df = pairwise.copy()
        sampled = df[df["Sampled"]].copy()

        rel = int(sampled["Related"].sum())
        n_pairs = int(len(sampled))
        prevalence = float(sampled["Related"].mean())

        summary_rows.append({
            "Scenario": name,
            "Description": cfg.get("description", ""),
            "N_pairs_Sampled": n_pairs,
            "N_related_Sampled": rel,
            "Prevalence_Sampled": prevalence,
            "k_inc":float(cfg["k_inc"]),
            "scale_inc":float(cfg["scale_inc"]),
            "k_E":float(cfg["k_E"]),
            "mu":float(cfg["mu"]),
            "k_I":float(cfg["k_I"]),
            "alpha":float(cfg["alpha"]),
            "prop_sampled": float(cfg["prop_sampled"]),
            "sampling_shape":float(cfg["sampling_shape"]),
            "sampling_scale":float(cfg["sampling_scale"]),
            "subs_rate": float(cfg["subs_rate"]),
            "relax_rate": bool(cfg["relax_rate"]),
            "subs_rate_sigma": float(cfg["subs_rate_sigma"]),
            "out_dir": str(sc_dir),
        })

        # Distribution plots (for “view distributions” section)
        fig = plot_hist(sampled["TemporalDist"].to_numpy(), f"{name}: Temporal distance", "Days")
        save_figure(fig, figs_dir / f"{name}_temporal_dist", formats); plt.close(fig)

        fig = plot_hist(sampled["PoissonDist"].to_numpy(), f"{name}: Poisson Genetic distance", "Distance")
        save_figure(fig, figs_dir / f"{name}_poisson_genetic_dist", formats)
        plt.close(fig)

        fig = plot_hist(sampled["LinearDist"].to_numpy(), f"{name}: Linear Genetic distance", "Distance")
        save_figure(fig, figs_dir / f"{name}_linear_genetic_dist", formats)
        plt.close(fig)

        fig = plot_bar(sampled["Related"].to_numpy(), name)
        save_figure(fig, figs_dir / f"{name}_class_balance", formats)
        plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tabs_dir / "scenario_summary.csv", index=False)

    print(f"\nSaved datasets to: {processed_dir}")
    print(f"Saved scenario summary to: {tabs_dir / 'scenario_summary.csv'}")
    print(f"Saved distribution figures to: {figs_dir}")


if __name__ == "__main__":
    main()