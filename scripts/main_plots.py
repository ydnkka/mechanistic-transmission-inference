from __future__ import annotations

import argparse

import pandas as pd

from utils import *

set_seaborn_paper_context()

# Model labels are abbreviated as follows: “Lin” and “Pois” denote linear- and Poisson-based distance models,
# respectively; “Score”, “Mech”, and “Logit” indicate heuristic, mechanistic, and logistic probability formulations.
LABEL_MAP = {
        "LinearDistScore": "Lin–Score",
        "PoissonDistScore": "Pois–Score",
        "MechProbLinearDist": "Lin–Mech",
        "MechProbPoissonDist": "Pois–Mech",
        "LogitProbLinearDist_0.1": "Lin–Logit(0.1)",
        "LogitProbLinearDist_1.0": "Lin–Logit(1)",
        "LogitProbPoissonDist_0.1": "Pois–Logit(0.1)",
        "LogitProbPoissonDist_1.0": "Pois–Logit(1)",
    }

def plot_pr_auc_grid(df, out_path, title, col_wrap=4):
    d = df.loc[:, ["Scenario", "Model", "PR_AUC"]].copy()

    d["DistType"] = d["Model"].apply(
        lambda m: "Poisson" if "Poisson" in m else "Linear"
    )

    d["ModelLabel"] = d["Model"].map(LABEL_MAP)
    model_order = list(LABEL_MAP.values())

    d["ModelLabel"] = d["Model"].map(LABEL_MAP)

    g = sns.catplot(
        data=d,
        kind="bar",
        x="ModelLabel",
        y="PR_AUC",
        hue="DistType",
        col="Scenario",
        col_wrap=col_wrap,
        height=3,
        aspect=1.1,
        sharey=True,
        dodge=False,
        order=model_order,
        legend=False,
    )

    g.set_axis_labels("", "PR-AUC")
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1.05))

    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=90)

    g.figure.suptitle(title, y=1.03)

    g.figure.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    g.figure.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(g.figure)
def plot_heatmaps_best_f1_and_opt_gamma(df, out_path, title):
    d = df.loc[:, ["Scenario", "Weight_Column", "gamma", "BCubed_F1_Score"]].copy()
    d["ModelLabel"] = d["Weight_Column"].map(LABEL_MAP)

    model_order = [m for m in LABEL_MAP.values() if m in d["ModelLabel"].unique()]

    # ---------- Best F1 ----------
    best = (
        d.groupby(["Scenario", "ModelLabel"], as_index=False)["BCubed_F1_Score"]
         .max()
         .rename(columns={"BCubed_F1_Score": "Best_F1"})
    )

    best = best.sort_values(by="Best_F1", ascending=False)
    scenario_order = list(pd.unique(best["Scenario"]))

    mat_f1 = (
        best.pivot(index="Scenario", columns="ModelLabel", values="Best_F1")
            .reindex(index=scenario_order, columns=model_order)
    )

    # ---------- Optimal gamma ----------
    idx = d.groupby(["Scenario", "ModelLabel"])["BCubed_F1_Score"].idxmax()
    opt = (
        d.loc[idx, ["Scenario", "ModelLabel", "gamma"]]
         .rename(columns={"gamma": "Opt_gamma"})
    )

    mat_gamma = (
        opt.pivot(index="Scenario", columns="ModelLabel", values="Opt_gamma")
           .reindex(index=scenario_order, columns=model_order)
    )

    # ---------- Plot ----------
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(8, 10),
        sharey=True,
        sharex=True,
        constrained_layout=True
    )

    sns.heatmap(
        mat_f1,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Best F1-score"},
        ax=axes[0]
    )
    axes[0].set_title("Best F1-score")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Scenario")

    sns.heatmap(
        mat_gamma,
        annot=True,
        fmt=".2g",
        cbar_kws={"label": r"Optimal $\gamma$"},
        ax=axes[1]
    )
    axes[1].set_title(r"Optimal Resolution ($\gamma$)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Scenario")

    fig.suptitle(title)

    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_gamma_curves_grid(df, out_path, metric="BCubed_F1_Score", col_wrap=3):
    d = df.loc[:, ["Scenario", "Weight_Column", "gamma", metric]].copy()
    d["Model"] = d["Weight_Column"].map(LABEL_MAP).fillna(d["Weight_Column"])
    model_order = [m for m in LABEL_MAP.values() if m in d["Model"].unique()]
    scenario_order = list(pd.unique(d["Scenario"]))

    d["Model"] = pd.Categorical(d["Model"], categories=model_order, ordered=True)
    d["Scenario"] = pd.Categorical(d["Scenario"], categories=scenario_order, ordered=True)

    g = sns.relplot(
        data=d,
        x="gamma",
        y=metric,
        hue="Model",
        hue_order=model_order,
        col="Scenario",
        col_wrap=col_wrap,
        kind="line",
        marker="o",
        height=3,
        aspect=1.15
    )

    g.set_titles("{col_name}")
    g.set_axis_labels(r"Resolution ($\gamma$)", "-".join(metric.split("_")[1:]))
    g.set(ylim=(0, 1.05))

    g.figure.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    g.figure.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(g.figure)

def plot_gamma_stability_curves_grid(df, out_path, metric="BCubed_F1_Score", col_wrap=3):
    d = df.loc[:, ["Scenario", "Weight_Column", "gamma1", metric]].copy()
    d["Model"] = d["Weight_Column"].map(LABEL_MAP).fillna(d["Weight_Column"])
    model_order = [m for m in LABEL_MAP.values() if m in d["Model"].unique()]
    scenario_order = list(pd.unique(d["Scenario"]))

    d["Model"] = pd.Categorical(d["Model"], categories=model_order, ordered=True)
    d["Scenario"] = pd.Categorical(d["Scenario"], categories=scenario_order, ordered=True)

    g = sns.relplot(
        data=d,
        x="gamma1",
        y=metric,
        hue="Model",
        hue_order=model_order,
        col="Scenario",
        col_wrap=col_wrap,
        kind="line",
        marker="o",
        height=3,
        aspect=1.15
    )

    g.set_titles("{col_name}")
    g.set_axis_labels(r"Resolution ($\gamma$)", "-".join(metric.split("_")[1:]))
    g.set(ylim=(0, 1.05))

    g.figure.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    g.figure.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(g.figure)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="../config/paths.yaml")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "main"], "../figures/main"))
    sup_figs_dir = Path(deep_get(paths_cfg, ["outputs", "figures", "supplementary"], "../figures/supplementary"))
    sup_figs_dir = sup_figs_dir / "clustering"
    tabs_dir = Path(deep_get(paths_cfg, ["outputs", "tables", "supplementary"], "../tables/supplementary"))
    ensure_dirs(figs_dir, sup_figs_dir, tabs_dir)

    fig1_data = pd.read_csv(tabs_dir / "edge_eval_metrics.csv")
    fig2a_data = pd.read_csv(tabs_dir / "clustering_metrics.csv")
    fig2b_data = pd.read_csv(tabs_dir / "clustering_stability.csv")

    plot_pr_auc_grid(
        df=fig1_data,
        out_path=figs_dir / "fig1_pr_auc_grid",
        title="PR-AUC across scenarios and models"
    )

    plot_heatmaps_best_f1_and_opt_gamma(
        df=fig2a_data,
        out_path=figs_dir / "fig2_heatmap_best_f1",
        title=""  # r"Best F1 score and $\gamma$ across scenarios and models"
    )

    plot_gamma_curves_grid(
        df=fig2a_data,
        out_path=sup_figs_dir / "gamma_curves_grid_F1_Score",
        metric="BCubed_F1_Score",
        col_wrap=3
    )

    plot_gamma_curves_grid(
        df=fig2a_data,
        out_path=sup_figs_dir / "gamma_curves_grid_Precision",
        metric="BCubed_Precision",
        col_wrap=3
    )

    plot_gamma_curves_grid(
        df=fig2a_data,
        out_path=sup_figs_dir / "gamma_curves_grid_Recall",
        metric="BCubed_Recall",
        col_wrap=3
    )

    plot_gamma_stability_curves_grid(
        df=fig2b_data,
        out_path=sup_figs_dir / "gamma_stability_curves_grid_F1_Score",
        col_wrap=3
    )


if __name__ == "__main__":
    main()