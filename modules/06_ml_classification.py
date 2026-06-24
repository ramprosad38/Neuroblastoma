"""
Phase 5 — Subgroup & Clinical Correlation Analysis
- MYCN, INSS stage, age stratified KM curves
- PPEF1 violin plots by stage
- Clinical correlation heatmap
- Forest plot of subgroup HRs
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from config import DATA, OUT, FIG, KEY_GENES, DPI


def subgroup_km(surv_df, group_col, gene=None, expr=None,
                title="", save_path=None):
    kmf = KaplanMeierFitter()
    palette = ["#E53935","#1E88E5","#43A047","#FB8C00","#8E24AA"]
    groups  = sorted(surv_df[group_col].dropna().unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    for grp, col in zip(groups, palette):
        sub = surv_df[surv_df[group_col]==grp].dropna(subset=["OS_time","OS_event"])
        kmf.fit(sub["OS_time"], sub["OS_event"], label=str(grp))
        kmf.plot_survival_function(ax=ax, ci_show=False, color=col)
    ax.set_title(title); ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability"); ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI); print(f"Saved {save_path}")
    plt.close()


def violin_by_stage(expr, meta, gene, cohort="GSE49710", save_path=None):
    idx     = meta[meta["cohort"]==cohort].index
    df      = meta.loc[idx, ["INSS_stage"]].copy()
    df[gene] = expr.loc[gene, idx]
    df      = df.dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(data=df, x="INSS_stage", y=gene,
                   palette="Set2", ax=ax, cut=0)
    ax.set_title(f"{gene} expression by INSS Stage ({cohort})")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI); print(f"Saved {save_path}")
    plt.close()


def clinical_heatmap(meta, genes_expr, save_path=None):
    clin_cols = ["MYCN_amp","INSS_stage","age_days","high_risk_merged"]
    df = meta[clin_cols].dropna().astype(float)
    corr = df.join(genes_expr.T).corr()[clin_cols].drop(clin_cols)
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Gene–Clinical Variable Correlation")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI); print(f"Saved {save_path}")
    plt.close()


def run(expr=None, meta=None):
    print("="*60, "\nPHASE 5 — Subgroup Analysis\n", "="*60)
    if expr is None:
        expr = pd.read_csv(DATA("integrated_expression_811x19860.csv"), index_col=0)
        meta = pd.read_csv(DATA("meta_clinical_unified.csv"), index_col=0)

    surv = meta[(meta["cohort"]=="GSE49710") & meta["OS_time"].notna()].copy()

    subgroup_km(surv, "MYCN_amp",    title="KM by MYCN Amplification",
                save_path=FIG("KM_MYCN_subgroup.png"))
    subgroup_km(surv, "INSS_stage",  title="KM by INSS Stage",
                save_path=FIG("KM_INSS_stage_subgroup.png"))

    for g in KEY_GENES:
        violin_by_stage(expr, meta, g,
                        save_path=FIG(f"violin_{g}_stage.png"))

    clinical_heatmap(meta, expr.loc[KEY_GENES],
                     save_path=FIG("clinical_correlation_heatmap.png"))
    print("Phase 5 complete.\n")


if __name__ == "__main__": run()
