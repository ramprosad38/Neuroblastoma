"""
Phase 1 — Data Integration & Quality Control
- Loads and merges 3 GEO cohorts on common genes
- Derives GSE120559 risk labels from INSS+MYCN
- PCA batch assessment plot
- Outputs: integrated_expression_811x19860.csv, meta_clinical_unified.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from config import DATA, OUT, FIG, COHORTS, KEY_GENES, DPI


def load_expression(path): return pd.read_csv(path, index_col=0)
def load_metadata(path):   return pd.read_csv(path, index_col=0)


def intersect_genes(*dfs):
    genes = set(dfs[0].index)
    for df in dfs[1:]: genes &= set(df.index)
    return sorted(genes)


def merge_cohorts(expr_dict, common_genes):
    return pd.concat(
        [df.loc[common_genes] for df in expr_dict.values()], axis=1
    )


def pca_batch_plot(expr, meta, color_col="cohort", save_path=None):
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(expr.T.values)
    cohorts = meta[color_col].values
    palette = {"GSE49710":"#E53935","GSE73517":"#1E88E5",
               "GSE120559":"#43A047","TARGET-NBL":"#FB8C00"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in sorted(set(cohorts)):
        idx = cohorts == c
        ax.scatter(coords[idx,0], coords[idx,1],
                   label=c, alpha=0.6, s=15, color=palette.get(c,"grey"))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA — Pre-correction cohort separation")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI); print(f"Saved {save_path}")
    plt.close()


def assign_risk_gse120559(meta):
    idx    = meta[meta["cohort"] == "GSE120559"].index
    stages = meta.loc[idx,"characteristics_ch1.2"].str.extract(
        r"stage: ([\w]+)", expand=False).str.strip().str.upper()
    mycns  = meta.loc[idx,"characteristics_ch1.3"].str.extract(
        r"mycn status: (.+)", expand=False).str.strip().str.lower()

    def _risk(stage, mycn):
        if pd.isna(stage): return np.nan
        s, amp = str(stage).strip().upper(), str(mycn).strip().lower() == "amplified"
        if s == "4": return 1
        elif s in ("2","3") and amp: return 1
        elif s in ("1","2","3","4S"): return 0
        return np.nan

    meta.loc[idx,"high_risk_merged"] = [_risk(s,m) for s,m in zip(stages,mycns)]
    return meta


def run():
    print("="*60, "\nPHASE 1 — Data Integration & QC\n", "="*60)
    expr_dict    = {c: load_expression(DATA(f"expr_{c}.csv")) for c in COHORTS}
    common_genes = intersect_genes(*expr_dict.values())
    print(f"Common genes: {len(common_genes):,}")

    expr_merged = merge_cohorts(expr_dict, common_genes)
    meta = assign_risk_gse120559(load_metadata(DATA("meta_clinical.csv")))

    pca_batch_plot(expr_merged, meta, save_path=FIG("PCA_pre_correction.png"))
    expr_merged.to_csv(DATA(f"integrated_expression_{expr_merged.shape[1]}x{len(common_genes)}.csv"))
    meta.to_csv(DATA("meta_clinical_unified.csv"))
    print("Phase 1 complete.\n")
    return expr_merged, meta


if __name__ == "__main__": run()
