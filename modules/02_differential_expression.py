"""
Phase 2 — IVW Meta-Analysis Differential Expression
- Per-cohort Welch t-test (high vs low risk)
- Inverse-Variance Weighted meta log-FC + FDR correction
- Volcano plot + top-50 heatmap
- Outputs: meta_DEG_all_genes.csv, consensus_DEG_filtered.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA, OUT, FIG, COHORTS, KEY_GENES, DPI


def cohort_deg(expr, meta, cohort):
    idx  = meta[meta["cohort"] == cohort].index
    labs = meta.loc[idx,"high_risk_merged"].dropna()
    hi, lo = labs[labs==1].index, labs[labs==0].index
    e    = expr[idx].T
    rows = []
    for gene in e.columns:
        a, b = e.loc[hi,gene].dropna().values, e.loc[lo,gene].dropna().values
        if len(a)<3 or len(b)<3: continue
        t, p  = stats.ttest_ind(a, b, equal_var=False)
        se    = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
        rows.append({"gene":gene,"logFC":a.mean()-b.mean(),
                     "SE":se,"t":t,"pval":p,"cohort":cohort})
    return pd.DataFrame(rows).set_index("gene")


def ivw_meta(dfs):
    all_genes = sorted(set.intersection(*[set(d.index) for d in dfs]))
    rows = []
    for g in all_genes:
        lfc = np.array([d.loc[g,"logFC"] for d in dfs])
        se  = np.array([d.loc[g,"SE"]    for d in dfs])
        w   = 1/se**2
        m_lfc = np.sum(w*lfc)/np.sum(w)
        m_se  = np.sqrt(1/np.sum(w))
        z     = m_lfc/m_se
        p     = 2*stats.norm.sf(np.abs(z))
        rows.append({"gene":g,"meta_logFC":m_lfc,"meta_SE":m_se,"Z":z,"pval":p})
    df = pd.DataFrame(rows).set_index("gene")
    _, padj, _, _ = multipletests(df["pval"], method="fdr_bh")
    df["padj"] = padj
    return df.sort_values("padj")


def volcano_plot(deg, save_path=None, fc_thr=0.3, padj_thr=0.05, key_genes=None):
    neg_log = -np.log10(deg["padj"].replace(0, 1e-300))
    sig_up  = (deg["padj"]<padj_thr)&(deg["meta_logFC"]> fc_thr)
    sig_dn  = (deg["padj"]<padj_thr)&(deg["meta_logFC"]<-fc_thr)
    color   = np.where(sig_up,"#E53935",np.where(sig_dn,"#1E88E5","lightgrey"))
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(deg["meta_logFC"], neg_log, c=color, s=5, alpha=0.6)
    ax.axhline(-np.log10(padj_thr), ls="--", lw=0.8, color="grey")
    ax.axvline(fc_thr, ls="--", lw=0.8, color="grey")
    ax.axvline(-fc_thr, ls="--", lw=0.8, color="grey")
    if key_genes:
        for g in key_genes:
            if g in deg.index:
                ax.annotate(g, (deg.loc[g,"meta_logFC"], -np.log10(deg.loc[g,"padj"])),
                            fontsize=7, arrowprops=dict(arrowstyle="-",lw=0.5))
    ax.set_xlabel("IVW meta log₂FC (high vs low risk)")
    ax.set_ylabel("-log₁₀(padj)")
    ax.set_title("Volcano — IVW Meta-DEA (3 cohorts, n=811)")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI); print(f"Saved {save_path}")
    plt.close()


def heatmap_top50(expr, meta, top_genes, save_path=None):
    samps   = meta["high_risk_merged"].dropna().index
    lab_ord = meta.loc[samps].sort_values("high_risk_merged")
    e       = expr.loc[top_genes, lab_ord.index]
    e_z     = ((e.T - e.T.mean())/e.T.std()).T
    col_colors = lab_ord["high_risk_merged"].map({1:"#E53935",0:"#1E88E5"})
    g = sns.clustermap(e_z, col_cluster=False, row_cluster=True,
                       col_colors=col_colors, yticklabels=True,
                       cmap="RdBu_r", vmin=-2, vmax=2, figsize=(12,10))
    plt.suptitle("Heatmap — Top 50 Consensus DEGs", y=1.01, fontsize=12)
    if save_path: plt.savefig(save_path,dpi=DPI,bbox_inches="tight"); print(f"Saved {save_path}")
    plt.close()


def run(expr=None, meta=None):
    print("="*60, "\nPHASE 2 — IVW Meta-Analysis DEA\n", "="*60)
    if expr is None:
        expr = pd.read_csv(DATA("integrated_expression_811x19860.csv"), index_col=0)
        meta = pd.read_csv(DATA("meta_clinical_unified.csv"), index_col=0)

    dfs      = [cohort_deg(expr, meta, c) for c in COHORTS]
    meta_deg = ivw_meta(dfs)
    meta_deg.to_csv(OUT("meta_DEG_all_genes.csv"))

    consensus = meta_deg[(meta_deg["padj"]<0.05)&(meta_deg["meta_logFC"].abs()>0.3)]
    consensus.to_csv(OUT("consensus_DEG_filtered.csv"))
    print(f"Consensus DEGs: {len(consensus):,}  "
          f"({(consensus['meta_logFC']>0).sum()} up / "
          f"{(consensus['meta_logFC']<0).sum()} down)")

    volcano_plot(meta_deg, FIG("volcano_meta_DEA.png"), key_genes=KEY_GENES)
    heatmap_top50(expr, meta, consensus.head(50).index.tolist(),
                  FIG("heatmap_top50DEGs.png"))
    print("Phase 2 complete.\n")
    return meta_deg, consensus


if __name__ == "__main__": run()
