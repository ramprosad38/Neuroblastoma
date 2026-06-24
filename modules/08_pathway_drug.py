"""
Phase 8 — Pathway Enrichment & Drug Repurposing
- GSEA (Hallmark + KEGG) via gseapy
- ORA (KEGG + GO-BP) via gseapy
- Drug repurposing via L1000CDS2
- Pathway crosstalk analysis
- Outputs: GSEA/ORA CSVs, drug_repurposing_L1000CDS2.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gseapy as gp
import requests
from config import DATA, OUT, FIG, DPI


def run_gsea(preranked_rnk: pd.DataFrame, gene_set: str, save_prefix: str):
    """Run GSEA preranked and return significant results."""
    res = gp.prerank(
        rnk=preranked_rnk, gene_sets=gene_set,
        min_size=15, max_size=500,
        permutation_num=1000, seed=42,
        outdir=None, verbose=False
    )
    df = res.res2d[res.res2d["FDR q-val"] < 0.25].sort_values("NES", ascending=False)
    df.to_csv(OUT(f"GSEA_{save_prefix}_results.csv"))
    print(f"GSEA {gene_set}: {len(df)} significant pathways")
    return df


def run_ora(gene_list: list, gene_set: str, save_prefix: str):
    enr = gp.enrichr(
        gene_list=gene_list, gene_sets=gene_set,
        organism="human", outdir=None
    )
    df  = enr.results[enr.results["Adjusted P-value"] < 0.05]
    df.to_csv(OUT(f"ORA_{save_prefix}_results.csv"), index=False)
    print(f"ORA {gene_set}: {len(df)} significant terms")
    return df


def query_l1000cds2(up_genes: list, dn_genes: list) -> pd.DataFrame:
    """Query L1000CDS2 for drugs that reverse the signature."""
    payload = {
        "upGenes": up_genes[:200], "dnGenes": dn_genes[:200],
        "aggravate": False, "share": False
    }
    try:
        resp = requests.post(
            "https://maayanlab.cloud/L1000CDS2/query",
            json=payload,
            headers={"Content-Type": "application/json"}, timeout=60
        )
        data = resp.json()
        entries = data.get("result", {}).get("combinations", [])
        drugs = []
        for e in entries[:30]:
            drug_name = e.get("sig", {}).get("pert_desc", e.get("pert_desc","Unknown"))
            drugs.append({
                "drug": drug_name,
                "cell_line": e.get("cell_id",""),
                "score": e.get("score", np.nan),
                "direction": "reverses"
            })
        df = pd.DataFrame(drugs)
        df.to_csv(OUT("drug_repurposing_L1000CDS2.csv"), index=False)
        print(f"Drug repurposing: {len(df)} candidates found")
        return df
    except Exception as ex:
        print(f"L1000CDS2 query failed: {ex}")
        return pd.DataFrame()


def gsea_dotplot(gsea_df, top_n=20, title="GSEA Hallmark", save_path=None):
    top = gsea_df.head(top_n).copy()
    top["-log10FDR"] = -np.log10(top["FDR q-val"].replace(0, 1e-10))
    fig, ax = plt.subplots(figsize=(8, top_n * 0.4 + 1))
    sc = ax.scatter(top["NES"], range(len(top)),
                    c=top["-log10FDR"], cmap="Reds",
                    s=top["Tag %"].str.rstrip("%").astype(float) * 5,
                    vmin=0, vmax=top["-log10FDR"].max())
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top["Term"], fontsize=8)
    ax.axvline(0, ls="--", color="grey", lw=0.8)
    ax.set_xlabel("NES"); ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="-log₁₀ FDR")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI); print(f"Saved {save_path}")
    plt.close()


def drug_barplot(drug_df, save_path=None):
    top = drug_df.head(15).sort_values("score")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["drug"], top["score"], color="#1E88E5")
    ax.set_xlabel("Reversal Score"); ax.set_title("Top Drug Repurposing Candidates")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI); print(f"Saved {save_path}")
    plt.close()


def run():
    print("="*60, "\nPHASE 8 — Pathway & Drug Repurposing\n", "="*60)

    meta_deg = pd.read_csv(OUT("meta_DEG_all_genes.csv"), index_col=0)
    rnk      = meta_deg[["Z"]].sort_values("Z", ascending=False)
    up200    = rnk.head(200).index.tolist()
    dn200    = rnk.tail(200).index.tolist()
    rnk.to_csv(OUT("meta_DEG_preranked.rnk"), sep="\t", header=False)

    hallmark = run_gsea(rnk, "MSigDB_Hallmark_2020", "hallmark")
    kegg     = run_gsea(rnk, "KEGG_2021_Human",      "KEGG")
    ora_kegg = run_ora(up200, "KEGG_2021_Human", "KEGG")
    ora_gobp = run_ora(up200, "GO_Biological_Process_2021", "GOBP")

    gsea_dotplot(hallmark, title="GSEA Hallmark", save_path=FIG("GSEA_hallmark_dotplot.png"))

    drug_df = query_l1000cds2(up200, dn200)
    if len(drug_df): drug_barplot(drug_df, save_path=FIG("drug_repurposing_barplot.png"))

    print("Phase 8 complete.\n")
    return hallmark, kegg, drug_df


if __name__ == "__main__": run()
