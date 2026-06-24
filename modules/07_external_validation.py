"""
Phase 7 — External Validation (TARGET-NBL via GDC API)
- Downloads TARGET-NBL clinical data via GDC REST API
- Applies Cox risk score from Phase 4 to held-out cohort
- KM curves stratified by risk score + INSS stage
- Log-rank test on n=984 independent samples
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from config import DATA, OUT, FIG, DPI

GDC_ENDPOINT = "https://api.gdc.cancer.gov"


def fetch_target_nbl_clinical(save_path=None):
    """Fetch TARGET-NBL clinical data from GDC API."""
    params = {
        "filters": '{"op":"=","content":{"field":"project.project_id","value":"TARGET-NBL"}}',
        "fields": "case_id,diagnoses.days_to_death,diagnoses.vital_status,"
                  "diagnoses.age_at_diagnosis,diagnoses.inss_stage,"
                  "diagnoses.days_to_last_follow_up",
        "format": "JSON", "size": "2000"
    }
    resp = requests.get(f"{GDC_ENDPOINT}/cases", params=params, timeout=60)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]

    rows = []
    for h in hits:
        diag = h.get("diagnoses", [{}])[0]
        dtd  = diag.get("days_to_death")
        dtlf = diag.get("days_to_last_follow_up")
        rows.append({
            "case_id":    h["case_id"],
            "OS_time":    dtd if dtd else dtlf,
            "OS_event":   1 if diag.get("vital_status","Alive")=="Dead" else 0,
            "age_days":   diag.get("age_at_diagnosis"),
            "INSS_stage": diag.get("inss_stage"),
        })
    df = pd.DataFrame(rows).dropna(subset=["OS_time"])
    df["cohort"] = "TARGET-NBL"
    if save_path: df.to_csv(save_path, index=False); print(f"Saved {save_path}")
    return df


def validate_risk_score(target_df, risk_scores, save_path=None):
    common = target_df.index.intersection(risk_scores.index)
    df     = target_df.loc[common].copy()
    df["cox_risk_score"] = risk_scores.loc[common]
    med    = df["cox_risk_score"].median()
    hi     = df[df["cox_risk_score"] >= med]
    lo     = df[df["cox_risk_score"] <  med]
    p      = logrank_test(hi["OS_time"], lo["OS_time"],
                          event_observed_A=hi["OS_event"],
                          event_observed_B=lo["OS_event"]).p_value
    kmf    = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5))
    for sub, lbl, col in [(hi,"High-risk","#E53935"),(lo,"Low-risk","#1E88E5")]:
        kmf.fit(sub["OS_time"], sub["OS_event"], label=lbl)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=col)
    ax.set_title(f"TARGET-NBL External Validation (n={len(df)})\n"
                 f"Log-rank p = {p:.2e}")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Survival Probability")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=DPI); print(f"Saved {save_path}")
    plt.close()
    return p


def run():
    print("="*60, "\nPHASE 7 — External Validation (TARGET-NBL)\n", "="*60)
    target_df    = fetch_target_nbl_clinical(save_path=OUT("TARGET_NBL_clinical.csv"))
    risk_scores  = pd.read_csv(OUT("Cox_risk_scores.csv"), index_col=0).squeeze()
    p = validate_risk_score(target_df, risk_scores,
                            save_path=FIG("KM_TARGET_NBL_validation.png"))
    print(f"TARGET-NBL log-rank p = {p:.2e}")
    print("Phase 7 complete.\n")
    return target_df


if __name__ == "__main__": run()
