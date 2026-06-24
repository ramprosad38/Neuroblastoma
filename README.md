# Multi-Cohort Neuroblastoma ML Pipeline

A 9-phase bioinformatics pipeline for neuroblastoma risk classification, survival analysis, and drug repurposing.

## Cohorts
- GSE49710 (n=498), GSE73517 (n=105), GSE120559 (n=208)
- External validation: TARGET-NBL (n=984)

## Key Results
- Consensus DEGs: 5,495 (IVW meta-analysis)
- XGBoost AUC: 0.9609 (pooled 5-CV), 0.9509 (LOCO-CV)
- Top biomarker: PPEF1 (padj=3.1e-06)

## Quick Start
```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Pipeline Phases
| Phase | Module | Description |
|-------|--------|-------------|
| 1 | 01_data_integration.py | Multi-cohort merging & QC |
| 2 | 02_differential_expression.py | IVW meta-DEA |
| 3 | 03_survival_analysis.py | Kaplan-Meier curves |
| 4 | 04_cox_regression.py | Cox PH regression |
| 5 | 05_subgroup_analysis.py | Clinical subgroups |
| 6 | 06_ml_classification.py | ML risk classifier |
| 7 | 07_external_validation.py | TARGET-NBL validation |
| 8 | 08_pathway_drug.py | GSEA & drug repurposing |
| 9 | 09_manuscript_tables.py | Publication tables |

## Citation
> Ram Prosad Sarker et al. (2026). Multi-cohort neuroblastoma risk prediction pipeline. *In preparation.*
