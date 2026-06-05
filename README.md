# Advisor Risk Profiler — Microfinance Portfolio Deterioration Model

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-9c27b0)](https://lightgbm.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-3.13-0194E2?logo=mlflow)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit)](https://streamlit.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.5-FEC52E)](https://duckdb.org)
[![Optuna](https://img.shields.io/badge/Optuna-4.9-blue)](https://optuna.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Business Problem

In Mexican microfinance institutions, loan portfolio quality is ultimately
determined by the behavior of individual field advisors. An advisor whose
client groups are not renewing their loans on schedule signals accumulated
delinquency before it appears in official reports.

This project builds a multiclass classification system that segments advisors
into three risk levels (LOW / MEDIUM / HIGH) using purely operational variables
— no contemporaneous delinquency data — making it a genuine early warning tool.

## What Makes This Methodologically Rigorous

### Audit-first approach
The original 2020 IPN thesis model was systematically audited before any code
was written. Three critical errors were identified and corrected:

| Error | Original model | This implementation |
|-------|---------------|---------------------|
| Data leakage | Random row split | Temporal split + advisor-level split |
| Target leakage | ATRASO and PROVISION included as features | Excluded — components of the target formula |
| Class collapse | Accuracy on imbalanced 8-class target | Macro F1 with `is_unbalance=True` |

The leakage detection story is worth noting: the Logistic Regression baseline
initially achieved macro F1 = 0.92, which triggered an immediate audit. After
removing leaking variables, it dropped to a realistic 0.57 (4-class version)
and 0.71 (final 3-class version).

### Variable engineering: event-based disbursement features
Original variables (CLIENTES_PRESTAMO, PRESTAMO, CLIENTES_NUEVOS) were
computed as simple averages including zero-value weeks, which confused two
distinct signals: group size and disbursement frequency.

Corrected approach — computed only over disbursement event weeks:

| New variable | Calculation | Business interpretation |
|-------------|-------------|------------------------|
| `TASA_DESEMBOLSO` | event rows / total rows | Portfolio renewal activity rate |
| `CLIENTES_PRESTAMO_EVENTO` | mean(CLIENTES_PRESTAMO) where > 0 | Avg group size at disbursement |
| `MONTO_DESEMBOLSO_EVENTO` | mean(PRESTAMO) where > 0 | Avg disbursement amount per group |
| `CLIENTES_NUEVOS_EVENTO` | mean(CLIENTES_NUEVOS) where > 0 | Avg new clients per disbursement |

`TASA_DESEMBOLSO` became the most important feature (SHAP = 0.935), with a
clear business interpretation: standard group loans run 12 weeks, so a healthy
advisor disburses in ~8% of their group-weeks. A HIGH-risk advisor disburses
in only ~4% — their groups are stuck in delinquency and cannot renew.

### 3-class target: evidence-based decision
Initial 4-class model (Q1-Q4 quartiles) showed F1 = 0.50/0.47 for intermediate
classes — the boundary between Q2 and Q3 was genuinely indistinguishable with
available features. Collapsing to 3 classes improved macro F1 by +11.7pp.

### Fairness analysis included
Demographic variables (gender, age) were analyzed for disparate impact.
Both are below the 10pp alert threshold, but proximity is documented:

| Variable | Gap | Threshold | Status |
|----------|-----|-----------|--------|
| Gender | 8.1pp | 10pp | Monitor |
| Age group | 9.5pp | 10pp | Monitor |

### Synthetic demo data
The dashboard runs on statistically representative synthetic data. No real
advisor data is exposed in this repository.

## Key Results

| Model | CV Macro F1 | Test Macro F1 | Cohen Kappa | AUC OvR |
|-------|------------|--------------|-------------|---------|
| Logistic Regression (baseline) | 0.666 ± 0.038 | 0.711 | 0.566 | 0.855 |
| **LightGBM (final model)** | **0.706 ± 0.016** | **0.717** | **0.577** | **0.881** |
| XGBoost | 0.710 ± — | 0.711 | 0.570 | 0.883 |

> **Note on LightGBM vs baseline gap (+0.6pp):** The small improvement in
> macro F1 is expected — the 3-class problem with operational variables is
> genuinely difficult. LightGBM's advantage lies in AUC OvR (0.881 vs 0.855)
> and lower variance between CV folds (0.016 vs 0.038), making it more
> reliable for probability-based prioritization.

### Performance by class

| Class | Precision | Recall | F1 | Business meaning |
|-------|-----------|--------|----|-----------------|
| BAJO (low risk) | 0.74 | 0.75 | 0.74 | Active portfolio, renewing on schedule |
| MEDIO (medium) | 0.63 | 0.59 | 0.61 | Mixed signals, standard supervision |
| ALTO (high risk) | 0.78 | 0.82 | 0.80 | Stalled groups, accumulated delinquency |

### Top predictive features (SHAP)

| Feature | SHAP importance | Business interpretation |
|---------|----------------|------------------------|
| TASA_DESEMBOLSO | 0.935 | Portfolio renewal rate — dominant signal |
| TASA_PROM | 0.223 | Higher rates correlate with riskier clients |
| N_SEMANAS_OBS | 0.183 | Advisor tenure in the observation period |
| INCREMENTO_CARTERA | 0.136 | Weekly portfolio growth/contraction |
| CLIENTES | 0.106 | Active client count |

## Repository Structure

```
advisor-risk-profiler/
├── notebooks/
│   ├── 01_eda_exploratory.ipynb         # Data quality audit (1.2M records)
│   ├── 02_target_engineering.ipynb      # TDNC target + event variables (v3)
│   ├── 03_feature_engineering.ipynb     # Pipeline: imputation, encoding, scaling
│   ├── 04_baseline_model.ipynb          # Logistic Regression baseline + MLflow
│   ├── 05_lgbm_xgb_training.ipynb       # LightGBM + XGBoost with Optuna
│   ├── 05b_feature_selection.ipynb      # Feature selection curve analysis
│   ├── 06_evaluation_fairness_shap.ipynb  # SHAP + fairness analysis
│   └── 07a_synthetic_data.ipynb         # Synthetic data generator
├── app/
│   ├── streamlit_app.py                 # Dashboard entry point
│   └── pages/
│       ├── 1_Comparacion_Modelos.py     # Model comparison
│       ├── 2_SHAP_Explorer.py           # SHAP visualizations
│       └── 3_Perfil_Asesor.py           # Real-time risk prediction
├── data/
│   └── sample/                          # Synthetic data (safe to share)
├── reports/
│   ├── figures/                         # Generated visualizations
│   └── reporte_ejecutivo.md             # Executive report (Spanish)
├── METHODOLOGY.md                       # All design decisions documented
└── pyproject.toml                       # Dependencies (uv)
```

## Installation

```bash
git clone https://github.com/MarinoSB577/Productividad-asesores-negocios.git
cd Productividad-asesores-negocios

# Install with uv
uv sync --extra dev

# Run the dashboard (uses synthetic data)
uv run streamlit run app/streamlit_app.py

# View MLflow experiments
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Methodological Audit Findings

Five issues from the original 2020 thesis corrected in this implementation:

1. **Target leakage:** ATRASO and PROVISION are components of the TDNC
   formula. Including them caused LR baseline F1 = 0.92 — impossible for
   this problem. Removed.
2. **Temporal data leakage:** Row-level random split corrected with
   temporal split (Target A) and advisor-level split (Target B).
3. **Class collapse:** Original model classified 2 of 8 classes using
   accuracy. Corrected with macro F1 and `is_unbalance=True`.
4. **Event variable distortion:** Disbursement variables averaged over
   all weeks (including zero weeks) mixed two signals. Corrected by
   computing only over disbursement event weeks.
5. **Target granularity:** 4 quartiles with indistinguishable intermediate
   classes (F1 = 0.50/0.47). Corrected by collapsing to 3 classes
   (+11.7pp improvement in macro F1).

## Business Impact

- Advisors classified as ALTO have groups renewing every ~28 group-weeks
  vs ~6 for BAJO — a 4.7x difference in portfolio renewal activity
- Model correctly identifies ALTO advisors 82% of the time (recall = 0.82)
  with 78% precision — strong enough for supervision prioritization
- Fairness gaps by gender (8.1pp) and age (9.5pp) are below the 10pp
  alert threshold but warrant monitoring on model updates
- MEDIO class (F1 = 0.61) has a genuinely fuzzy boundary — human judgment
  from coordinators should complement model output for this group

## Known Limitations

- Dataset: ~2,004 advisors from a single institution (2016-2019)
- Product-specific: Credito Grupal only
- Temporal scope: requires retraining for periods after January 2019
- Fairness variables included: legal review required before HR decisions
- Must not be used for automatic dismissal or disciplinary decisions

## Tech Stack

Python 3.11 · LightGBM · XGBoost · scikit-learn · SHAP · MLflow · DuckDB ·
Streamlit · Optuna · pandas · scipy · joblib · pytest
