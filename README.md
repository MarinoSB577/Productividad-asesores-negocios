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
determined by the behavior of individual field advisors. An advisor who manages
too many groups, operates with high interest rates, or shows low client
retention generates systematically higher delinquency rates.

Identifying these risk profiles **before** deterioration materializes enables
proactive interventions: targeted coaching, portfolio rebalancing, or adjusted
supervision frequency.

This project builds a multiclass classification system that segments advisors
into four risk quartiles (Q1_LOW to Q4_HIGH) using purely operational variables
— no contemporaneous delinquency data — making it usable as an early warning tool.

## What Makes This Methodologically Rigorous

### Audit-first approach
The original 2020 IPN thesis model was systematically audited before any code
was written. Three critical errors were identified and corrected:

| Error | Original model | This implementation |
|-------|---------------|---------------------|
| Data leakage | Random row split | Temporal split + advisor-level split |
| Target leakage | ATRASO and PROVISION included as features | Excluded — they are components of the target formula |
| Class collapse | Accuracy on imbalanced classes | Macro F1 with `is_unbalance=True` |

The leakage detection story is worth noting: the Logistic Regression baseline
achieved macro F1 = 0.92, which triggered an immediate audit. After removing
the leaking variables, it dropped to 0.57 — a realistic baseline for this problem.

### Fairness analysis included
Demographic variables (gender, age) were analyzed for disparate impact before
inclusion in the model. Neither showed significant disparity (gap < 10pp),
but the analysis is documented and must be re-run on every model update.

### Synthetic demo data
The dashboard runs on statistically representative synthetic data. No real
advisor data is exposed in this repository.

## Key Results

| Model | CV Macro F1 | Test Macro F1 | Cohen's Kappa | AUC OvR |
|-------|------------|--------------|---------------|---------|
| Logistic Regression (baseline) | 0.586 ± 0.016 | 0.570 | 0.425 | 0.808 |
| **LightGBM (final model)** | **0.614 ± 0.016** | **0.614** | **0.481** | **0.838** |
| XGBoost | — | 0.570 | 0.428 | 0.830 |

> **Decision metric:** Macro F1 — treats all risk classes equally regardless of frequency.
> CV ≈ Test F1 confirms no overfitting despite the small dataset (~2,004 advisors).

### Performance by class

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Q1_BAJO (low risk) | 0.76 | 0.70 | 0.73 |
| Q2_MEDIO_BAJO | 0.47 | 0.54 | 0.50 |
| Q3_MEDIO_ALTO | 0.48 | 0.46 | 0.47 |
| Q4_ALTO (high risk) | 0.77 | 0.75 | 0.76 |

The intermediate quartiles have a genuinely fuzzy boundary — this is expected
in ordinal classification and is documented as a known limitation.

### Top predictive features (SHAP)

| Feature | SHAP importance | Business interpretation |
|---------|----------------|------------------------|
| CLIENTES_PRESTAMO | 0.976 | Active clients with loans — key portfolio health signal |
| PRESTAMO | 0.329 | Average loan amount — lower amounts correlate with higher risk |
| TASA_PROM | 0.317 | Average interest rate — higher rates predict deterioration |
| CLIENTES_NUEVOS | 0.257 | New client acquisition — low acquisition signals portfolio contraction |

### Operational decision rules (4 variables, real units)

| Profile | CLIENTES_PRESTAMO | TASA_PROM | CLIENTES_NUEVOS | PRESTAMO | Risk |
|---------|-----------------|-----------|-----------------|---------|------|
| Consolidated active advisor | > 1.9 | < 220 | > 0.3 | > $11,400 | Q1_BAJO |
| Advisor in contraction | < 0.5 | > 237 | < 0.1 | < $2,900 | Q4_ALTO |

### Fairness results

| Variable | Max gap between groups | Threshold | Result |
|----------|----------------------|-----------|--------|
| SEXO (gender) | 2.3pp | 10pp | No significant disparity |
| EDAD (age groups) | 6.8pp | 10pp | No significant disparity |

## Repository Structure

```
advisor-risk-profiler/
├── notebooks/
│   ├── 01_eda_exploratory.ipynb       # Data quality audit
│   ├── 02_target_engineering.ipynb    # Target construction + temporal split
│   ├── 03_feature_engineering.ipynb   # Pipeline: imputation, encoding, scaling
│   ├── 04_baseline_model.ipynb        # Logistic Regression baseline + MLflow
│   ├── 05_lgbm_xgb_training.ipynb     # LightGBM + XGBoost with Optuna
│   ├── 05b_feature_selection.ipynb    # Feature selection curve
│   ├── 06_evaluation_fairness_shap.ipynb  # SHAP + fairness analysis
│   └── 07a_synthetic_data.ipynb       # Synthetic data generator
├── src/advisor_risk/                  # Installable Python package
├── app/
│   ├── streamlit_app.py               # Dashboard entry point
│   └── pages/
│       ├── 1_Comparacion_Modelos.py   # Model comparison
│       ├── 2_SHAP_Explorer.py         # SHAP visualizations
│       └── 3_Perfil_Asesor.py         # Real-time risk prediction
├── data/
│   └── sample/                        # Synthetic data (safe to share)
│       └── advisors_sample_anon.parquet
├── reports/figures/                   # Generated visualizations
├── METHODOLOGY.md                     # All design decisions documented
└── pyproject.toml                     # Dependencies (uv)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/MarinoSB577/Productividad-asesores-negocios.git
cd Productividad-asesores-negocios

# Install with uv (recommended)
uv sync --extra dev

# Run the dashboard (uses synthetic data — no real data required)
uv run streamlit run app/streamlit_app.py
```

## Reproducing the Analysis

All notebooks run sequentially on the synthetic dataset.
To run on real data, place your files in `data/raw/` following the schema
in `METHODOLOGY.md`.

```bash
# Run notebooks sequentially
uv run jupyter notebook
```

MLflow experiment tracking:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Tech Stack

Python 3.11 · LightGBM · XGBoost · scikit-learn · SHAP · MLflow · DuckDB ·
Streamlit · Optuna · pandas · scipy · joblib · pytest

## Methodological Audit Findings

Three issues from the original 2020 thesis corrected in this implementation:

1. **Target leakage:** ATRASO and PROVISION variables are components of the
   TDNC formula used to build the target. Including them caused the baseline
   LR to reach F1=0.92 — an impossible result for this problem.
2. **Temporal data leakage:** Row-level random split allowed future
   observations to inform training. Corrected with temporal split (Target A)
   and advisor-level split (Target B).
3. **Class collapse and wrong metric:** The original model effectively
   classified 2 of 8 target classes using accuracy. Corrected with macro F1
   and `is_unbalance=True` in LightGBM.

## Business Impact

- Advisors in Q4_ALTO carry the highest structural portfolio risk — early
  identification enables proactive supervision before deterioration materializes
- Model correctly identifies Q4_ALTO advisors 75% of the time (recall=0.75)
  with 77% precision — strong enough for triage prioritization
- No significant fairness disparity by gender or age group (both gaps < 10pp)
- Intermediate quartiles (Q2/Q3) have a fuzzy boundary (F1~0.48) —
  the model should not be used as the sole criterion for these groups

## Known Limitations

- Dataset: ~2,004 advisors from a single institution (2016-2019)
- Product-specific: Crédito Grupal only — not validated for other products
- Temporal scope: requires retraining for periods after January 2019
- Fairness variables included: legal review required before HR decisions
- Must not be used for automatic dismissal or disciplinary decisions
