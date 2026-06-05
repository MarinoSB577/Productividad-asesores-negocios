# Clasificación de Riesgo de Cartera — Asesores de Microfinanzas

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-9c27b0)](https://lightgbm.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-3.13-0194E2?logo=mlflow)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit)](https://streamlit.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.5-FEC52E)](https://duckdb.org)
[![Optuna](https://img.shields.io/badge/Optuna-4.9-blue)](https://optuna.org)
[![Demo en vivo](https://img.shields.io/badge/Demo-Streamlit_Cloud-FF4B4B?logo=streamlit)](https://app-deterioro-cartera-asesores-marin.streamlit.app)

## El problema

En una institución microfinanciera mexicana, la calidad de la cartera depende en gran medida del comportamiento de los asesores de campo. Un asesor cuyos grupos no están renovando su crédito a tiempo está acumulando mora semana a semana — pero esa señal puede pasar desapercibida en los reportes tradicionales hasta que el daño ya está hecho.

Este proyecto construye un sistema de clasificación que segmenta a los asesores en tres niveles de riesgo de deterioro de cartera: **BAJO, MEDIO y ALTO**. La clave del modelo no está en medir el atraso que ya ocurrió, sino en detectar las señales operacionales que lo preceden.

**Demo en vivo:** [app-deterioro-cartera-asesores-marin.streamlit.app](https://app-deterioro-cartera-asesores-marin.streamlit.app)

---

## El hallazgo central

La variable más importante del modelo no existía en los datos originales. Se construyó durante el análisis:

> **`TASA_DESEMBOLSO` = semanas con desembolso activo / total semanas observadas**

Un crédito grupal estándar dura 12 semanas. Un asesor con cartera sana tiene grupos que terminan su ciclo y renuevan de inmediato. Uno con mora acumulada tiene grupos que no califican para renovación — y por eso su tasa de desembolso cae dramáticamente:

| Nivel de riesgo | `TASA_DESEMBOLSO` promedio | Cada cuántas semanas renueva |
|----------------|--------------------------|------------------------------|
| BAJO | 15.9% | ~1 de cada 6 semanas |
| MEDIO | 6.6% | ~1 de cada 15 semanas |
| ALTO | 3.6% | ~1 de cada 28 semanas |

Esta variable terminó con una importancia SHAP de **0.935** — más de 4 veces la siguiente variable más importante.

---

## Variable objetivo

La variable objetivo es la **Tasa de Deterioro Neto de Cartera (TDNC)** por asesor, calculada colapsando el dataset de nivel semanal a nivel asesor:

> **TDNC = promedio(ATRASO) / promedio(CARTERA_HOY)**

Los asesores se segmentan en tres clases usando terciles de TDNC:
- **BAJO:** TDNC promedio = 0.013 — cartera prácticamente sin atraso
- **MEDIO:** TDNC promedio = 0.095 — atraso entre 5% y 15% de la cartera
- **ALTO:** TDNC promedio = 0.372 — atraso superior al 20% de la cartera

---

## Por qué este proyecto es metodológicamente sólido

### Auditoría antes de modelar

La tesis original de 2020 fue auditada sistemáticamente antes de escribir una sola línea de código. Se identificaron y corrigieron cinco problemas críticos:

| Problema | Modelo original | Este proyecto |
|----------|----------------|---------------|
| Leakage temporal | Split aleatorio por fila | Split temporal + split por asesor |
| Leakage de construcción | ATRASO y PROVISION como features | Excluidas — son componentes de la fórmula del target |
| Colapso de clases | Accuracy en 8 clases desbalanceadas | Macro F1 con `is_unbalance=True` |
| Variables de evento distorsionadas | Promedios incluyendo semanas sin desembolso | Calculadas solo sobre semanas con evento |
| Granularidad del target | 4 cuartiles con frontera indistinguible | 3 clases (+11.7pp de macro F1) |

La señal de detección del leakage fue un baseline de Logistic Regression con F1 = 0.92 — un resultado imposible que obligó a revisar el pipeline completo.

### Variables construidas desde cero

Las variables originales de desembolso (`CLIENTES_PRESTAMO`, `PRESTAMO`, etc.) estaban promediadas sobre todas las semanas, incluyendo las semanas sin evento. Eso mezcla dos señales distintas: el tamaño del grupo y la frecuencia de renovación. Se reemplazaron por cuatro variables calculadas correctamente:

| Variable construida | Cálculo | Interpretación |
|--------------------|---------|----------------|
| `TASA_DESEMBOLSO` | filas con evento / total filas | Actividad de renovación de cartera |
| `CLIENTES_PRESTAMO_EVENTO` | media de `CLIENTES_PRESTAMO` donde > 0 | Tamaño promedio del grupo al desembolso |
| `MONTO_DESEMBOLSO_EVENTO` | media de `PRESTAMO` donde > 0 | Monto promedio desembolsado |
| `CLIENTES_NUEVOS_EVENTO` | media de `CLIENTES_NUEVOS` donde > 0 | Clientes nuevos promedio en desembolsos |

### Análisis de equidad algorítmica

Las variables sociodemográficas (género, edad) se analizaron para detectar impacto dispar antes de incluirlas en el modelo. Ambos gaps están por debajo del umbral de alerta de 10pp, pero su proximidad está documentada como riesgo a monitorear.

---

## Resultados

| Modelo | CV Macro F1 | Test Macro F1 | AUC OvR |
|--------|------------|--------------|---------|
| Logistic Regression (baseline) | 0.666 ± 0.038 | 0.711 | 0.855 |
| **LightGBM (modelo final)** | **0.706 ± 0.016** | **0.717** | **0.881** |
| XGBoost | 0.710 | 0.711 | 0.883 |

La mejora de LightGBM sobre el baseline es pequeña en macro F1 (+0.6pp), pero su ventaja en AUC OvR (0.881 vs 0.855) y menor varianza entre folds lo hacen más confiable para priorización basada en probabilidades.

---

## Estructura del repositorio

```
productividad-asesores-negocios/
├── notebooks/
│   ├── 01_eda_exploratory.ipynb         # Auditoría de calidad (1.2M registros)
│   ├── 02_target_engineering.ipynb      # Construcción de TDNC y variables de evento
│   ├── 03_feature_engineering.ipynb     # Pipeline de preprocesamiento
│   ├── 04_baseline_model.ipynb          # Regresión Logística + MLflow
│   ├── 05_lgbm_xgb_training.ipynb       # LightGBM + XGBoost con Optuna
│   ├── 05b_feature_selection.ipynb      # Curva de selección de features
│   ├── 06_evaluation_fairness_shap.ipynb  # SHAP + análisis de equidad
│   └── 07a_synthetic_data.ipynb         # Generador de datos sintéticos
├── app/
│   ├── streamlit_app.py                 # Punto de entrada del dashboard
│   └── pages/
│       ├── 1_Comparacion_Modelos.py     # Comparativa de los 3 modelos
│       ├── 2_SHAP_Explorer.py           # Visualizaciones SHAP interactivas
│       └── 3_Perfil_Asesor.py           # Predicción en tiempo real
├── data/
│   ├── processed/                       # Modelos entrenados (.joblib)
│   └── sample/                          # Datos sintéticos anonimizados
├── reports/
│   ├── figures/                         # Gráficas generadas
│   └── reporte_ejecutivo.md             # Reporte ejecutivo en español
├── METHODOLOGY.md                       # Registro de decisiones de diseño
└── pyproject.toml                       # Dependencias (uv)
```

---

## Instalación y ejecución

```bash
# Clonar el repositorio
git clone https://github.com/MarinoSB577/Productividad-asesores-negocios.git
cd Productividad-asesores-negocios

# Instalar dependencias con uv
uv sync --extra dev

# Ejecutar el dashboard (usa datos sintéticos — no requiere datos reales)
uv run streamlit run app/streamlit_app.py

# Ver experimentos de MLflow
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

---

## Limitaciones conocidas

- Dataset de entrenamiento: ~2,004 asesores de una sola institución (2016-2019)
- Producto específico: Crédito Grupal — no validado para otros productos
- Scope temporal: requiere reentrenamiento para períodos posteriores a enero 2019
- Variables de equidad incluidas: revisión legal requerida antes de usar en decisiones de personal
- No debe utilizarse para decisiones automáticas de despido o sanción

---

## Stack tecnológico

Python 3.11 · LightGBM · XGBoost · scikit-learn · SHAP · MLflow · DuckDB ·
Streamlit · Optuna · pandas · scipy · joblib
