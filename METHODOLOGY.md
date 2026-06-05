# METHODOLOGY.md — Decisiones de Diseño

Documento de registro formal de las decisiones metodológicas tomadas durante
el desarrollo del proyecto. Cada decisión incluye la justificación y el sprint
donde se tomó.

---

## Dataset

- **Fuente:** Tabla_Hist_Asesores — dos archivos CSV (NUEVA_CARGA_1.csv, NUEVA_CARGA_2.csv)
- **Encoding:** ISO-8859-1 (latin-1) — contiene nombres con acentos
- **Separador:** tabulador (`\t`) — confirmado por inspección directa del archivo
- **Encabezado:** ausente — nombres de columna provienen del DDL original de SQL Server
- **Período:** 2016-06-22 → 2019-01-30
- **Total de registros:** 1,218,831 filas (nivel semanal)
- **Asesores únicos:** 2,004

---

## Variable objetivo

### Decisión: Target_B como target principal (Sprint 2)

Se evaluaron dos formulaciones del target en paralelo:

| Target | Nivel | Variable | Observaciones |
|--------|-------|----------|---------------|
| Target_A | Semanal | DIAS_MORA_RANGO (8 clases) | 993,911 train / 224,920 test |
| Target_B | Asesor | Cuartiles TDNC (4 clases) | 1,603 train / 401 test |

**TDNC = mean(ATRASO) / mean(CARTERA_HOY)** por asesor, colapsando el dataset
de nivel semanal a nivel asesor.

**Justificación de elección de Target_B:** responde directamente la pregunta
de negocio de la tesis original (caracterización de asesores por riesgo),
tiene clases balanceadas (501 asesores por cuartil) y el split por asesor
garantiza ausencia de leakage.

### Split temporal (Target_A — referencia)
- Corte: percentil 80 de fechas únicas de HOY = 2018-07-18
- Verificación: `max(train) < min(test)` ✓

### Split por asesor (Target_B — principal)
- Estratificado por TARGET_B, test_size=0.20, random_state=42
- Verificación: ningún CODIGO_ASESOR en train Y test simultáneamente ✓

---

## Variables excluidas del modelo

### Leakage contemporáneo (Sprint 1 + Sprint 3)
| Variable | Razón |
|----------|-------|
| PASE1, PASE15, PASE30 | Medidas en el mismo corte que el target |
| DIAS_MORA | Base numérica del target |
| DIAS_MORA_RANGO | Es el Target_A |

### Leakage de construcción del target (detectado en Sprint 4)
Las siguientes variables tienen correlación > 0.28 con TDNC porque son
los componentes directos de la fórmula:

| Variable | Correlación con TDNC | Razón de exclusión |
|----------|---------------------|-------------------|
| ATRASO | 0.851 | Numerador de TDNC |
| PROVISION_HOY | 0.821 | Función directa del atraso |
| PROVISION_SEMANT | 0.813 | Función directa del atraso |
| GASTO_PROVISION_SEMANAL | 0.377 | Derivada del atraso |
| CARTERA_HOY | 0.343 | Denominador de TDNC |
| CARTERA_SEMANT | 0.280 | Denominador de TDNC |

**Señal de detección:** baseline de Logistic Regression alcanzó macro F1 = 0.9175,
valor que indicó leakage y no rendimiento real del modelo.

### Variables operacionales excluidas (justificación de negocio)
| Variable | Razón |
|----------|-------|
| SEMANA_ACTUAL_RANGO | Consecuencia del deterioro, no causa |
| ASESOR, NOMBRE, COORDINADOR | Identificadores textuales sin valor predictivo |
| NUMERO_EMPLEADO | Redundante con CODIGO_ASESOR |
| FECHA_ALTA, FECHA_BAJA, FECHA_NACIMIENTO | Alta cardinalidad sin valor predictivo en formato fecha |

---

## Pipeline de preprocesamiento (Sprint 3)

- **Imputación numérica:** mediana (robusta a outliers)
- **Imputación categórica:** constante 'SD' (preserva ausencia como categoría)
- **Encoding categórico:** OrdinalEncoder con unknown_value=-1
- **Escalado:** StandardScaler (requerido para Logistic Regression baseline)
- **Features finales:** 22 (10 numéricas + 12 categóricas)

### Correcciones de calidad previas al pipeline
| Variable | Problema | Corrección |
|----------|----------|-----------|
| TIPO_BAJA | String vacío para asesores activos | Imputar como 'ACTIVO' |
| EXP_MICROFINANZAS | String vacío sin dato | Imputar como 'SD' |
| NIVEL_ESTUDIOS | Inconsistencia de mayúsculas | Normalizar a mayúsculas |
| MOTIVO_BAJA | String vacío para activos | Imputar como 'ACTIVO' |

---

## Variables de fairness (Sprint 6)

SEXO, EDAD y ESTADO_CIVIL se incluyeron en el modelo con análisis de equidad
obligatorio antes de usar en decisiones de RH.

| Variable | Gap entre grupos | Resultado |
|----------|-----------------|-----------|
| SEXO | 2.3pp (H vs M) | Sin disparidad significativa (< 10pp) |
| EDAD (grupos) | 6.8pp (máximo) | Sin disparidad significativa (< 10pp) |

**Umbral de alerta:** diferencia de macro F1 > 10pp entre grupos requiere
revisión legal antes de usar el modelo en decisiones de personal.

---

## Selección de features para el dashboard (Sprint 7)

Se evaluó el modelo con subconjuntos de features ordenadas por importancia nativa
de LightGBM. Resultado:

| N features | Test Macro F1 | Caída vs modelo completo |
|------------|--------------|--------------------------|
| 4 | 0.5630 | -5.1pp |
| 6 | 0.5809 | -3.4pp |
| 8 | 0.6126 | -0.2pp ← mínimo aceptable |
| 22 | 0.6144 | referencia |

**Decisión (Opción C):** el modelo de predicción usa las 22 features completas.
El dashboard de perfil de asesor usa 6 sliders operacionales con las variables
restantes tomando el valor promedio del portafolio histórico.

---

## Métricas de evaluación

- **Métrica principal:** Macro F1 — trata todas las clases por igual
- **Justificación:** accuracy es engañoso con clases desbalanceadas
- **Métricas secundarias:** Cohen's Kappa, AUC OvR macro
- **Accuracy:** reportado pero no usado para decisiones

---

## Resultados finales

| Modelo | CV Macro F1 | Test Macro F1 | Kappa | AUC OvR |
|--------|------------|--------------|-------|---------|
| Logistic Regression (baseline) | 0.5862 ± 0.016 | 0.5696 | 0.4248 | 0.8079 |
| LightGBM (modelo final) | 0.6138 ± 0.016 | 0.6144 | 0.4813 | 0.8380 |
| XGBoost | — | 0.5700 | 0.4282 | 0.8303 |

**Modelo seleccionado:** LightGBM (+4.5pp sobre baseline, sin overfitting CV≈Test)

---

## Limitaciones conocidas

1. Dataset pequeño (~2,004 asesores) — clases intermedias con F1 < 0.51
2. Período de entrenamiento: 2016-2019 — requiere reentrenamiento para períodos posteriores
3. Producto específico: Crédito Grupal — no generalizable a otros productos sin validación
4. Variables de fairness incluidas: requieren revisión legal antes de uso en decisiones de RH
5. El modelo no debe usarse para despido o sanción automática de asesores
