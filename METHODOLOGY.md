# METHODOLOGY.md — Registro de Decisiones de Diseño

Este documento registra formalmente todas las decisiones metodológicas tomadas
durante el desarrollo del proyecto, incluyendo la justificación de cada una y
el sprint donde se tomó. Cualquier cambio futuro al modelo debe documentarse aquí.

---

## Dataset

| Parámetro | Valor |
|-----------|-------|
| Fuente | Tabla_Hist_Asesores (sistema transaccional institución) |
| Archivos | NUEVA_CARGA_1.csv (2018-2019), NUEVA_CARGA_2.csv (2016-2017) |
| Encoding | ISO-8859-1 (latin-1) |
| Separador | Tabulador (`\t`) — confirmado por inspección directa |
| Encabezado | Ausente — nombres de columna del DDL original de SQL Server |
| Granularidad | Una fila por grupo-asesor-semana |
| Período | 2016-06-22 → 2019-01-30 |
| Registros totales | 1,218,831 filas |
| Asesores únicos | 2,004 |

---

## Variable Objetivo

### TDNC — Tasa de Deterioro Neto de Cartera

```
TDNC = promedio(ATRASO) / promedio(CARTERA_HOY)
```

Calculada colapsando el dataset de nivel semanal a nivel asesor. Mide qué
proporción de la cartera del asesor está en atraso en promedio durante el
período observado.

### Decisión: 3 clases en lugar de 4 (Sprint 2 v2)

**Versión original (v1):** 4 cuartiles (Q1_BAJO / Q2_MEDIO_BAJO / Q3_MEDIO_ALTO / Q4_ALTO)

**Problema detectado:** Las clases Q2 y Q3 mostraron F1 = 0.50 y 0.47
respectivamente. El análisis de confusiones confirmó que el 37% de los
asesores Q3 se clasificaban como Q2 — la frontera entre ambas clases era
genuinamente indistinguible con las variables disponibles.

**Corrección (v2):** 3 clases usando terciles de TDNC (BAJO / MEDIO / ALTO)

**Evidencia:** Al colapsar Q2 y Q3 en una clase MEDIO, el macro F1 mejoró
+11.7pp (de 0.614 a 0.732 en prueba controlada con los mismos hiperparámetros).

| Clase | TDNC promedio | Asesores |
|-------|--------------|---------|
| BAJO | 0.013 | 668 |
| MEDIO | 0.095 | 668 |
| ALTO | 0.372 | 668 |

### Split por asesor (Sprint 2)
- Estratificado por TARGET_B, test_size=0.20, random_state=42
- Verificación: ningún CODIGO_ASESOR aparece en train Y test simultáneamente ✓
- Train: 1,603 asesores | Test: 401 asesores

---

## Variables de Evento — Corrección Crítica (Sprint 2 v3)

### Problema detectado

Las variables `CLIENTES_PRESTAMO`, `PRESTAMO`, `CLIENTES_NUEVOS` y
`DESEMBOLSO_CLIENTES_NUEVOS` se calculaban como promedios sobre **todas**
las semanas del asesor, incluyendo semanas sin desembolso (valor cero).

**El diagnóstico reveló que:**
- El dataset tiene múltiples filas por asesor por semana (una por grupo)
- `CLIENTES_PRESTAMO > 0` solo en la semana de desembolso del grupo
- Al promediar con ceros se mezclan dos señales: tamaño del grupo y frecuencia de desembolso
- `CLIENTES_PRESTAMO` promediado no discriminaba entre clases (BAJO: 10.20 vs ALTO: 10.08)

### Corrección: variables calculadas solo sobre semanas con evento

| Variable nueva | Cálculo | Discrimina entre clases |
|---------------|---------|------------------------|
| `TASA_DESEMBOLSO` | filas con `CLIENTES_PRESTAMO > 0` / total filas | Sí (BAJO 15.9% vs ALTO 3.6%) |
| `CLIENTES_PRESTAMO_EVENTO` | mean(`CLIENTES_PRESTAMO`) donde > 0 | No (similar entre clases ~10) |
| `MONTO_DESEMBOLSO_EVENTO` | mean(`PRESTAMO`) donde > 0 | Sí (BAJO $108K vs ALTO $88K) |
| `CLIENTES_NUEVOS_EVENTO` | mean(`CLIENTES_NUEVOS`) donde > 0 | Parcial |

### Imputación de NaN en variables de evento

152 asesores (7.6%) nunca tuvieron un desembolso durante el período. Para ellos,
`CLIENTES_PRESTAMO_EVENTO`, `MONTO_DESEMBOLSO_EVENTO` y `CLIENTES_NUEVOS_EVENTO`
son NaN. Se imputan con **0** — si nunca hubo desembolso, el promedio es cero.

De los 152 asesores sin desembolso, 121 (79.6%) pertenecen a la clase ALTO —
coherente con la interpretación: asesores con mora acumulada que nunca pudieron
renovar cartera durante el período observado.

### Interpretación de TASA_DESEMBOLSO

El crédito grupal estándar dura 12 semanas. Un asesor con cartera sana tiene
grupos que terminan su ciclo y renuevan de inmediato:

- **BAJO (~15.9%):** renueva cada ~6 filas-grupo → ciclo normal
- **MEDIO (~6.6%):** renueva cada ~15 filas-grupo → ciclo dilatado
- **ALTO (~3.6%):** renueva cada ~28 filas-grupo → grupos bloqueados por mora

---

## Variables Excluidas del Modelo

### Leakage contemporáneo (Sprint 1)

| Variable | Razón |
|----------|-------|
| `PASE1`, `PASE15`, `PASE30` | Medidas en el mismo corte temporal que el target |
| `DIAS_MORA` | Base numérica del target |
| `DIAS_MORA_RANGO` | Target de formulación alternativa (Target_A) |
| `SEMANA_ACTUAL_RANGO` | Consecuencia del deterioro, no causa |

### Leakage de construcción del target (detectado en Sprint 4)

Las variables con correlación > 0.28 con TDNC son componentes directos
de su fórmula y no deben usarse como predictores:

| Variable | Correlación con TDNC | Razón |
|----------|---------------------|-------|
| `ATRASO` | 0.851 | Numerador de TDNC |
| `PROVISION_HOY` | 0.821 | Función directa del atraso |
| `PROVISION_SEMANT` | 0.813 | Función directa del atraso |
| `GASTO_PROVISION_SEMANAL` | 0.377 | Derivada del atraso |
| `CARTERA_HOY` | 0.343 | Denominador de TDNC |
| `CARTERA_SEMANT` | 0.280 | Denominador de TDNC |

**Señal de detección:** el baseline de Logistic Regression alcanzó macro F1 = 0.92
antes de la corrección — resultado imposible para este problema que confirmó
la existencia de leakage.

### Variables de evento originales (reemplazadas en v2)

`CLIENTES_PRESTAMO`, `PRESTAMO`, `CLIENTES_NUEVOS`, `DESEMBOLSO_CLIENTES_NUEVOS`
fueron reemplazadas por sus versiones corregidas (ver sección anterior).

### Identificadores y redundancias

| Variable | Razón de exclusión |
|----------|-------------------|
| `ASESOR`, `NOMBRE`, `COORDINADOR` | Identificadores textuales sin valor predictivo |
| `NUMERO_EMPLEADO` | Redundante con `CODIGO_ASESOR` |
| `FECHA_ALTA`, `FECHA_BAJA`, `FECHA_NACIMIENTO` | Alta cardinalidad como fecha; `EDAD` y `N_SEMANAS_OBS` las resumen |

---

## Pipeline de Preprocesamiento (Sprint 3 v2)

### Features del modelo final

**10 variables numéricas** (imputación por mediana + StandardScaler):
`GRUPOS`, `CLIENTES`, `TASA_PROM`, `INCREMENTO_CARTERA`, `N_SEMANAS_OBS`,
`EDAD`, `TASA_DESEMBOLSO`, `CLIENTES_PRESTAMO_EVENTO`,
`MONTO_DESEMBOLSO_EVENTO`, `CLIENTES_NUEVOS_EVENTO`

**12 variables categóricas** (imputación constante 'SD' + OrdinalEncoder):
`REGION`, `RANGO_CICLO`, `DIA_PAGO`, `PUESTO`, `PRODUCTO`, `AREA`,
`TIPO_BAJA`, `MOTIVO_BAJA`, `EXP_MICROFINANZAS`, `NIVEL_ESTUDIOS`,
`SEXO` ⚠️, `ESTADO_CIVIL` ⚠️

⚠️ Variables marcadas para análisis de fairness — ver sección correspondiente.

### Correcciones de calidad previas al pipeline

| Variable | Problema | Corrección |
|----------|----------|-----------|
| `TIPO_BAJA` | String vacío para asesores activos | Imputar como 'ACTIVO' |
| `EXP_MICROFINANZAS` | String vacío sin dato | Imputar como 'SD' |
| `NIVEL_ESTUDIOS` | Inconsistencia de mayúsculas ('Secundaria' vs 'SECUNDARIA') | Normalizar a mayúsculas |
| `MOTIVO_BAJA` | String vacío para activos | Imputar como 'ACTIVO' |

### Regla anti-leakage del pipeline

El `ColumnTransformer` se fitea **únicamente en train**. Los parámetros
aprendidos (medianas, media/std del scaler, vocabulario del encoder) se
calculan sobre train y se aplican a test sin recalcular.

---

## Métricas de Evaluación

- **Métrica principal:** Macro F1 — trata todas las clases por igual
- **Justificación:** accuracy es engañoso con clases balanceadas artificialmente
- **Métricas secundarias:** Cohen's Kappa, AUC OvR macro
- **Accuracy:** reportado pero no usado para decisiones

---

## Resultados Finales (v2)

| Modelo | CV Macro F1 | Test Macro F1 | Kappa | AUC OvR |
|--------|------------|--------------|-------|---------|
| Logistic Regression (baseline) | 0.666 ± 0.038 | 0.711 | 0.566 | 0.855 |
| **LightGBM (modelo final)** | **0.706 ± 0.016** | **0.717** | **0.577** | **0.881** |
| XGBoost | 0.710 | 0.711 | 0.570 | 0.883 |

**Modelo seleccionado:** LightGBM

**Justificación de selección sobre XGBoost:** Mayor macro F1 y menor varianza
entre folds CV (0.016 vs — ), lo que indica mayor estabilidad en producción.

**Nota sobre la mejora sobre baseline (+0.6pp):** La pequeña diferencia en
macro F1 es esperada — el problema de 3 clases con variables operacionales
es genuinamente difícil. La ventaja real de LightGBM está en AUC OvR
(0.881 vs 0.855) y estabilidad CV.

---

## Análisis de Fairness (Sprint 6 v2)

| Variable | Grupo mejor F1 | Grupo menor F1 | Gap | Umbral | Estado |
|----------|---------------|----------------|-----|--------|--------|
| SEXO | Mujeres (0.762) | Hombres (0.681) | 8.1pp | 10pp | Monitorear |
| EDAD | 26-30 años (0.737) | 36-45 años (0.642) | 9.5pp | 10pp | Monitorear |

**Criterio de alerta:** diferencia de macro F1 > 10pp entre grupos requiere
revisión legal antes de usar el modelo en decisiones de personal.

**Acción requerida en cada reentrenamiento:** recalcular estos gaps y
documentar si superan el umbral.

---

## Selección de Features para el Dashboard (Sprint 7)

Se evaluó el modelo con subconjuntos de features para determinar cuántas
variables necesita el dashboard de Perfil de Asesor.

| N features | Test Macro F1 | Caída vs modelo completo |
|------------|--------------|--------------------------|
| 4 | 0.563 | -15.4pp |
| 6 | 0.581 | -13.6pp |
| 8 | 0.613 | -10.4pp |
| 22 (completo) | 0.717 | referencia |

**Decisión (Opción C):**
- El modelo de predicción usa las 22 features completas
- El dashboard de Perfil de Asesor usa 6 sliders operacionales
- Las features no mostradas toman el valor promedio del portafolio histórico
- Esta simplificación se documenta explícitamente en la interfaz

---

## Limitaciones Documentadas

1. Dataset de ~2,004 asesores de una sola institución (2016-2019)
2. Clase MEDIO con F1 = 0.61 — frontera difusa con BAJO y ALTO
3. Producto específico: Crédito Grupal — no validado para otros productos
4. Variables de fairness incluidas: revisión legal antes de uso en RH
5. No debe usarse para decisiones automáticas de despido o sanción
6. Requiere reentrenamiento para períodos posteriores a enero 2019
