# Reporte Ejecutivo — Modelo de Caracterización de Asesores por Riesgo de Cartera

**Institución:** Microfinanciera mexicana (datos anonimizados)
**Período de análisis:** Junio 2016 – Enero 2019
**Modelo:** LightGBM — Clasificación en 3 niveles de riesgo
**Versión:** 2.0 | Junio 2026

---

## Resumen Ejecutivo

Se analizaron **2,004 asesores de negocios** usando registros operacionales
del período 2016-2019. El modelo clasifica a cada asesor en uno de tres
niveles de riesgo de deterioro de cartera con un **Macro F1 de 0.717**.

Los asesores de riesgo ALTO se identifican correctamente en el **82% de los
casos** — suficiente para priorizar supervisión sin revisar el 100% de la cartera.

---

## El Problema de Negocio

La mora en microfinanzas no aparece de golpe — se acumula semana a semana
en grupos que no renuevan su crédito. Un asesor con grupos estancados puede
tener cartera deteriorada **meses antes** de que el sistema lo registre
formalmente.

El modelo responde la pregunta: **¿qué tan activo está el asesor renovando
su cartera hoy?** — y usa esa señal para predecir su nivel de riesgo.

---

## Hallazgo Metodológico Central

La variable más importante del modelo no existía en el sistema original.
Se construyó durante el análisis:

### TASA_DESEMBOLSO

```
TASA_DESEMBOLSO = filas con desembolso activo / total filas del asesor
```

**Interpretación:** El crédito grupal estándar dura 12 semanas. Un asesor
con cartera sana tiene grupos que terminan su ciclo puntualmente y renuevan
de inmediato. Un asesor con mora acumulada tiene grupos que no califican
para renovación — y por eso desembolsa mucho menos frecuentemente.

| Nivel de riesgo | TASA_DESEMBOLSO | Cada cuántas semanas renueva |
|----------------|----------------|------------------------------|
| BAJO | 15.9% | ~1 de cada 6 semanas |
| MEDIO | 6.6% | ~1 de cada 15 semanas |
| ALTO | 3.6% | ~1 de cada 28 semanas |

Un asesor BAJO renueva su cartera cada 6 semanas aproximadamente — es decir,
está dentro del ciclo normal de 12 semanas con varios grupos en distintas etapas.
Un asesor ALTO tarda 28 semanas en promedio — sus grupos están bloqueados por mora.

---

## Resultados del Modelo

### Desempeño por nivel de riesgo

| Nivel | Precisión | Recall | F1 | Interpretación |
|-------|-----------|--------|----|----------------|
| BAJO | 0.74 | 0.75 | 0.74 | Bien identificado |
| MEDIO | 0.63 | 0.59 | 0.61 | Frontera difusa con BAJO y ALTO |
| ALTO | 0.78 | 0.82 | 0.80 | Bien identificado |

**Interpretación práctica:** El modelo detecta correctamente 82 de cada
100 asesores de alto riesgo real. De los que clasifica como ALTO, 78 de
cada 100 efectivamente lo son. Para un instrumento de priorización
operacional, esto es suficientemente confiable.

### Comparación de modelos

| Modelo | Macro F1 | AUC OvR | Kappa |
|--------|----------|---------|-------|
| Regresión Logística (baseline) | 0.711 | 0.855 | 0.566 |
| **LightGBM (modelo final)** | **0.717** | **0.881** | **0.577** |
| XGBoost | 0.711 | 0.883 | 0.570 |

La ventaja de LightGBM sobre el baseline es pequeña en macro F1 (+0.6pp),
pero significativa en AUC OvR (0.881 vs 0.855) — importante cuando se usa
el score de probabilidad para priorizar, no solo la clase predicha.

---

## Variables Más Importantes

| Variable | Importancia SHAP | Interpretación de negocio |
|----------|-----------------|--------------------------|
| TASA_DESEMBOLSO | 0.935 | Actividad de renovación de cartera |
| TASA_PROM | 0.223 | Tasas altas se asocian con clientes de mayor riesgo |
| N_SEMANAS_OBS | 0.183 | Antigüedad del asesor en el período observado |
| INCREMENTO_CARTERA | 0.136 | Crecimiento o contracción semanal de cartera |
| CLIENTES | 0.106 | Número de clientes activos |

---

## Reglas de Decisión Operacionales

Basadas en las 4 variables más interpretables por un coordinador de zona:

### Perfil BAJO — Asesor de bajo riesgo

> TASA_DESEMBOLSO > 0.12 **Y** TASA_PROM < 220 **Y**
> MONTO_DESEMBOLSO_EVENTO > $108,000 **Y** CLIENTES_NUEVOS_EVENTO > 3.0

**Significado:** El asesor renueva frecuentemente, sus grupos tienen tasas
moderadas y montos saludables, y está incorporando clientes nuevos activamente.

### Perfil ALTO — Asesor de alto riesgo

> TASA_DESEMBOLSO < 0.05 **Y** TASA_PROM > 236 **Y**
> MONTO_DESEMBOLSO_EVENTO < $107,000 **Y** CLIENTES_NUEVOS_EVENTO < 2.5

**Significado:** El asesor casi no renueva cartera (grupos bloqueados por mora),
opera con tasas altas y montos bajos, y tiene poca captación de clientes nuevos.

### Tabla de perfiles completa

| Perfil | TASA_DESEMBOLSO | TASA_PROM | MONTO_EVENTO | CLIENTES_NUEVOS | Riesgo |
|--------|----------------|-----------|--------------|-----------------|--------|
| Consolidado activo | > 12% | < 220 | > $108K | > 3.0 | BAJO |
| Moderado estable | 5% – 12% | 190 – 235 | $95K – $120K | 2.0 – 3.5 | MEDIO |
| Cartera estancada | < 5% | > 236 | < $107K | < 2.5 | ALTO |

---

## Análisis de Equidad Algorítmica

| Variable | Grupo mejor F1 | Grupo menor F1 | Diferencia | Umbral | Estado |
|----------|---------------|----------------|------------|--------|--------|
| Género | Mujeres (0.762) | Hombres (0.681) | 8.1pp | 10pp | Monitorear |
| Edad | 26-30 años (0.737) | 36-45 años (0.642) | 9.5pp | 10pp | Monitorear |

Ambos gaps están por debajo del umbral de alerta de 10pp, pero su proximidad
requiere monitoreo activo en cada actualización del modelo. Antes de usar
el modelo en decisiones de personal, se recomienda revisión legal.

---

## Aplicaciones Recomendadas

### 1. Priorización de supervisión
Enfocar visitas de coordinadores en asesores clasificados como ALTO.
El modelo reduce el universo de revisión al 33% de la cartera con 82%
de precisión en la detección.

### 2. Alerta temprana por TASA_DESEMBOLSO
Un asesor que cae de TASA_DESEMBOLSO > 0.10 a < 0.05 en dos cortes
consecutivos es una señal de alerta temprana — antes de que la mora
aparezca en los reportes de cartera.

### 3. Segmentación para campañas comerciales
Asesores BAJO son el universo prioritario para campañas de ampliación
de crédito. Asesores ALTO deben recibir intervención de cobranza antes
de cualquier oferta comercial.

---

## Limitaciones y Uso Responsable

**El modelo NO debe usarse para:**
- Decisiones de despido o sanción sin revisión humana complementaria
- Productos distintos a Crédito Grupal sin revalidación
- Períodos posteriores a enero 2019 sin reentrenamiento
- Instituciones distintas a la de origen sin validación

**Confiabilidad por zona de predicción:**
- Alta confiabilidad: BAJO o ALTO con probabilidad > 70%
- Confiabilidad media: MEDIO — complementar con criterio del coordinador
- Confiabilidad baja: cualquier predicción con probabilidad máxima < 55%

---

## Stack Tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Lenguaje | Python 3.11 |
| Modelo | LightGBM 4.6 + Optuna 4.9 |
| Interpretabilidad | SHAP (TreeExplainer) |
| Tracking | MLflow 3.13 (SQLite) |
| Dashboard | Streamlit 1.58 |
| Almacenamiento | DuckDB 1.5 |
| Entorno | uv + pyproject.toml |

---

*Para reproducir el análisis completo, consultar el repositorio GitHub.*
