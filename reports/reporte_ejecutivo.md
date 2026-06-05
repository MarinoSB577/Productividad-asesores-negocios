# Reporte Ejecutivo — Modelo de Caracterización de Asesores por Riesgo de Cartera

**Institución:** Microfinanciera mexicana (datos anonimizados)
**Período de análisis:** Junio 2016 – Enero 2019
**Modelo:** LightGBM — Clasificación multiclase (4 cuartiles de riesgo)
**Versión:** 1.0 | Fecha: Junio 2026

---

## Resumen Ejecutivo

Se analizaron **2,004 asesores de negocios** usando registros operacionales
semanales del período 2016-2019. El modelo clasifica a cada asesor en uno de
cuatro cuartiles de riesgo de deterioro de cartera con un **Macro F1 de 0.614**,
mejorando en 4.5 puntos porcentuales sobre el modelo de referencia.

Los asesores en los extremos — bajo riesgo (Q1) y alto riesgo (Q4) — se
identifican con precisión superior al 75%, que es donde la intervención
temprana tiene mayor valor de negocio.

---

## El Problema de Negocio

La calidad de la cartera en microfinanzas está determinada en gran medida por
el comportamiento operacional del asesor: cuántos grupos gestiona, qué tasas
maneja, cuántos clientes nuevos incorpora. Un asesor con señales de deterioro
puede intervenir antes de que la mora se materialice — si se detecta a tiempo.

El modelo responde la pregunta: **dado el perfil operacional actual de un asesor,
¿cuál es su cuartil de riesgo de deterioro de cartera?**

---

## Metodología

### Construcción del target

Se definió la **Tasa de Deterioro Neto de Cartera (TDNC)** por asesor:

> TDNC = Atraso promedio / Cartera promedio

Los asesores se clasificaron en cuatro cuartiles:

| Cuartil | TDNC promedio | Interpretación |
|---------|--------------|----------------|
| Q1_BAJO | 0.006 | Cartera prácticamente sin atraso |
| Q2_MEDIO_BAJO | 0.057 | Atraso menor al 10% de la cartera |
| Q3_MEDIO_ALTO | 0.141 | Atraso entre 10% y 20% de la cartera |
| Q4_ALTO | 0.435 | Atraso superior al 20% de la cartera |

### Variables utilizadas

El modelo usa **22 variables puramente operacionales** — sin incluir el atraso
ni la provisión, que son consecuencia del deterioro y no causas.

Las cuatro variables de mayor impacto (SHAP):

| Variable | Importancia SHAP | Unidad |
|----------|-----------------|--------|
| CLIENTES_PRESTAMO | 0.976 | Clientes con préstamo activo |
| PRESTAMO | 0.329 | Monto promedio de préstamo (pesos) |
| TASA_PROM | 0.317 | Tasa promedio de la cartera |
| CLIENTES_NUEVOS | 0.257 | Clientes nuevos por período |

---

## Resultados del Modelo

### Desempeño por cuartil

| Cuartil | Precisión | Recall | F1 | Interpretación |
|---------|-----------|--------|----|----------------|
| Q1_BAJO | 0.76 | 0.70 | 0.73 | Bien identificado |
| Q2_MEDIO_BAJO | 0.47 | 0.54 | 0.50 | Frontera difusa con Q3 |
| Q3_MEDIO_ALTO | 0.48 | 0.46 | 0.47 | Frontera difusa con Q2 |
| Q4_ALTO | 0.77 | 0.75 | 0.76 | Bien identificado |

**Interpretación:** Los cuartiles extremos son identificables con alta
confianza. La frontera entre Q2 y Q3 es genuinamente difusa — los asesores
en estos cuartiles intermedios requieren supervisión adicional más allá del
modelo para una clasificación definitiva.

### Patrón de errores

Las confusiones del modelo ocurren casi exclusivamente entre clases adyacentes:
Q3 se confunde con Q2 en el 37% de los casos de Q3. Ningún asesor Q1_BAJO
se clasifica como Q4_ALTO excepto en 3 casos de 100 — el modelo no comete
errores graves.

---

## Reglas de Decisión Operacionales

Basadas en las 4 variables más importantes, expresadas en unidades reales
de negocio:

### Perfil Q1_BAJO — Asesor de bajo riesgo

> CLIENTES_PRESTAMO > 1.9 **Y** TASA_PROM < 220 **Y** CLIENTES_NUEVOS > 0.3 **Y** PRESTAMO > $11,400

**Interpretación:** Asesores con cartera activa y creciente, tasas moderadas
y captación continua de nuevos clientes. Son el perfil de referencia de la institución.

### Perfil Q4_ALTO — Asesor de alto riesgo

> CLIENTES_PRESTAMO < 0.5 **Y** TASA_PROM > 237 **Y** CLIENTES_NUEVOS < 0.1 **Y** PRESTAMO < $2,900

**Interpretación:** Asesores con muy pocos clientes activos, cartera de bajo
monto y sin captación de nuevos clientes. Son el perfil de mayor urgencia para
intervención.

### Tabla de perfiles completa

| Perfil | CLIENTES_PRESTAMO | TASA_PROM | CLIENTES_NUEVOS | PRESTAMO promedio | Riesgo |
|--------|-----------------|-----------|-----------------|------------------|--------|
| Consolidado activo | > 1.9 clientes | 179 – 256 | > 0.3/sem | > $11,400 | Q1_BAJO |
| Estable moderado | 0.6 – 0.8 clientes | 178 – 229 | 0.1 – 0.2/sem | $6,100 – $10,300 | Q2_MEDIO_BAJO |
| En deterioro incipiente | 0.5 – 0.7 clientes | 189 – 235 | 0.1 – 0.2/sem | $4,700 – $8,600 | Q3_MEDIO_ALTO |
| En crisis | < 0.5 clientes | > 237 | < 0.1/sem | < $4,800 | Q4_ALTO |

---

## Análisis de Equidad Algorítmica

Se verificó que el modelo no discrimina por características sociodemográficas:

| Variable | Grupo con mejor F1 | Grupo con menor F1 | Diferencia | Resultado |
|----------|-------------------|-------------------|------------|-----------|
| Género | Mujeres (0.618) | Hombres (0.595) | 2.3pp | Sin disparidad |
| Edad | ≤ 25 años (0.636) | 31-35 años (0.568) | 6.8pp | Sin disparidad |

**Umbral institucional recomendado:** diferencias mayores a 10pp requieren
revisión legal antes de usar el modelo en decisiones de personal.

---

## Aplicaciones Recomendadas

### 1. Priorización de supervisión
Concentrar visitas de coordinadores en asesores clasificados como Q4_ALTO.
El modelo identifica correctamente este grupo en el 75% de los casos —
reduciendo el universo de revisión del 100% al 25% con alta precisión.

### 2. Onboarding de asesores nuevos
Con 8 semanas de operación es posible asignar una categoría de riesgo
preliminar y definir la intensidad de acompañamiento inicial.

### 3. Señal de alerta temprana
Asesores que migran de Q1 a Q2 en períodos consecutivos representan una
señal de deterioro incipiente — intervención en esta etapa tiene mayor
efectividad que cuando ya están en Q4.

---

## Limitaciones y Uso Responsable

**El modelo NO debe usarse para:**
- Decisiones de despido o sanción sin revisión humana complementaria
- Productos distintos a Crédito Grupal sin revalidación
- Períodos posteriores a enero 2019 sin reentrenamiento
- Instituciones distintas a la de origen sin validación

**Confiabilidad por zona de predicción:**
- Alta: asesores clasificados Q1 o Q4 con probabilidad > 70%
- Media: asesores en Q2 o Q3 — complementar con criterio del coordinador
- Baja: cualquier predicción con probabilidad máxima < 55%

---

## Stack Tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Lenguaje | Python 3.11 |
| Modelo | LightGBM 4.6 con optimización Optuna |
| Interpretabilidad | SHAP (TreeExplainer) |
| Tracking | MLflow 3.13 (backend SQLite) |
| Dashboard | Streamlit 1.58 |
| Almacenamiento | DuckDB 1.5 |
| Gestión de entorno | uv + pyproject.toml |

---

*Reporte generado automáticamente desde los artefactos del proyecto.
Para reproducir el análisis completo, consultar el repositorio GitHub.*
