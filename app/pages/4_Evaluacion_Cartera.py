"""
Pagina 4 — Evaluacion de Cartera
Clasificacion masiva de asesores desde un archivo CSV.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Evaluacion de Cartera", layout="wide")
st.title("Evaluacion de Cartera — Clasificacion Masiva de Asesores")
st.markdown(
    "Sube un archivo CSV con los datos operacionales de tus asesores para clasificarlos "
    "en tres niveles de riesgo de forma automatica. "
    "El resultado se ordena por probabilidad de riesgo ALTO de mayor a menor."
)

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"
SAMPLE    = Path(__file__).parent.parent.parent / "data" / "sample"

@st.cache_resource
def cargar_modelo():
    modelo        = joblib.load(PROCESSED / "modelo_final.joblib")
    label_encoder = joblib.load(PROCESSED / "label_encoder.joblib")
    preprocessor  = joblib.load(PROCESSED / "preprocessor.joblib")
    return modelo, label_encoder, preprocessor

modelo, label_encoder, preprocessor = cargar_modelo()
CLASS_NAMES = list(label_encoder.classes_)

NUM_COLS = [
    "GRUPOS", "CLIENTES", "TASA_PROM", "INCREMENTO_CARTERA",
    "N_SEMANAS_OBS", "EDAD", "TASA_DESEMBOLSO",
    "CLIENTES_PRESTAMO_EVENTO", "MONTO_DESEMBOLSO_EVENTO",
    "CLIENTES_NUEVOS_EVENTO",
]
CAT_COLS = [
    "REGION", "RANGO_CICLO", "DIA_PAGO", "PUESTO", "PRODUCTO", "AREA",
    "TIPO_BAJA", "MOTIVO_BAJA", "EXP_MICROFINANZAS", "NIVEL_ESTUDIOS",
    "SEXO", "ESTADO_CIVIL",
]
ALL_FEATURES = NUM_COLS + CAT_COLS

COLORES_SEMAFORO = {
    "ALTO" : "🔴",
    "MEDIO": "🟡",
    "BAJO" : "🟢",
}
COLORES_BG = {
    "ALTO" : "#FCE4D6",
    "MEDIO": "#FFF2CC",
    "BAJO" : "#D5E8F0",
}

# ── Instrucciones y descarga de plantilla ──────────────────────────────────
with st.expander("Ver instrucciones y descargar plantilla CSV", expanded=False):
    st.markdown("""
    **Columnas requeridas en el CSV:**

    | Columna | Tipo | Descripcion |
    |---------|------|-------------|
    | `CODIGO_ASESOR` | Entero | ID unico del asesor (opcional, para identificacion) |
    | `TASA_DESEMBOLSO` | Decimal 0-1 | Fraccion de semanas con desembolso activo |
    | `TASA_PROM` | Decimal | Tasa promedio de la cartera |
    | `CLIENTES` | Entero | Total clientes activos |
    | `GRUPOS` | Entero | Grupos activos |
    | `MONTO_DESEMBOLSO_EVENTO` | Decimal | Monto promedio desembolsado por grupo |
    | `CLIENTES_NUEVOS_EVENTO` | Decimal | Clientes nuevos promedio en desembolsos |
    | `INCREMENTO_CARTERA` | Decimal | Variacion semanal de cartera |
    | `N_SEMANAS_OBS` | Entero | Semanas observadas |
    | `EDAD` | Entero | Edad del asesor |
    | `CLIENTES_PRESTAMO_EVENTO` | Decimal | Tamano promedio del grupo al desembolso |
    | Categoricas | Texto | REGION, RANGO_CICLO, DIA_PAGO, PUESTO, PRODUCTO, AREA, TIPO_BAJA, MOTIVO_BAJA, EXP_MICROFINANZAS, NIVEL_ESTUDIOS, SEXO, ESTADO_CIVIL |

    Los campos no incluidos se imputaran con los valores promedio del portafolio historico.
    """)

    # Descargar datos sinteticos como plantilla
    sample_path = SAMPLE / "advisors_sample_anon.parquet"
    if sample_path.exists():
        df_sample = pd.read_parquet(sample_path)
        cols_disponibles = [c for c in ["CODIGO_ASESOR"] + ALL_FEATURES if c in df_sample.columns]
        csv_plantilla = df_sample[cols_disponibles].head(10).to_csv(index=False)
        st.download_button(
            label="Descargar plantilla CSV (10 registros de ejemplo)",
            data=csv_plantilla,
            file_name="plantilla_asesores.csv",
            mime="text/csv",
        )

# ── Carga de archivo ──────────────────────────────────────────────────────
st.markdown("---")
archivo = st.file_uploader(
    "Sube tu archivo CSV de asesores",
    type=["csv"],
    help="El archivo debe contener al menos TASA_DESEMBOLSO y TASA_PROM para resultados confiables."
)

# Valores promedio para imputar columnas faltantes
PROMEDIOS_NUM = {
    "GRUPOS": 1.02, "CLIENTES": 9.75, "TASA_PROM": 218.71,
    "INCREMENTO_CARTERA": 5157.85, "N_SEMANAS_OBS": 33.82, "EDAD": 30.88,
    "TASA_DESEMBOLSO": 0.09, "CLIENTES_PRESTAMO_EVENTO": 10.04,
    "MONTO_DESEMBOLSO_EVENTO": 110971.60, "CLIENTES_NUEVOS_EVENTO": 3.03,
}
PROMEDIOS_CAT = {
    "REGION": "Oriente - Morelos", "RANGO_CICLO": "CICLO_MAYOR_5",
    "DIA_PAGO": "3", "PUESTO": "ASESOR DE NEGOCIOS",
    "PRODUCTO": "CREDITO GRUPAL", "AREA": "COMERCIAL",
    "TIPO_BAJA": "ACTIVO", "MOTIVO_BAJA": "ACTIVO",
    "EXP_MICROFINANZAS": "SI", "NIVEL_ESTUDIOS": "PREPARATORIA",
    "SEXO": "H", "ESTADO_CIVIL": "SOLTERO",
}

if archivo is not None:
    try:
        df_input = pd.read_csv(archivo)
        st.success(f"Archivo cargado: {len(df_input):,} asesores")

        # Imputar columnas faltantes con promedios
        for col in NUM_COLS:
            if col not in df_input.columns:
                df_input[col] = PROMEDIOS_NUM.get(col, 0)
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(PROMEDIOS_NUM.get(col, 0))

        for col in CAT_COLS:
            if col not in df_input.columns:
                df_input[col] = PROMEDIOS_CAT.get(col, "SD")
            df_input[col] = df_input[col].fillna(PROMEDIOS_CAT.get(col, "SD")).astype(str)

        # Aplicar pipeline y predecir
        X = preprocessor.transform(df_input[ALL_FEATURES])
        proba  = modelo.predict_proba(X)
        clases = label_encoder.classes_[np.argmax(proba, axis=1)]

        # Construir resultado
        df_result = pd.DataFrame()
        if "CODIGO_ASESOR" in df_input.columns:
            df_result["Asesor"] = df_input["CODIGO_ASESOR"].values
        else:
            df_result["Asesor"] = [f"Asesor_{i+1}" for i in range(len(df_input))]

        # Agregar columnas clave si existen
        for col in ["REGION", "SUCURSAL", "COORDINADOR"]:
            if col in df_input.columns:
                df_result[col] = df_input[col].values

        df_result["Riesgo"]      = clases
        df_result["Semaforo"]    = [COLORES_SEMAFORO.get(c, "⚪") for c in clases]
        df_result["Prob_ALTO"]   = (proba[:, CLASS_NAMES.index("ALTO")] * 100).round(1)
        df_result["Prob_MEDIO"]  = (proba[:, CLASS_NAMES.index("MEDIO")] * 100).round(1)
        df_result["Prob_BAJO"]   = (proba[:, CLASS_NAMES.index("BAJO")] * 100).round(1)

        # Ordenar por probabilidad de ALTO descendente
        df_result = df_result.sort_values("Prob_ALTO", ascending=False).reset_index(drop=True)

        # ── Resumen ejecutivo ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Resumen de la cartera")

        n_alto  = (clases == "ALTO").sum()
        n_medio = (clases == "MEDIO").sum()
        n_bajo  = (clases == "BAJO").sum()
        total   = len(clases)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total asesores", f"{total:,}")
        col2.metric("🔴 Riesgo ALTO", f"{n_alto:,}", f"{n_alto/total*100:.1f}%")
        col3.metric("🟡 Riesgo MEDIO", f"{n_medio:,}", f"{n_medio/total*100:.1f}%")
        col4.metric("🟢 Riesgo BAJO", f"{n_bajo:,}", f"{n_bajo/total*100:.1f}%")

        # ── Filtros ───────────────────────────────────────────────────────
        st.markdown("---")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filtro_riesgo = st.multiselect(
                "Filtrar por nivel de riesgo:",
                ["ALTO", "MEDIO", "BAJO"],
                default=["ALTO", "MEDIO", "BAJO"],
            )
        with col_f2:
            if "REGION" in df_result.columns:
                regiones = ["Todas"] + sorted(df_result["REGION"].dropna().unique().tolist())
                filtro_region = st.selectbox("Filtrar por region:", regiones)
            else:
                filtro_region = "Todas"

        # Aplicar filtros
        df_show = df_result[df_result["Riesgo"].isin(filtro_riesgo)].copy()
        if filtro_region != "Todas" and "REGION" in df_show.columns:
            df_show = df_show[df_show["REGION"] == filtro_region]

        st.caption(f"Mostrando {len(df_show):,} de {total:,} asesores — ordenados por probabilidad de riesgo ALTO")

        # ── Tabla con semaforo ─────────────────────────────────────────────
        def color_fila(row):
            color = COLORES_BG.get(row["Riesgo"], "#FFFFFF")
            return [f"background-color: {color}"] * len(row)

        cols_mostrar = ["Semaforo", "Asesor"]
        for c in ["REGION", "SUCURSAL", "COORDINADOR"]:
            if c in df_show.columns:
                cols_mostrar.append(c)
        cols_mostrar += ["Riesgo", "Prob_ALTO", "Prob_MEDIO", "Prob_BAJO"]

        st.dataframe(
            df_show[cols_mostrar].style.apply(color_fila, axis=1).format({
                "Prob_ALTO": "{:.1f}%",
                "Prob_MEDIO": "{:.1f}%",
                "Prob_BAJO": "{:.1f}%",
            }),
            use_container_width=True,
            height=450,
        )

        # ── Descarga ──────────────────────────────────────────────────────
        csv_resultado = df_result.to_csv(index=False)
        st.download_button(
            label="Descargar resultado completo en CSV",
            data=csv_resultado,
            file_name="clasificacion_riesgo_asesores.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        st.info("Verifica que el archivo tenga el formato correcto. Descarga la plantilla de ejemplo arriba.")

else:
    # Mostrar demo con datos sinteticos
    st.info("Sube un archivo CSV para comenzar. Puedes usar la plantilla de ejemplo en las instrucciones.")

    sample_path = SAMPLE / "advisors_sample_anon.parquet"
    if sample_path.exists():
        st.markdown("**Vista previa con datos sinteticos de ejemplo:**")
        df_demo = pd.read_parquet(sample_path)

        for col in NUM_COLS:
            if col not in df_demo.columns:
                df_demo[col] = PROMEDIOS_NUM.get(col, 0)
            df_demo[col] = pd.to_numeric(df_demo[col], errors='coerce').fillna(PROMEDIOS_NUM.get(col, 0))
        for col in CAT_COLS:
            if col not in df_demo.columns:
                df_demo[col] = PROMEDIOS_CAT.get(col, "SD")
            df_demo[col] = df_demo[col].fillna(PROMEDIOS_CAT.get(col, "SD")).astype(str)

        X_demo    = preprocessor.transform(df_demo[ALL_FEATURES])
        proba_d   = modelo.predict_proba(X_demo)
        clases_d  = label_encoder.classes_[np.argmax(proba_d, axis=1)]

        df_preview = pd.DataFrame({
            "Semaforo"  : [COLORES_SEMAFORO.get(c, "⚪") for c in clases_d],
            "Riesgo"    : clases_d,
            "Prob_ALTO" : (proba_d[:, CLASS_NAMES.index("ALTO")] * 100).round(1),
            "Prob_MEDIO": (proba_d[:, CLASS_NAMES.index("MEDIO")] * 100).round(1),
            "Prob_BAJO" : (proba_d[:, CLASS_NAMES.index("BAJO")] * 100).round(1),
        }).sort_values("Prob_ALTO", ascending=False).reset_index(drop=True)

        def color_fila_demo(row):
            color = COLORES_BG.get(row["Riesgo"], "#FFFFFF")
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df_preview.style.apply(color_fila_demo, axis=1).format({
                "Prob_ALTO": "{:.1f}%",
                "Prob_MEDIO": "{:.1f}%",
                "Prob_BAJO": "{:.1f}%",
            }),
            use_container_width=True,
            height=400,
        )
        st.caption("Datos sinteticos anonimizados — solo para demostracion.")
