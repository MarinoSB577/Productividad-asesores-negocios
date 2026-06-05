import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="Perfil de Asesor", layout="wide")
st.title("Perfil de Asesor -- Prediccion de Riesgo")
st.markdown(
    "Ajusta las variables operacionales del asesor para ver el nivel de riesgo predicho. "
    "Las variables no mostradas toman el valor promedio del portafolio historico 2016-2019."
)

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"

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

# Valores promedio del portafolio — categoricas como string
PROMEDIOS_NUM = {
    "GRUPOS"                   : 1.02,
    "CLIENTES"                 : 9.75,
    "TASA_PROM"                : 218.71,
    "INCREMENTO_CARTERA"       : 5157.85,
    "N_SEMANAS_OBS"            : 33.82,
    "EDAD"                     : 30.88,
    "TASA_DESEMBOLSO"          : 0.09,
    "CLIENTES_PRESTAMO_EVENTO" : 10.04,
    "MONTO_DESEMBOLSO_EVENTO"  : 110971.60,
    "CLIENTES_NUEVOS_EVENTO"   : 3.03,
}
PROMEDIOS_CAT = {
    "REGION"           : "Oriente - Morelos",
    "RANGO_CICLO"      : "CICLO_MAYOR_5",
    "DIA_PAGO"         : "3",       # string — el pipeline fue entrenado con strings
    "PUESTO"           : "ASESOR DE NEGOCIOS",
    "PRODUCTO"         : "CREDITO GRUPAL",
    "AREA"             : "COMERCIAL",
    "TIPO_BAJA"        : "ACTIVO",
    "MOTIVO_BAJA"      : "ACTIVO",
    "EXP_MICROFINANZAS": "SI",
    "NIVEL_ESTUDIOS"   : "PREPARATORIA",
    "SEXO"             : "H",
    "ESTADO_CIVIL"     : "SOLTERO",
}

st.subheader("Variables operacionales del asesor")
st.caption(
    "TASA_DESEMBOLSO es la variable mas importante del modelo. "
    "Representa la fraccion de filas-grupo con desembolso activo."
)

col1, col2 = st.columns(2)

with col1:
    tasa_desembolso = st.slider(
        "Tasa de desembolso (0=sin renovacion, 0.50=renovacion activa)",
        min_value=0.00, max_value=0.50, value=0.09, step=0.01,
        help="BAJO~0.16, MEDIO~0.07, ALTO~0.04"
    )
    tasa_prom = st.slider("Tasa promedio de la cartera", 131, 306, 219)
    clientes  = st.slider("Total clientes activos", 8, 12, 10)

with col2:
    monto_evento = st.slider(
        "Monto promedio desembolsado por grupo (pesos)",
        min_value=24000, max_value=200000, value=110972, step=1000,
    )
    clientes_nuevos_evento = st.slider(
        "Clientes nuevos promedio en desembolsos",
        min_value=0.0, max_value=10.0, value=3.0, step=0.5,
    )
    grupos = st.slider("Grupos activos", 1, 5, 1)


def predecir_perfil(tasa_d, tasa_p, clts, monto, cn_ev, grps):
    vals_num = PROMEDIOS_NUM.copy()
    vals_num["TASA_DESEMBOLSO"]         = float(tasa_d)
    vals_num["TASA_PROM"]               = float(tasa_p)
    vals_num["CLIENTES"]                = float(clts)
    vals_num["MONTO_DESEMBOLSO_EVENTO"] = float(monto)
    vals_num["CLIENTES_NUEVOS_EVENTO"]  = float(cn_ev)
    vals_num["GRUPOS"]                  = float(grps)

    # Construir DataFrame con exactamente las columnas que espera el pipeline
    fila_num = {col: [vals_num[col]] for col in NUM_COLS}
    fila_cat = {col: [str(PROMEDIOS_CAT[col])] for col in CAT_COLS}

    fila = pd.DataFrame({**fila_num, **fila_cat})

    X_proc = preprocessor.transform(fila)
    proba  = modelo.predict_proba(X_proc)[0]
    clase  = label_encoder.classes_[np.argmax(proba)]
    return clase, proba


clase_pred, probabilidades = predecir_perfil(
    tasa_desembolso, tasa_prom, clientes,
    monto_evento, clientes_nuevos_evento, grupos
)

COLORES = {"ALTO": "#dc3545", "BAJO": "#28a745", "MEDIO": "#ffc107"}
DESCRIPCIONES = {
    "ALTO" : "Alto riesgo de deterioro -- intervencion prioritaria.",
    "BAJO" : "Cartera saludable -- bajo riesgo de deterioro.",
    "MEDIO": "Riesgo moderado -- supervision periodica recomendada.",
}

st.markdown("---")
st.subheader("Resultado de la prediccion")

color = COLORES.get(clase_pred, "#6c757d")
desc  = DESCRIPCIONES.get(clase_pred, "")

col_res, col_gauge = st.columns(2)

with col_res:
    st.markdown(
        f"<div style='background:{color}22; border-left:6px solid {color};"
        f"padding:20px; border-radius:8px;'>"
        f"<h2 style='color:{color}; margin:0;'>{clase_pred}</h2>"
        f"<p style='font-size:16px; margin-top:8px;'>{desc}</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Probabilidades por clase:**")
    for i, cls in enumerate(CLASS_NAMES):
        pct = probabilidades[i] * 100
        c   = COLORES.get(cls, "#6c757d")
        st.markdown(
            f"<div style='display:flex; align-items:center; margin-bottom:6px;'>"
            f"<span style='width:60px; font-size:13px;'>{cls}</span>"
            f"<div style='background:#eee; border-radius:4px; width:220px; height:18px;'>"
            f"<div style='background:{c}; width:{pct:.0f}%; height:100%;"
            f"border-radius:4px;'></div></div>"
            f"<span style='margin-left:8px; font-size:13px;'>{pct:.1f}%</span></div>",
            unsafe_allow_html=True,
        )

with col_gauge:
    nivel = {"BAJO": 1, "MEDIO": 2, "ALTO": 3}
    val   = nivel.get(clase_pred, 1)
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={"text": "Nivel de riesgo", "font": {"size": 16}},
        gauge={
            "axis"  : {"range": [1, 3], "tickvals": [1, 2, 3],
                       "ticktext": ["BAJO", "MEDIO", "ALTO"]},
            "bar"   : {"color": color},
            "steps" : [
                {"range": [1, 2], "color": "#d4edda"},
                {"range": [2, 3], "color": "#f8d7da"},
            ],
        }
    ))
    fig_g.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig_g, use_container_width=True)

st.caption(
    "Variables no mostradas (INCREMENTO_CARTERA, N_SEMANAS_OBS, EDAD, "
    "CLIENTES_PRESTAMO_EVENTO y todas las categoricas) toman el valor promedio "
    "del portafolio historico 2016-2019."
)

st.markdown("---")
st.markdown("**Referencia de TASA_DESEMBOLSO por clase:**")
ref = pd.DataFrame({
    "Clase": ["BAJO", "MEDIO", "ALTO"],
    "Tasa promedio": ["0.159 (16%)", "0.066 (7%)", "0.036 (4%)"],
    "Interpretacion": [
        "1 de cada 6 filas-grupo renueva (~ciclo normal 12 semanas)",
        "1 de cada 15 filas-grupo renueva",
        "1 de cada 28 filas-grupo renueva (mora acumulada)",
    ]
})
st.dataframe(ref, use_container_width=True, hide_index=True)
