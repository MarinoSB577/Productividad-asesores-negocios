import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="Perfil de Asesor", layout="wide")
st.title("Perfil de Asesor -- Prediccion de Riesgo")

st.markdown(
    "Ajusta las variables operacionales del asesor para ver el cuartil de riesgo predicho. "
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

PROMEDIOS = {
    "GRUPOS": 1.02, "CLIENTES": 9.75, "PRESTAMO": 9247.54,
    "CLIENTES_PRESTAMO": 0.86, "TASA_PROM": 218.71,
    "INCREMENTO_CARTERA": 5157.85, "CLIENTES_NUEVOS": 0.29,
    "DESEMBOLSO_CLIENTES_NUEVOS": 2105.08, "N_SEMANAS_OBS": 33.82, "EDAD": 30.88,
}
PROMEDIOS_CAT = {
    "REGION": "Oriente - Morelos", "RANGO_CICLO": "CICLO_MAYOR_5",
    "DIA_PAGO": 3, "PUESTO": "ASESOR DE NEGOCIOS",
    "PRODUCTO": "CREDITO GRUPAL", "AREA": "COMERCIAL",
    "TIPO_BAJA": "ACTIVO", "MOTIVO_BAJA": "ACTIVO",
    "EXP_MICROFINANZAS": "SI", "NIVEL_ESTUDIOS": "PREPARATORIA",
    "SEXO": "H", "ESTADO_CIVIL": "SOLTERO",
}

NUM_COLS = [
    "GRUPOS","CLIENTES","PRESTAMO","CLIENTES_PRESTAMO","TASA_PROM",
    "INCREMENTO_CARTERA","CLIENTES_NUEVOS","DESEMBOLSO_CLIENTES_NUEVOS",
    "N_SEMANAS_OBS","EDAD",
]
CAT_COLS = [
    "REGION","RANGO_CICLO","DIA_PAGO","PUESTO","PRODUCTO","AREA",
    "TIPO_BAJA","MOTIVO_BAJA","EXP_MICROFINANZAS","NIVEL_ESTUDIOS",
    "SEXO","ESTADO_CIVIL",
]

st.subheader("Variables operacionales del asesor")
col1, col2 = st.columns(2)

with col1:
    clientes_prestamo = st.slider("Clientes con prestamo activo", 0, 7, 1)
    tasa_prom         = st.slider("Tasa promedio", 131, 306, 219)
    clientes_nuevos   = st.slider("Clientes nuevos (promedio semanal)", 0, 5, 0)

with col2:
    prestamo  = st.slider("Monto promedio de prestamo (pesos)", 0, 42000, 9248, step=500)
    clientes  = st.slider("Total clientes activos", 8, 12, 10)
    grupos    = st.slider("Grupos activos", 1, 2, 1)

def predecir_perfil(cp, tp, cn, p, c, g):
    vals_num = PROMEDIOS.copy()
    vals_num["CLIENTES_PRESTAMO"] = cp
    vals_num["TASA_PROM"]         = tp
    vals_num["CLIENTES_NUEVOS"]   = cn
    vals_num["PRESTAMO"]          = p
    vals_num["CLIENTES"]          = c
    vals_num["GRUPOS"]            = g
    fila_num = pd.DataFrame([{col: vals_num[col] for col in NUM_COLS}])
    fila_cat = pd.DataFrame([PROMEDIOS_CAT])
    fila     = pd.concat([fila_num, fila_cat], axis=1)
    X_proc   = preprocessor.transform(fila)
    proba    = modelo.predict_proba(X_proc)[0]
    clase    = label_encoder.classes_[np.argmax(proba)]
    return clase, proba

clase_pred, probabilidades = predecir_perfil(
    clientes_prestamo, tasa_prom, clientes_nuevos, prestamo, clientes, grupos
)

COLORES = {
    "Q1_BAJO": "#28a745", "Q2_MEDIO_BAJO": "#87c540",
    "Q3_MEDIO_ALTO": "#ffc107", "Q4_ALTO": "#dc3545",
}
DESCRIPCIONES = {
    "Q1_BAJO"       : "Cartera saludable -- bajo riesgo de deterioro.",
    "Q2_MEDIO_BAJO" : "Cartera estable con algunos indicadores de atencion.",
    "Q3_MEDIO_ALTO" : "Senales de alerta -- supervision recomendada.",
    "Q4_ALTO"       : "Alto riesgo de deterioro -- intervencion prioritaria.",
}

st.markdown("---")
st.subheader("Resultado")

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
            f"<span style='width:140px; font-size:13px;'>{cls}</span>"
            f"<div style='background:#eee; border-radius:4px; width:200px; height:18px;'>"
            f"<div style='background:{c}; width:{pct:.0f}%; height:100%; border-radius:4px;'></div>"
            f"</div><span style='margin-left:8px; font-size:13px;'>{pct:.1f}%</span></div>",
            unsafe_allow_html=True,
        )

with col_gauge:
    nivel = {"Q1_BAJO": 1, "Q2_MEDIO_BAJO": 2, "Q3_MEDIO_ALTO": 3, "Q4_ALTO": 4}
    val   = nivel.get(clase_pred, 1)
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={"text": "Nivel de riesgo", "font": {"size": 16}},
        gauge={
            "axis"  : {"range": [1, 4], "tickvals": [1,2,3,4],
                       "ticktext": ["Q1","Q2","Q3","Q4"]},
            "bar"   : {"color": color},
            "steps" : [
                {"range": [1, 2], "color": "#d4edda"},
                {"range": [2, 3], "color": "#fff3cd"},
                {"range": [3, 4], "color": "#f8d7da"},
            ],
        }
    ))
    fig_g.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig_g, use_container_width=True)

st.caption(
    "Variables no mostradas (INCREMENTO_CARTERA, N_SEMANAS_OBS, EDAD, "
    "DESEMBOLSO_CLIENTES_NUEVOS y categoricas) toman el valor promedio "
    "del portafolio historico 2016-2019."
)
