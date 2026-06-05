import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Explorador SHAP", layout="wide")
st.title("Explorador SHAP -- Importancia de Variables")

st.markdown(
    "SHAP cuantifica la contribucion de cada variable a las predicciones del modelo."
)

FIG_DIR = Path(__file__).parent.parent.parent / "reports" / "figures"


importancia = pd.DataFrame({
    "Feature": [
        "CLIENTES_PRESTAMO","PRESTAMO","TASA_PROM","CLIENTES_NUEVOS",
        "INCREMENTO_CARTERA","N_SEMANAS_OBS","CLIENTES","DESEMBOLSO_CLIENTES_NUEVOS",
        "GRUPOS","EDAD","REGION","MOTIVO_BAJA","DIA_PAGO","ESTADO_CIVIL",
        "RANGO_CICLO","EXP_MICROFINANZAS","NIVEL_ESTUDIOS","TIPO_BAJA",
        "PUESTO","SEXO","AREA","PRODUCTO",
    ],
    "SHAP_mean": [
        0.976,0.329,0.317,0.257,0.252,0.242,0.175,0.160,
        0.156,0.110,0.098,0.094,0.075,0.056,
        0.052,0.036,0.030,0.027,0.024,0.020,0.009,0.000,
    ],
}).sort_values("SHAP_mean", ascending=True)

st.subheader("Importancia global -- todas las clases")
fig = px.bar(
    importancia, x="SHAP_mean", y="Feature",
    orientation="h",
    title="SHAP -- Importancia Global (mean |SHAP value|)",
    color="SHAP_mean", color_continuous_scale="Blues",
)
fig.update_layout(plot_bgcolor="white", showlegend=False, height=600)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Beeswarm SHAP por clase")
clase_sel = st.selectbox("Clase a analizar:", ["Q1_BAJO", "Q4_ALTO"])
img_path = FIG_DIR / f"09_shap_beeswarm_{clase_sel}.png"
if img_path.exists():
    st.image(Image.open(img_path), use_column_width=True)
    if clase_sel == "Q1_BAJO":
        st.info("Q1_BAJO: CLIENTES_PRESTAMO alto impulsa hacia bajo riesgo.")
    else:
        st.info("Q4_ALTO: CLIENTES_PRESTAMO bajo y TASA_PROM alta aumentan el riesgo.")
else:
    st.warning(f"Imagen no encontrada: {img_path}")

st.subheader("Dependence plots -- Top 3 variables (impacto sobre Q4_ALTO)")
col1, col2, col3 = st.columns(3)
plots = [
    ("10_shap_dependence_1_CLIENTES_PRESTAMO.png", "CLIENTES_PRESTAMO"),
    ("10_shap_dependence_2_PRESTAMO.png", "PRESTAMO"),
    ("10_shap_dependence_3_TASA_PROM.png", "TASA_PROM"),
]
for col, (fname, nombre) in zip([col1, col2, col3], plots):
    with col:
        st.markdown(f"**{nombre}**")
        p = FIG_DIR / fname
        if p.exists():
            st.image(Image.open(p), use_column_width=True)
        else:
            st.warning("Imagen no encontrada")
