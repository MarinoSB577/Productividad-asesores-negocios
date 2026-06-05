import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Explorador SHAP", layout="wide")
st.title("Explorador SHAP -- Importancia de Variables")
st.markdown(
    "SHAP cuantifica la contribucion de cada variable a las predicciones del modelo. "
    "Valores calculados sobre el conjunto de test (401 asesores)."
)

FIG_DIR = Path(__file__).parent.parent.parent / "reports" / "figures"

importancia = pd.DataFrame({
    "Feature": [
        "TASA_DESEMBOLSO", "TASA_PROM", "N_SEMANAS_OBS", "INCREMENTO_CARTERA",
        "CLIENTES", "CLIENTES_PRESTAMO_EVENTO", "GRUPOS", "ESTADO_CIVIL",
        "MONTO_DESEMBOLSO_EVENTO", "CLIENTES_NUEVOS_EVENTO", "REGION",
        "EXP_MICROFINANZAS", "EDAD", "MOTIVO_BAJA", "TIPO_BAJA",
        "DIA_PAGO", "NIVEL_ESTUDIOS", "SEXO", "RANGO_CICLO", "AREA",
        "PUESTO", "PRODUCTO",
    ],
    "SHAP_mean": [
        0.935, 0.223, 0.183, 0.136, 0.106, 0.091, 0.082, 0.070,
        0.066, 0.060, 0.047, 0.038, 0.033, 0.029, 0.023,
        0.022, 0.017, 0.017, 0.016, 0.013, 0.006, 0.000,
    ],
}).sort_values("SHAP_mean", ascending=True)

st.subheader("Importancia global -- todas las clases")
st.caption(
    "TASA_DESEMBOLSO domina con importancia 0.935 -- mas de 4x la segunda variable. "
    "Mide la fraccion de filas-grupo con desembolso activo."
)

fig = px.bar(
    importancia, x="SHAP_mean", y="Feature",
    orientation="h",
    title="SHAP -- Importancia Global (mean |SHAP value|)",
    color="SHAP_mean", color_continuous_scale="Blues",
)
fig.update_layout(plot_bgcolor="white", showlegend=False, height=650)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Beeswarm SHAP por clase")
clase_sel = st.selectbox("Clase a analizar:", ["ALTO", "BAJO", "MEDIO"])
img_path = FIG_DIR / f"09_shap_beeswarm_{clase_sel}_v2.png"
if img_path.exists():
    st.image(Image.open(img_path), use_container_width=True)
    descripciones = {
        "ALTO" : "TASA_DESEMBOLSO baja impulsa hacia ALTO -- asesores con grupos que no renuevan.",
        "BAJO" : "TASA_DESEMBOLSO alta impulsa hacia BAJO -- asesores con renovacion activa.",
        "MEDIO": "Frontera difusa -- multiples variables contribuyen sin una dominante clara.",
    }
    st.info(descripciones[clase_sel])
else:
    st.warning(f"Imagen no encontrada: {img_path}")

st.subheader("Dependence plots -- Top 3 variables (impacto sobre clase ALTO)")
col1, col2, col3 = st.columns(3)
plots = [
    ("10_shap_dependence_1_TASA_DESEMBOLSO_v2.png", "TASA_DESEMBOLSO"),
    ("10_shap_dependence_2_TASA_PROM_v2.png", "TASA_PROM"),
    ("10_shap_dependence_3_N_SEMANAS_OBS_v2.png", "N_SEMANAS_OBS"),
]
for col, (fname, nombre) in zip([col1, col2, col3], plots):
    with col:
        st.markdown(f"**{nombre}**")
        p = FIG_DIR / fname
        if p.exists():
            st.image(Image.open(p), use_container_width=True)
        else:
            st.warning("Imagen no encontrada")
