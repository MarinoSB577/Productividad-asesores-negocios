import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Comparacion de Modelos", layout="wide")
st.title("Comparacion de Modelos")

metricas = pd.DataFrame({
    "Modelo"  : ["Logistic Regression", "LightGBM", "XGBoost"],
    "Macro F1": [0.5696, 0.6144, 0.5700],
    "Kappa"   : [0.4248, 0.4813, 0.4282],
    "AUC OvR" : [0.8079, 0.8380, 0.8303],
    "Accuracy": [0.5686, 0.6110, 0.5711],
})

st.subheader("Tabla comparativa")
st.markdown("**Metrica principal: Macro F1** -- trata todas las clases por igual.")

def highlight_best(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in ["Macro F1", "Kappa", "AUC OvR", "Accuracy"]:
        idx_best = df[col].idxmax()
        styles.loc[idx_best, col] = "background-color: #d4edda; font-weight: bold"
    return styles

st.dataframe(
    metricas.style.apply(highlight_best, axis=None).format({
        "Macro F1": "{:.4f}", "Kappa": "{:.4f}",
        "AUC OvR": "{:.4f}", "Accuracy": "{:.4f}",
    }),
    use_container_width=True,
)

st.subheader("Comparacion visual")
metrica_sel = st.selectbox(
    "Selecciona metrica:",
    ["Macro F1", "Kappa", "AUC OvR", "Accuracy"],
    index=0,
)

fig = px.bar(
    metricas, x="Modelo", y=metrica_sel,
    color="Modelo",
    color_discrete_map={
        "Logistic Regression": "#ADB5BD",
        "LightGBM": "#4C72B0",
        "XGBoost": "#E8694A",
    },
    title=f"{metrica_sel} por modelo",
    text=metrica_sel,
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig.update_layout(showlegend=False, plot_bgcolor="white", yaxis_range=[0.4, 0.7])
st.plotly_chart(fig, use_container_width=True)

st.subheader("F1 por clase -- LightGBM (modelo final)")
f1_clase = pd.DataFrame({
    "Clase": ["Q1_BAJO", "Q2_MEDIO_BAJO", "Q3_MEDIO_ALTO", "Q4_ALTO"],
    "F1"   : [0.7292, 0.5000, 0.4670, 0.7614],
})
fig2 = px.bar(
    f1_clase, x="Clase", y="F1",
    color="F1", color_continuous_scale="Blues",
    title="F1 por clase -- LightGBM",
    text="F1",
)
fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig2.update_layout(plot_bgcolor="white", yaxis_range=[0, 0.9], showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

st.info(
    "Los cuartiles extremos (Q1 y Q4) se identifican con mayor precision (F1 > 0.73). "
    "Los cuartiles intermedios tienen frontera difusa -- esperado en clasificacion ordinal."
)

st.subheader("Matriz de confusion normalizada")
cm = np.array([
    [0.70, 0.21, 0.06, 0.03],
    [0.14, 0.54, 0.25, 0.07],
    [0.06, 0.37, 0.46, 0.12],
    [0.02, 0.04, 0.19, 0.75],
])
clases = ["Q1_BAJO", "Q2_MEDIO_BAJO", "Q3_MEDIO_ALTO", "Q4_ALTO"]
fig3 = go.Figure(data=go.Heatmap(
    z=cm, x=clases, y=clases,
    colorscale="Blues",
    text=np.round(cm, 2),
    texttemplate="%{text}",
))
fig3.update_layout(
    title="Matriz de Confusion Normalizada",
    xaxis_title="Predicho", yaxis_title="Real",
    yaxis_autorange="reversed",
)
st.plotly_chart(fig3, use_container_width=True)
