import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Comparacion de Modelos", layout="wide")
st.title("Comparacion de Modelos")
st.markdown("**Metrica principal: Macro F1** -- trata todas las clases por igual.")
st.markdown("Target: 3 clases (BAJO / MEDIO / ALTO) usando terciles de TDNC.")

metricas = pd.DataFrame({
    "Modelo"  : ["Logistic Regression", "LightGBM", "XGBoost"],
    "Macro F1": [0.7107, 0.7166, 0.7107],
    "Kappa"   : [0.5661, 0.5773, 0.5699],
    "AUC OvR" : [0.8554, 0.8812, 0.8830],
    "Accuracy": [0.7107, 0.7182, 0.7132],
})

st.subheader("Tabla comparativa")

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

st.info(
    "LightGBM mejora 0.6pp sobre LR baseline en macro F1. "
    "La ventaja principal es AUC OvR (0.881 vs 0.855) y mayor estabilidad."
)

st.subheader("Comparacion visual")
metrica_sel = st.selectbox("Selecciona metrica:", ["Macro F1", "Kappa", "AUC OvR", "Accuracy"])

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
fig.update_layout(showlegend=False, plot_bgcolor="white", yaxis_range=[0.6, 0.8])
st.plotly_chart(fig, use_container_width=True)

st.subheader("F1 por clase -- LightGBM (modelo final)")
f1_clase = pd.DataFrame({
    "Clase": ["ALTO", "BAJO", "MEDIO"],
    "F1"   : [0.8015, 0.7407, 0.6077],
    "Precision": [0.7842, 0.7353, 0.6270],
    "Recall"   : [0.8195, 0.7463, 0.5896],
})
fig2 = px.bar(
    f1_clase, x="Clase", y="F1",
    color="F1", color_continuous_scale="Blues",
    title="F1 por clase -- LightGBM v2",
    text="F1",
)
fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig2.update_layout(plot_bgcolor="white", yaxis_range=[0, 0.95], showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

st.info(
    "ALTO y BAJO se identifican con F1 > 0.74. "
    "MEDIO tiene frontera difusa con ambas clases adyacentes (F1=0.61) -- "
    "esperado en clasificacion ordinal de 3 clases."
)

st.subheader("Matriz de confusion normalizada")
cm = np.array([
    [0.82, 0.04, 0.15],
    [0.05, 0.74, 0.20],
    [0.24, 0.32, 0.59],
])
clases = ["ALTO", "BAJO", "MEDIO"]
fig3 = go.Figure(data=go.Heatmap(
    z=cm, x=clases, y=clases,
    colorscale="Blues",
    text=np.round(cm, 2),
    texttemplate="%{text}",
))
fig3.update_layout(
    title="Matriz de Confusion Normalizada (valores = recall por clase)",
    xaxis_title="Predicho", yaxis_title="Real",
    yaxis_autorange="reversed",
)
st.plotly_chart(fig3, use_container_width=True)
