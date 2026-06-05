import streamlit as st

st.set_page_config(
    page_title="Riesgo de Cartera",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Clasificacion de Riesgo de Cartera -- Asesores de Negocios")
st.markdown("""
Sistema de clasificacion de riesgo basado en LightGBM entrenado sobre datos
de una institucion microfinanciera mexicana (2016-2019).

Navega por las paginas del menu lateral para explorar los resultados.

---

**Aviso:** Este dashboard usa datos sinteticos anonimizados con fines demostrativos.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Macro F1 (LightGBM)", "0.6144", "+4.5pp vs baseline")
with col2:
    st.metric("Asesores analizados", "2,004", "periodo 2016-2019")
with col3:
    st.metric("Fairness gap (SEXO)", "2.3pp", "< umbral 10pp")