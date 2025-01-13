import streamlit as st
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Simulador de neurona", page_icon="🧠")

# Imagen cabecera
st.image("./images/image.png", use_container_width=True)

# Título de la página
st.title("Simulador de neurona 🧠")

st.markdown("## X entradas con bias (sesgo)")

# Slider para número de entradas y pesos
n_inputs = st.slider("Número de entradas / pesos", 1, 8, 3)

st.markdown("### Entradas")
cols = st.columns(n_inputs)
inputs = []  
for i in range(n_inputs):
    with cols[i]:
        input_val = st.number_input(f"x{i+1}", key=f"x{i}")
        inputs.append(input_val)
        
st.markdown("### Pesos")
cols = st.columns(n_inputs)
weights = []
for i in range(n_inputs):
    with cols[i]:       
        weight = st.slider(f"w{i+1}", -10.0, 10.0, 0.0, key=f"w{i}")
        weights.append(weight)

b = st.slider("Bias", -10.0, 10.0, 0.0, key="b")

y = sum(w * x for w, x in zip(weights, inputs)) + b
y = round(y, 2)

# Mostrar los arrays de inputs y weights en una tabla
data = {
    "Entrada (x)": inputs,
    "Peso (w)": weights
}
df = pd.DataFrame(data)
df.index = [""] * len(df)  # Ocultar la columna de índices
st.table(df)

# Construir la fórmula en LaTeX
latex_formula = "y = " + " + ".join([f"w_{i} \\cdot x_{i}" for i in range(n_inputs)]) + " + b"
st.latex(latex_formula)

# Mostrar la salida de la neurona de una manera más vistosa
st.metric(label="Salida de la neurona (y)", value=y)
st.success(f"La salida de la neurona es: {y}")
