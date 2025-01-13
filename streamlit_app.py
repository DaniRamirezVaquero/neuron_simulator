import streamlit as st
import pandas as pd
from neuron import Neuron, ActivationFunction

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
        weight =st.number_input(f"w{i+1}", key=f"w{i}")
        weights.append(weight)
        
colBias, colFunc = st.columns(2)

st.divider()

with colBias:
    b = st.number_input("Sesgo(bias)", key="b")
    
with colFunc:
    func = st.selectbox("Función de activación", list(ActivationFunction))

# Crear la neurona
neuron = Neuron(weights, b, ActivationFunction(func))

# Calcular la salida de la neurona
y = neuron.run(inputs)

# Mostrar los arrays de inputs y weights en una tabla
data = {
    "Entrada (x)": inputs,
    "Peso (w)": weights
}
df = pd.DataFrame(data)
st.table(df)

# Construir la fórmula en LaTeX
latex_formula = "y = " + " + ".join([f"w_{i} \\cdot x_{i}" for i in range(n_inputs)]) + " + b"
st.latex(latex_formula)

# Mostrar la salida de la neurona de una manera más vistosa
st.metric(label="Salida de la neurona (y)", value=y)
