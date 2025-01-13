import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Simulador de neurona", page_icon="🧠")

# Imagen cabecera
st.image("./images/image.png", use_container_width=True)

# Título de la página
st.title("Simulador de neurona 🧠")

st.markdown("## X entradas con bias (sesgo)")

st.write("Elige el número de entradas / pesos que tendrá la neurona")

# Slider para número de entradas y pesos
n_inputs = st.slider("Número de entradas", 1, 8, 3)

cols = st.columns(n_inputs)
weights = []
inputs = []

st.markdown("### Entradas")

for i in range(n_inputs):
    with cols[i]:
        input_val = st.number_input(f"x{i+1}", key=f"x{i}")
        inputs.append(input_val)
        
st.write(inputs)
        
st.markdown("### Pesos")
        
for i in range(n_inputs):
    with cols[i]:       
        weight = st.slider(f"w{i+1}", -10.0, 10.0, 0.0, key=f"w{i}")
        weights.append(weight)
        
st.write(weights)
        

b = st.slider("Bias", -10.0, 10.0, 0.0, key="b")

y = sum(w * x for w, x in zip(weights, inputs)) + b
y = round(y, 2)

# Construir la fórmula en LaTeX
latex_formula = "y = " + " + ".join([f"w_{i} \\cdot x_{i}" for i in range(n_inputs)]) + " + b"
st.latex(latex_formula)

st.write("La salida de la neurona es", y)
