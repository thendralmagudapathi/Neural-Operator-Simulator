import streamlit as st
import numpy as np
from materials import MATERIAL_PROPERTIES
from simulator import create_hotspot, simulate
from utils import render_heatmap_grid
import matplotlib.pyplot as plt


st.set_page_config(page_title="Neural Operator Simulator", layout="centered")
st.title("Neural Operator Simulator")

# Model Selection
model_type = st.selectbox("Choose Simulation Equation:", ["Heat", "Burgers"])
# Plate Material Selection
material = st.selectbox("Select Plate Material", ["Steel", "Aluminum", "Iron"])

# Hotspot Selection (slider)
x = st.slider("X Coordinate of Hotspot", min_value=0, max_value=63, value=32)
y = st.slider("Y Coordinate of Hotspot", min_value=0, max_value=63, value=32)

if st.button("Run Simulation"):
    st.write(f"Running {model_type} simulation with hotspot at ({x}, {y})")
    hotspot = create_hotspot(loc=(y, x))
    print("Hotspot max", hotspot.max())
    hotspot = hotspot / hotspot.max()  # Normalize to [0, 1]


    # Visualize initial hotspot
    st.subheader("Initial Hotspot")
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(hotspot, cmap='hot', interpolation='nearest')
    ax1.plot(x, y, 'bo')  # Blue dot on hotspot
    st.pyplot(fig1)

    timesteps = 20

    # Run the simulation
    output_seq = simulate(model_type, hotspot, timesteps)  # output_seq: (T, H, W)

    st.write("Output value stats:")
    st.write("Min:", output_seq.min())
    st.write("Max:", output_seq.max())
    st.write("Mean:", output_seq.mean())
    st.write("Std:", output_seq.std())

    # Visualize each timestep (or last one)
    st.subheader("Final Heat Distribution")
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(output_seq[-1], cmap='hot', interpolation='nearest')  # Last timestep
    st.pyplot(fig2)

    # Optional: View all timesteps
    st.subheader("Animation of Heat Diffusion")
    for t in range(timesteps):
        st.write(f"Timestep {t+1}")
        fig, ax = plt.subplots()
        im = ax.imshow(output_seq[t], cmap='hot', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)