import streamlit as st
import numpy as np
from materials import MATERIAL_PROPERTIES
from simulator import create_hotspot, simulate
from utils import render_heatmap_grid

st.set_page_config(page_title="Neural Operator Simulator", layout="left")
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