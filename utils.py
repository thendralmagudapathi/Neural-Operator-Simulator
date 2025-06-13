import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def render_heatmap_grid(frames):
    fig, ax = plt.subplots()
    img = ax.imshow(frames[0], cmap='hot', interpolation='nearest')
    stframe = st.empty()
    for frame in frames:
        img.set_data(frame)
        ax.set_title("Simulation Time Step")
        stframe.pyplot(fig)
