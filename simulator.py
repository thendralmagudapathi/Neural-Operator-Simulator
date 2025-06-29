import torch
import numpy as np
from models import load_model


def create_hotspot(shape=(64, 64), loc=(32, 32), intensity=1.0, radius=3):
    x = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i - loc[0])**2 + (j - loc[1])**2 <= radius**2:
                x[i, j] = intensity
    return x


def simulate(model_type, hotspot, timesteps=20):
    print("Model Input")
    model = load_model(model_type)
    input_seq = torch.tensor(hotspot)[None, None, ...].repeat(1, timesteps, 1, 1)  # (B, T, H, W)
    with torch.no_grad():
        output_seq = model(input_seq)
    return output_seq.squeeze().numpy()  # Shape: (T, H, W)





"""
Output value stats:

Min: 0.050115284

Max: 0.058853433

Mean: 0.058774374

Std: 0.00073018076"""