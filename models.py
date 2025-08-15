import torch
import os
import torch.fft
import torch.nn as nn
import deepxde as dde
import numpy as np
from deepxde.backend import tf
from materials import material_alpha



class FNO2D(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=32):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(1, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):  # x shape: (B, T, H, W)
        B, T, H, W = x.shape
        out_seq = []

        for t in range(T):
            xt = x[:, t:t+1, :, :]  # (B, 1, H, W)
            xt = xt.permute(0, 2, 3, 1)  # (B, H, W, 1)
            xt = self.fc0(xt)
            xt = xt.permute(0, 3, 1, 2)  # (B, C, H, W)

            x1 = self.conv0(xt) + self.w0(xt)
            x2 = self.conv1(x1) + self.w1(x1)
            x3 = self.conv2(x2) + self.w2(x2)
            x4 = self.conv3(x3) + self.w3(x3)

            x4 = x4.permute(0, 2, 3, 1)  # (B, H, W, C)
            x4 = self.fc1(x4)
            x4 = torch.relu(x4)
            x4 = self.fc2(x4).squeeze(-1)  # (B, H, W)

            out_seq.append(x4)

        return torch.stack(out_seq, dim=1)  # (B, T, H, W)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def complex_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_channels, H, W//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x

# Dummy FNO-like model (replace with real FNO/DeepONet in production)
class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x * torch.exp(-0.1 * torch.arange(x.shape[1])[None, :, None, None])


MODEL_REGISTRY = {
    "heat": FNO2D(),
    "burgers": DummyModel()
}


def load_model(model_type):
    if model_type.lower() == "heat":
        model = FNO2D(modes1=8, modes2=8, width=16)
        model.load_state_dict(torch.load("models/fno_heat_checkpoint.pt", map_location="cpu"))
        return model.eval()
    elif model_type.lower() == "burgers":
        return DummyModel().eval()

