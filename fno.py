# fourier Neural Operator FNO for 2D Paritial Differential Equation

import torch
import torch.nn as nn
import torch.fft as fft
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Fno module
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.width = width
        self.fc0 = nn.Linear(1, width)
        self.weights = nn.Parameter(torch.randn(width, width, modes1, modes2))
        self.fc1 = nn.Linear(width, 1)

    def forward(self, x):
        # x: (B, H, W, 1)
        x = self.fc0(x).permute(0,3,1,2)
        x_ft = fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:,:, :self.modes1, :self.modes2] = torch.einsum(
            "bchw,hwnm->bcwnm", x_ft[:,:,:self.modes1,:self.modes2], self.weights)
        x = fft.irfft2(out_ft, s=(x.shape[2], x.shape[3]))
        x = x.permute(0,2,3,1)
        return self.fc1(x)

# dataload
def load_data(path):
    arr = np.load(path)
    inp, out = arr['a'], arr['u']  # assume shape (N,H,W)
    inp = inp[..., None].astype(np.float32)
    out = out[..., None].astype(np.float32)
    return TensorDataset(torch.from_numpy(inp), torch.from_numpy(out))

# data train
def train():
    train_ds = load_data('data/train.npz')
    val_ds = load_data('data/val.npz')
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    vloader = DataLoader(val_ds, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNO2d(modes1=12, modes2=12, width=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0
        for a, u in loader:
            a, u = a.to(device), u.to(device)
            pred = model(a)
            loss = criterion(pred, u)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Train loss = {total_loss/len(loader):.4f}")

        
        #  model internal validation
        model.eval() 
        val_loss = 0.0
        with torch.no_grad():
            for a, u in vloader:
                a, u = a.to(device), u.to(device)
                val_loss += criterion(model(a), u).item()
        print(f"→ Val loss = {val_loss/len(vloader):.4f}")

if __name__ == '__main__':
    train()
# fno_train.py
