import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thermal conductivity (alpha) for materials
material_alpha = {
    "steel": 0.0001,
    "aluminum": 0.0003,
    "copper": 0.0005
}

# Convert material to numeric ID
def get_alpha(material):
    return torch.tensor([material_alpha[material]], dtype=torch.float32).to(device)

# Neural Network
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def physics_loss(model, xyt_hm, alpha):
    xyt_hm.requires_grad = True
    u = model(xyt_hm)

    grads = torch.autograd.grad(u, xyt_hm, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y, u_t = grads[:,0], grads[:,1], grads[:,2]

    # Second-order derivatives
    u_xx = torch.autograd.grad(u_x, xyt_hm, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0]
    u_yy = torch.autograd.grad(u_y, xyt_hm, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:,1]

    # Heat equation: u_t = α * (u_xx + u_yy)
    residual = u_t - alpha * (u_xx + u_yy)
    return torch.mean(residual**2)

def generate_heat_distribution(hx, hy, alpha, grid_size=100, time=0.5):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Analytical Gaussian heat kernel
    dist_sq = (X - hx)**2 + (Y - hy)**2
    temp = np.exp(-dist_sq / (4 * alpha * time)) / (4 * np.pi * alpha * time)
    
    return temp  # shape: [100, 100]

inputs = []
targets = []

num_samples = 10000
grid_size = 100
alpha = 0.0001
time = 0.5

for _ in range(num_samples):
    hx = np.random.uniform(0, 1)
    hy = np.random.uniform(0, 1)
    
    temp_grid = generate_heat_distribution(hx, hy, alpha, grid_size, time)

    # Now flatten inputs: every (x, y) point becomes one sample
    x_vals = np.linspace(0, 1, grid_size)
    y_vals = np.linspace(0, 1, grid_size)
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            inputs.append([x, y, time, hx, hy, alpha, 1.0])
            targets.append(temp_grid[j, i])  # note: j=row, i=col


inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=512, shuffle=True)


model = PINN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def sample_batch(N, material):
    x = torch.rand(N, 1).to(device)
    y = torch.rand(N, 1).to(device)
    t = torch.rand(N, 1).to(device)

    # Hotspot location (fixed or sampled)
    hx = torch.full_like(x, 0.5)  # fixed or sampled
    hy = torch.full_like(y, 0.5)

    alpha = get_alpha(material)

    mat_id = torch.full_like(x, material_alpha[material])  # just to feed something
    input = torch.cat([x, y, t, hx, hy, mat_id, torch.ones_like(x)], dim=1)
    return input, alpha

epochs = 20
print("T^raining...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_inputs, batch_targets in loader:
        optimizer.zero_grad()

        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")
    torch.save(model.state_dict(), "./app/models/heat_predictor.pt")



# C:\Users\HP\Desktop\Next\Neural Operator Simulation\app\models

# Inference
model.eval()

# Example hotspot input from user
user_hotspot = (0.5, 0.5)  # (hx, hy)

# You can also use input() in a real app
# hx = float(input("Enter hotspot x (0-1): "))
# hy = float(input("Enter hotspot y (0-1): "))

hx, hy = user_hotspot

# Grid to predict temperature at time t = 0.5
x = torch.linspace(0, 1, 100)
y = torch.linspace(0, 1, 100)
X, Y = torch.meshgrid(x, y, indexing="ij")

material = "steel"
alpha = material_alpha[material]

# Construct input tensor for the network
xyt_hm = torch.cat([
    X.reshape(-1, 1),
    Y.reshape(-1, 1),
    torch.full((10000, 1), 0.5),             # t
    torch.full((10000, 1), hx),              # hx from user
    torch.full((10000, 1), hy),              # hy from user
    torch.full((10000, 1), alpha),           # material
    torch.ones((10000, 1))                   # constant dummy
], dim=1).to(device)


with torch.no_grad():
    u_pred = model(xyt_hm).cpu().numpy().reshape(100, 100)

plt.imshow(u_pred, cmap="hot", extent=[0,1,0,1])
plt.scatter([hx], [hy], color="blue", label="Hotspot")
plt.colorbar(label="Temperature")
plt.title(f"Predicted Heat (hotspot at {hx}, {hy})")
plt.legend()
plt.show()
