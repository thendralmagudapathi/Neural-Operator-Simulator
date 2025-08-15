import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import FNO2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = np.load("heat_train_data.npy")  # (N,T,H,W)
inputs = data[:,0:1,:,:]  # (N,1,H,W)
targets = data  # (N,T,H,W)

# Convert to torch tensors
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

# Split
train_inputs = inputs_tensor[:80]
train_targets = targets_tensor[:80]
val_inputs = inputs_tensor[80:]
val_targets = targets_tensor[80:]

# Model
model = FNO2D(modes1=8, modes2=8, width=16).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 10
batch_size = 4

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(train_inputs.size(0))
    epoch_loss = 0.0

    for i in range(0, train_inputs.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_in = train_inputs[indices].to(device)  # (B,1,H,W)
        batch_out = train_targets[indices].to(device)  # (B,T,H,W)

        optimizer.zero_grad()
        pred = model(batch_in.repeat(1,batch_out.shape[1],1,1))  # Repeat input across T

        loss = criterion(pred, batch_out)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(val_inputs.to(device).repeat(1,val_targets.shape[1],1,1))
        val_loss = criterion(val_pred, val_targets.to(device)).item()

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

# Save model
torch.save(model.state_dict(), "fno_heat_checkpoint.pt")
print("Model saved to fno_heat_checkpoint.pt")
