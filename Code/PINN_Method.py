import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  # For progress bar

# Exact Solution of differential equation
def exact_solution(q, L, E, I, x):
    stiffness = E * I
    A = q / (24 * stiffness)
    w = A * (x**4 - 2 * L * x**3 + L**3 * x)
    return w

# Define PINN Model
class FCN(nn.Module):
    """Define standard fully connected neural network in PyTorch"""
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh()
        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation)
        self.fch = nn.Sequential(*[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

# Set Device for Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

# Define neural network to train
pinn = FCN(1, 1, 32, 3).to(device)

# Define boundary points for the boundary loss
x_boundary1 = torch.tensor([[0.]], requires_grad=True).to(device)
x_boundary2 = torch.tensor([[1.]], requires_grad=True).to(device)
x_boundary3 = torch.tensor([[0.5]], requires_grad=True).to(device)

# Define training points over the entire domain for the physics loss
x_physics = torch.linspace(0, 1, 300).view(-1, 1).to(device).requires_grad_(True)

# Train the neural network
E, I = 1, 1
L, q = 1, 1
x_test = torch.linspace(0, 1, 300).view(-1, 1).to(device)
u_exact = exact_solution(q, L, E, I, x_test).to(device)

# Define Optimizer
optimiser = torch.optim.AdamW(pinn.parameters(), lr=1e-3, weight_decay=1e-4)

# Training Configuration
total_epochs = 20001
progress_interval = 500  # Update every 500 epochs
loss_history = []

# Training Loop with Progress Bar
with tqdm(total=total_epochs, desc="Training Progress", ncols=100) as pbar:
    for i in range(1, total_epochs + 1):
        optimiser.zero_grad()

        # Compute each term of the PINN loss function
        lambda1, lambda2, lambda3 = 1e-1, 1e-1, 1e-4

        # Compute boundary loss
        u1, u2, u3 = pinn(x_boundary1), pinn(x_boundary2), pinn(x_boundary3)
        loss1 = (torch.squeeze(u1) - 0) ** 2 + (torch.squeeze(u2) - 0) ** 2  

        dudt_1 = torch.autograd.grad(u1, x_boundary1, torch.ones_like(u1), create_graph=True)[0]
        dudt_2 = torch.autograd.grad(u2, x_boundary2, torch.ones_like(u2), create_graph=True)[0]

        du2dt2_1 = torch.autograd.grad(dudt_1, x_boundary1, torch.ones_like(dudt_1), create_graph=True)[0]
        du2dt2_2 = torch.autograd.grad(dudt_2, x_boundary2, torch.ones_like(dudt_2), create_graph=True)[0]

        loss2 = ((torch.squeeze(du2dt2_1) - 0) ** 2 + (torch.squeeze(du2dt2_2) - 0) ** 2) / 2  

        dudt_3 = torch.autograd.grad(u3, x_boundary3, torch.ones_like(u3), create_graph=True)[0]
        loss3 = (torch.squeeze(dudt_3) - 0) ** 2  

        # Compute physics loss
        u = pinn(x_physics)

        dudt = torch.autograd.grad(u, x_physics, torch.ones_like(u), create_graph=True)[0]
        du2dt2 = torch.autograd.grad(dudt, x_physics, torch.ones_like(dudt), create_graph=True)[0]
        du3dt3 = torch.autograd.grad(du2dt2, x_physics, torch.ones_like(du2dt2), create_graph=True)[0]
        du4dt4 = torch.autograd.grad(du3dt3, x_physics, torch.ones_like(du3dt3), create_graph=True)[0]

        loss4 = torch.mean((du4dt4 - q / (E * I)) ** 2)

        # Backpropagate joint loss and take optimizer step
        loss = 2 * loss1 + lambda1 * loss2 + lambda2 * loss3 + lambda3 * loss4
        loss.backward()
        optimiser.step()

        # Save loss history
        if i % progress_interval == 0:
            loss_history.append(loss.item())
            pbar.update(progress_interval)
            print(f"Epoch {i}: Loss = {loss.item():.6f}")

# Save the Entire Model
torch.save(pinn, "pinn_full_model.pth")
print("Model saved successfully.")

# Compute Deflection (After Training)
pinn.eval()
u_pinn = pinn(x_test).detach().cpu().numpy()

# Save Results to CSV
df_deflection = pd.DataFrame({"Position (m)": x_test.cpu().numpy().flatten(), "Deflection (mm)": u_pinn.flatten()})
df_deflection.to_csv("pinn_deflection.csv", index=False)

# Plot and Save Training Loss Curve
plt.figure(figsize=(6, 4))
plt.plot(range(progress_interval, total_epochs+1, progress_interval), loss_history, marker='o', linestyle='-', color='b')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.savefig("training_loss.png")  # Save loss plot
plt.show()

# Plot and Save Deflection Curve
plt.figure(figsize=(6, 4))
plt.plot(x_test.cpu().numpy(), u_exact.cpu().numpy(), label="Exact Solution", color="gray")
plt.plot(x_test.cpu().numpy(), u_pinn, label="PINN Predicted Deflection", color="green")
plt.xlabel("Position along Beam (m)")
plt.ylabel("Deflection (mm)")
plt.title("Beam Deflection Predicted by PINN")
plt.legend()
plt.grid(True)
plt.savefig("deflection_plot.png")  # Save deflection plot
plt.show()

print("Figures saved successfully!")
