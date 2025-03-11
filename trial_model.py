import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Exact Solution of differential equation
def exact_solution(q, L, E, I, x):
    stiffness = E * I
    A = q / (24 * stiffness)
    B1 = x**4
    B2 = 2 * L * (x**3)
    B3 = x * (L**3)
    w = A * (B1 - B2 + B3)
    return w

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

# Create training loop
torch.manual_seed(123)

# Define neural network to train
pinn = FCN(1, 1, 32, 3)

# Define boundary points for the boundary loss
x_boundary1 = torch.tensor([[0.]], requires_grad=True)
x_boundary2 = torch.tensor([[1.]], requires_grad=True)
x_boundary3 = torch.tensor([[0.5]], requires_grad=True)

# Define training points over the entire domain for the physics loss
x_physics = torch.linspace(0, 1, 300).view(-1, 1).requires_grad_(True)

# Train the neural network
E, I = 1, 1
L, q = 1, 1
x_test = torch.linspace(0, 1, 300).view(-1, 1)
u_exact = exact_solution(q, L, E, I, x_test)

optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)

for i in range(20001):
    optimiser.zero_grad()

    # Compute each term of the PINN loss function
    lambda1, lambda2, lambda3 = 1e-1, 1e-1, 1e-4

    # Compute boundary loss
    u1 = pinn(x_boundary1)
    u2 = pinn(x_boundary2)
    u3 = pinn(x_boundary3)

    loss1 = (torch.squeeze(u1) - 0) ** 2 + (torch.squeeze(u2) - 0) ** 2  # Fixed loss1 calculation

    dudt_1 = torch.autograd.grad(u1, x_boundary1, torch.ones_like(u1), create_graph=True)[0]
    dudt_2 = torch.autograd.grad(u2, x_boundary2, torch.ones_like(u2), create_graph=True)[0]

    du2dt2_1 = torch.autograd.grad(dudt_1, x_boundary1, torch.ones_like(dudt_1), create_graph=True)[0]
    du2dt2_2 = torch.autograd.grad(dudt_2, x_boundary2, torch.ones_like(dudt_2), create_graph=True)[0]

    loss2 = ((torch.squeeze(du2dt2_1) - 0) ** 2 + (torch.squeeze(du2dt2_2) - 0) ** 2) / 2  # Fixed loss2 calculation

    dudt_3 = torch.autograd.grad(u3, x_boundary3, torch.ones_like(u3), create_graph=True)[0]
    loss3 = (torch.squeeze(dudt_3) - 0) ** 2  # Fixed loss3 calculation

    # Compute physics loss
    u = pinn(x_physics)

    dudt = torch.autograd.grad(u, x_physics, torch.ones_like(u), create_graph=True)[0]
    du2dt2 = torch.autograd.grad(dudt, x_physics, torch.ones_like(dudt), create_graph=True)[0]
    du3dt3 = torch.autograd.grad(du2dt2, x_physics, torch.ones_like(du2dt2), create_graph=True)[0]
    du4dt4 = torch.autograd.grad(du3dt3, x_physics, torch.ones_like(du3dt3), create_graph=True)[0]

    loss4 = torch.mean((du4dt4 - q / (E * I)) ** 2)  # Fixed physics loss term

    # Backpropagate joint loss and take optimizer step
    loss = 2*loss1 + lambda1 * loss2 + lambda2 * loss3 + lambda3 * loss4
    loss.backward()
    optimiser.step()

# Define the path to save the model
model_path = "pinn_model.pth"
# Save the entire model
torch.save(pinn, model_path)
print("Model saved successfully.")

# Compute Deflection
u_pinn = pinn(x_test).detach().numpy()

# Save Results
df_deflection = pd.DataFrame({"Position (m)": x_test, "Deflection (mm)": u_pinn})
df_deflection.to_csv("pinn_deflection.csv", index=False)

# Plot the Result
plt.figure(figsize=(6, 2.5))
plt.scatter(x_physics.detach().numpy(), np.zeros_like(x_physics.detach().numpy()), s=20, lw=0, color="tab:green", alpha=0.6)
plt.scatter(x_boundary1.detach().numpy(), np.zeros_like(x_boundary1.detach().numpy()), s=20, lw=0, color="tab:red", alpha=0.6)
plt.scatter(x_boundary2.detach().numpy(), np.zeros_like(x_boundary2.detach().numpy()), s=20, lw=0, color="tab:red", alpha=0.6)
plt.scatter(x_boundary3.detach().numpy(), np.zeros_like(x_boundary3.detach().numpy()), s=20, lw=0, color="tab:red", alpha=0.6)

plt.plot(x_test.numpy(), u_exact, label="Exact solution", color="tab:grey", alpha=0.6)
plt.plot(x_test.numpy(), u_pinn, label="PINN solution", color="tab:green")

plt.title(f"Final Training Result")
plt.legend()
plt.show()