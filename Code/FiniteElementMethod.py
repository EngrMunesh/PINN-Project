import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Beam Parameters
L = 1.0  # Beam length (m)
q0 = 1.0  # UDL in N/m
E = 1.0  # Elastic modulus in Pa
I = 1.0  # Moment of inertia in m^4

# Compute Deflection Coefficient
c1 = (0.013071 * q0 * L**4) / (E * I)

# Compute Deflection
x_values = np.linspace(0, L, 100)
deflection_values = c1 * np.sin(np.pi * x_values / L) * 1000  # Convert to mm

# Save Results
df_deflection = pd.DataFrame({"Position (m)": x_values, "Deflection (mm)": deflection_values})
df_deflection.to_csv("fem_deflection.csv", index=False)

# Print Maximum Deflection
print(f"Maximum deflection (mm): {np.max(deflection_values):.4f}")

# Plot Deflection Curve
plt.figure(figsize=(8, 5))
plt.plot(x_values, deflection_values, label="FEM Solution", color="r", linewidth=2)
# Highlight the beam axis in bold
plt.plot([0, L], [0, 0], color="black", linewidth=4, label="Beam Axis")
plt.axvline(L/2, linestyle="dashed", color="gray", label="Midspan (Max Deflection)")
plt.xlabel("Position along Beam (m)")
plt.ylabel("Deflection (mm)")
plt.title("Beam Deflection using Finite Element Method")
plt.legend()
plt.grid(True)
plt.show()


