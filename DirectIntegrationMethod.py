import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Beam Parameters
L = 1.0  # Length of beam (m)
q = 1.0  # UDL in N/m
E = 1.0  # Elastic modulus in Pa
I = 1.0  # Moment of inertia in m^4

# Function for Deflection
def deflection(x, L, q, E, I):
    return -(1 / (E * I)) * ((q * L * x**3) / 12 - (q * x**4) / 24 - (q * L**3 * x) / 24)

# Compute Deflection
x_values = np.linspace(0, L, 100)
y_values = deflection(x_values, L, q, E, I)*1000  # Convert to mm

# Save Results
df_deflection = pd.DataFrame({"Position (m)": x_values, "Deflection (mm)": y_values})
df_deflection.to_csv("analytical_def.csv", index=False)

# Print Maximum Deflection
print(f"Maximum deflection (mm): {np.max(y_values):.4f}")

# Plot Deflection Curve
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label="Analytical Solution", color="b", linewidth=2)
# Highlight the beam axis in bold
plt.plot([0, L], [0, 0], color="black", linewidth=4, label="Beam Axis")
plt.axvline(L/2, linestyle="dashed", color="gray", label="Midspan (Max Deflection)")
plt.xlabel("Position along Beam (m)")
plt.ylabel("Deflection (mm)")
plt.title("Beam Deflection using Analytical Method")
plt.legend()
plt.grid(True)
plt.show()