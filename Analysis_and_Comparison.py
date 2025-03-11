import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
# from PINN_Method import FCN

# Load Deflection Data
analytical = pd.read_csv("analytical_def.csv")
fem = pd.read_csv("fem_deflection.csv")
pinn = pd.read_csv("pinn_deflection.csv")

# Plot Comparison
plt.figure(figsize=(8, 5))
# Highlight the beam axis in bold
plt.plot([0, 1], [0, 0], color="black", linewidth=4, label="Beam Axis")
plt.axvline(1/2, linestyle="dashed", color="gray", label="Midspan (Max Deflection)")
plt.plot(analytical["Position (m)"], analytical["Deflection (mm)"], label="Analytical", color="b", linewidth=2)
plt.plot(fem["Position (m)"], fem["Deflection (mm)"], label="FEM", color="r", linewidth=2)
plt.plot(pinn["Position (m)"], pinn["Deflection (mm)"]*1000, label="PINN", color="g", linestyle="dashed", linewidth=2)
plt.xlabel("Position along Beam (m)")
plt.ylabel("Deflection (mm)")
plt.title("Comparison of Beam Deflection Methods")
plt.legend()
plt.grid(True)
plt.savefig("Analytical-FEM-PINN-Compare_plot.png")  # Save deflection plot
plt.show()
