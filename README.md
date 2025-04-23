# Physics-Informed Neural Network (PINN) for Beam Deflection

This project implements a **Physics-Informed Neural Network (PINN)** to solve the **Euler-Bernoulli Beam Equation** for a simply supported beam subjected to a uniformly distributed load. The PINN results are compared with those from the **Finite Element Method (FEM)** and the **Analytical Solution (Direct Integration Method)**.

---

## Overview

Three methods are used to evaluate beam deflection:

1. **Analytical Solution** (Direct Integration of the governing equation)  
2. **Finite Element Method (FEM)** (Numerical approximation using discretized elements)  
3. **Physics-Informed Neural Network (PINN)** (Deep learning model guided by physics)

The PINN approach integrates the governing differential equation and boundary conditions into the training process to learn an accurate approximation of beam deflection.

---

## Governing Equation

The deflection ( w(x) ) of a simply supported beam under a uniformly distributed load ( q ) is described by the Euler-Bernoulli beam equation:
[\frac{d^4 w}{dx^4} = \frac{q}{EI}]
Where:

( w(x) ): Beam deflection  
( q ): Uniformly distributed load (N/m)  
( E ): Young’s modulus (Pa)  
( I ): Moment of inertia (m⁴)  
( L ): Length of the beam (m)

Boundary Conditions:

Displacement: ( w(0) = 0 ), ( w(L) = 0 )  
Moment-free ends: ( \frac{d^2 w}{dx^2}\bigg|{x=0} = 0 ), ( \frac{d^2 w}{dx^2}\bigg|{x=L} = 0 )

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/PINN_Beam_Deflection.git
   cd PINN_Beam_Deflection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the PINN Model**
   ```bash
   python src/PINN_Method.py
   ```

4. **Perform the Comparison**
   ```bash
   python src/Analysis_and_Comparison.py
   ```

---

## Project Structure

```
PINN_Beam_Deflection/
├── src/
│   ├── DirectIntegrationMethod.py
│   ├── FiniteElementMethod.py
│   ├── PINN_Method.py
│   └── Analysis_and_Comparison.py
├── models/
│   └── pinn_full_model.pth
├── results/
│   ├── pinn_deflection.csv
│   ├── training_loss.png
│   └── deflection_plot.png
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Methods Summary

**Analytical Solution**  
- Solves the beam equation using symbolic integration.  
- Serves as the exact reference.

**Finite Element Method (FEM)**  
- Uses numerical approximation by dividing the beam into finite elements.  
- Common in structural analysis applications.

**Physics-Informed Neural Network (PINN)**  
- Trains a deep neural network with a physics-based loss function.  
- Combines data-driven learning with differential equations.

---

## Results

The following plot shows deflection results obtained using the three methods:

![Deflection Curve](Results/Analytical-FEM-PINN-Compare_plot.png)

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).  
  *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*  
  Journal of Computational Physics, 378, 686–707.  
  [https://doi.org/10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018).  
  *Automatic Differentiation in Machine Learning: a Survey.*  
  Journal of Machine Learning Research, 18, 1–43.  
  [http://jmlr.org/papers/v18/17-468.html](http://jmlr.org/papers/v18/17-468.html)

---
