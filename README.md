# Physics-Informed Neural Network (PINN) for Beam Deflection

This project implements a Physics-Informed Neural Network (PINN) to solve the Euler-Bernoulli Beam Equation for a simply supported beam under a uniformly distributed load. The PINN solution is compared with the Finite Element Method (FEM) and the Direct Integration Method (Analytical Solution).

## Project Overview

The goal of this project is to approximate the beam deflection under a given load using three different methods:

- Direct Integration Method (Analytical Solution)
- Finite Element Method (FEM)
- Physics-Informed Neural Network (PINN)

The results of the PINN model are validated against the analytical and FEM solutions.

## Problem Formulation

The beam deflection is governed by the Euler-Bernoulli beam equation:

\[
\frac{d^4 w}{dx^4} = \frac{q}{EI}
\]

where:
- \( w(x) \) is the beam deflection
- \( q \) is the uniformly distributed load (N/m)
- \( E \) is the Young’s modulus (Pa)
- \( I \) is the moment of inertia (m⁴)
- \( L \) is the beam length (m)

### Boundary Conditions
For a simply supported beam:
- \( w(0) = 0 \) and \( w(L) = 0 \) (no displacement at the supports)
- \( \frac{d^2w}{dx^2} \big|_{x=0} = 0 \) and \( \frac{d^2w}{dx^2} \big|_{x=L} = 0 \) (no bending moment at the supports)

## Installation and Setup

### 1. Clone the Repository
```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/PINN_Beam_Deflection.git
cd PINN_Beam_Deflection
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the PINN Model
Train the Physics-Informed Neural Network (PINN):
```sh
python src/PINN_Method.py
```

### 4. Run the Comparison Analysis
Compare the PINN, FEM, and Analytical solutions:
```sh
python src/Analysis_and_Comparison.py
```

## Project Structure

```
PINN_Beam_Deflection/
│── src/                         # Source Code
│   ├── DirectIntegrationMethod.py  # Analytical Solution
│   ├── FiniteElementMethod.py      # FEM Solution
│   ├── PINN_Method.py              # PINN Model Training
│   ├── Analysis_and_Comparison.py  # Code to Compare All Methods
│── models/                      # Saved Model Files
│   ├── pinn_full_model.pth         # Saved PINN Model
│── results/                      # Output Data and Plots
│   ├── pinn_deflection.csv         # PINN Deflection Data
│   ├── training_loss.png           # Loss Plot
│   ├── deflection_plot.png         # Deflection Plot
│── README.md                    # Project Documentation
│── requirements.txt              # Dependencies
│── .gitignore                    # Ignore unnecessary files
│── LICENSE                       # License for Open-Source Sharing (Optional)
```

## Methods Used

### Direct Integration (Analytical Solution)
- Solves the Euler-Bernoulli equation symbolically.
- Provides an exact reference solution.

### Finite Element Method (FEM)
- Discretizes the beam into elements and applies numerical methods.
- Used for structural analysis in engineering.

### Physics-Informed Neural Network (PINN)
- Uses deep learning to approximate the beam deflection.
- Encodes physical laws (Euler-Bernoulli equation) as part of the loss function.

## Results and Visualizations

### Beam Deflection Curve
The following plot compares the deflection computed by PINN, FEM, and the analytical solution:

![Deflection Curve](results/deflection_plot.png)

### Training Loss
The PINN training loss over epochs:

![Training Loss](results/training_loss.png)

## References
- Raissi, Perdikaris, & Karniadakis (2019) - Physics-Informed Neural Networks
- Timoshenko - Theory of Elasticity
- Bathe - Finite Element Analysis
