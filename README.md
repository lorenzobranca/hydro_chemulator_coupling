# Chemical Emulators for Astrophysical Hydrodynamics

## Overview
This project explores the integration of **trained chemical emulators** into **astrophysical hydrodynamics simulations**. The goal is to replace expensive on-the-fly chemical network solvers with **machine-learning-based surrogates**, significantly reducing computational cost while maintaining accuracy.

This repository contains **two toy models** demonstrating this approach:
1. A **standard chemical evolution model** in a simplified 1D hydrodynamic setup.
2. A **stiff chemical system model**, incorporating a more complex reaction network with highly varying timescales.

In both cases, a **DeepONet-based emulator** predicts the evolution of chemical species, and its performance is compared to a traditional ODE solver.

## Key Features
- **DeepONet-based chemical evolution**: Uses a trained DeepONet model to predict species evolution.
- **Reference ODE solution**: Compares the emulator's output with a standard numerical solver.
- **Hydrodynamic framework**: Simple 1D advection scheme to test coupling strategies.
- **Stiff and non-stiff chemical networks**: Includes a model designed for stiff chemical systems.
- **Visualization & animation**: Generates side-by-side comparisons of emulated vs. true chemical evolution.

## Structure of the Code

### 1. **Loading the Trained DeepONet Models**
- The standard model (`model_3species_td_multi/model-10000.ckpt`) is loaded and compiled.
- The stiff model (`model_3species_td_multi_stiff/model-10000.ckpt`) is also available for testing highly stiff reactions.
- Both models are trained to map initial chemical conditions to their future states.

### 2. **Hydrodynamic Setup**
- Defines a **1D domain** with `nx = 100` grid points.
- Basic fluid properties: density, velocity, and pressure.
- A simple **explicit advection scheme** is used for density and velocity.

### 3. **Chemical Evolution**
- **DeepONet predicts** the chemical evolution for each grid cell.
- **Reference solution** computed using `solve_ivp` (Runge-Kutta 45 method).
- **Time-dependent reaction rates** to introduce non-trivial behavior.
- **Stiff system modeling** using highly varying reaction rates, similar to the Robertson problem.

### 4. **Time Integration & Coupling**
- Hydrodynamic and chemical updates are performed at each time step.
- DeepONet replaces an explicit ODE solver in evolving chemical species.
- Snapshots are stored every 10 steps for visualization.

### 5. **Visualization & Output**
- Side-by-side comparison of DeepONet predictions vs. ODE solver.
- The results are animated and saved as `hydro_plus_chem.gif` and `hydro_plus_chem_stiff.gif`.

## How to Run the Toy Models
### **Requirements**
Ensure you have the following dependencies installed:
```bash
pip install numpy matplotlib scipy deepxde
```

### **Running the Standard Simulation**
Execute the script:
```bash
python hydro_chem_cupled.py
```
This will run the time evolution and generate an animation of species evolution.

### **Running the Stiff System Simulation**
Execute the script:
```bash
python hydro_chem_coupled_stiff.py
```
This will run the stiff chemical system and generate its respective animation.

## Future Directions
These toy models serve as **proofs of concept** for integrating ML-based chemical emulators into astrophysical simulations. The next steps include:
- **Extending to full astrophysical hydro codes** (e.g., RAMSES, FLASH, ATHENA++, AREPO).
- **Increasing chemical complexity** (more species and reactions).
- **Handling multi-dimensional hydro simulations** (2D, 3D flows).
- **Exploring alternative ML architectures** (Fourier Neural Operators, Transformer-based surrogates).
- **Refining the emulator for stiff systems** to better capture astrophysical chemistry.

## Contact
For questions or collaborations, reach out at `lorenzo.branca@iwr.uni-heidelberg.de`.

---
This repository is part of an ongoing research project on coupling trained chemical emulators with astrophysical hydro simulations.


