import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import deepxde as dde

# Load the trained DeepONet model
model_path = 'model_3species_td_multi/model-10000.ckpt'
m = 3  # Input dimension of the branch network
dim_x = 1  # Input dimension of the trunk network
activation = "LAAF-5 tanh"
net = dde.nn.DeepONetCartesianProd(
    [m, 60, 60],  # Branch network architecture
    [dim_x, 60, 20],  # Trunk network architecture
    activation,
    "Glorot normal",
    num_outputs=3,
    multi_output_strategy="split_branch"
)
data_dummy = dde.data.Triple(
    X_train=(np.zeros((1, 3)), np.zeros((1, 1))), 
    y_train=np.zeros((1, 1, 3)), 
    X_test=(np.zeros((1, 3)), np.zeros((1, 1))), 
    y_test=np.zeros((1, 1, 3))
)
model = dde.Model(data_dummy, net)
model.compile("adam", lr=0.001)
model.restore(model_path, verbose=1)

# Simulation parameters
nx = 100  # Number of grid points
Lx = 1.0  # Domain length
x = np.linspace(0, Lx, nx)  # Spatial grid

dt = 0.0005  # Time step
total_time = 0.2
nt = int(total_time / dt)

# Initialize hydrodynamic variables
density = np.ones(nx)
velocity = np.zeros(nx)
pressure = np.ones(nx)

density[:nx // 2] = 1.0
pressure[:nx // 2] = 1.0
density[nx // 2:] = 0.125
pressure[nx // 2:] = 0.1

gamma = 1.4
energy = pressure / (gamma - 1) + 0.5 * density * velocity**2

# Initialize chemical species
n_species = np.zeros((nx, 3))
n_species[:, 0] = 0.9 * density
n_species[:, 1] = 0.1 * density
n_species[:, 2] = 0.0 * density

# ODE system for reference solution
def ode_system(t, n):
    n1, n2, n3 = n
    conv = 5e0
    k1, k2, k3 = 0.8 - np.sin(conv * t), 0.5 + np.cos(conv * t), 0.2 + np.sin(2 * t)
    return [
        conv * (- k1 * n1 - k3 * n1 * n3), 
        conv * (k1 * n1 - k2 * n2 + 2. * k3 * n1 * n3), 
        conv * (k2 * n2 - k3 * n1 * n3)
    ]

# Function to evolve chemistry using the trained model
def evolve_chemistry(n_species_cell, dt):
    t_space = np.expand_dims(np.array([0, dt]), axis=-1)
    IC = np.expand_dims(n_species_cell, axis=0)
    x_input = (IC, t_space)
    y_pred = model.predict(x_input)
    return y_pred[0, -1, :]

# HLL Riemann Solver
def hll_flux(density_L, velocity_L, pressure_L, density_R, velocity_R, pressure_R, gamma):
    """ Compute the HLL flux for a 1D Euler system. """
    c_L = np.sqrt(gamma * pressure_L / density_L)
    c_R = np.sqrt(gamma * pressure_R / density_R)
    S_L = min(velocity_L - c_L, velocity_R - c_R)
    S_R = max(velocity_L + c_L, velocity_R + c_R)

    flux_L = np.array([
        density_L * velocity_L, 
        density_L * velocity_L**2 + pressure_L, 
        velocity_L * (pressure_L / (gamma - 1) + 0.5 * density_L * velocity_L**2 + pressure_L)
    ])
    
    flux_R = np.array([
        density_R * velocity_R, 
        density_R * velocity_R**2 + pressure_R, 
        velocity_R * (pressure_R / (gamma - 1) + 0.5 * density_R * velocity_R**2 + pressure_R)
    ])

    if S_L >= 0:
        return flux_L
    elif S_R <= 0:
        return flux_R
    else:
        return (S_R * flux_L - S_L * flux_R + S_L * S_R * (
            np.array([density_R, density_R * velocity_R, pressure_R / (gamma - 1) + 0.5 * density_R * velocity_R**2]) - 
            np.array([density_L, density_L * velocity_L, pressure_L / (gamma - 1) + 0.5 * density_L * velocity_L**2])
        )) / (S_R - S_L)

# Store snapshots for animation
snapshots, reference_snapshots = [], []
density_snapshots, velocity_snapshots, pressure_snapshots = [], [], []

for t in range(nt):
    fluxes = np.zeros((nx - 1, 3))
    n_species[:, 0] = n_species[:,0] / density
    n_species[:, 1] = n_species[:, 1] / density
    n_species[:, 2] = n_species[:,2] / density

    for i in range(nx - 1):
        fluxes[i, :] = hll_flux(
            density[i], velocity[i], pressure[i],
            density[i + 1], velocity[i + 1], pressure[i + 1],
            gamma
        )

    density[1:-1] -= dt * (fluxes[1:, 0] - fluxes[:-1, 0]) / (x[1] - x[0])
    momentum = density * velocity
    momentum[1:-1] -= dt * (fluxes[1:, 1] - fluxes[:-1, 1]) / (x[1] - x[0])
    energy[1:-1] -= dt * (fluxes[1:, 2] - fluxes[:-1, 2]) / (x[1] - x[0])

    velocity = momentum / density
    pressure = (gamma - 1) * (energy - 0.5 * density * velocity**2)

    n_species[:, 0] = n_species[:,0] * density
    n_species[:, 1] = n_species[:, 1] * density
    n_species[:, 2] = n_species[:,2] * density



    reference_state = np.zeros_like(n_species)
    for i in range(nx):
        n_species[i, :] = evolve_chemistry(n_species[i, :], dt)
        sol = solve_ivp(ode_system, [0, dt], n_species[i, :], method='RK45')
        reference_state[i, :] = sol.y[:, -1]

    if t % 10 == 0:
        snapshots.append(n_species.copy())
        reference_snapshots.append(reference_state.copy())
        density_snapshots.append(density.copy())
        velocity_snapshots.append(velocity.copy())
        pressure_snapshots.append(pressure.copy())

# Visualization
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

lines_species = [axs[0, i].plot(x, snapshots[0][:, i])[0] for i in range(3)]
lines_reference = [axs[0, i].plot(x, reference_snapshots[0][:, i], linestyle="--")[0] for i in range(3)]
lines_difference = [axs[0, i].plot(x, np.abs(snapshots[0][:, i] - reference_snapshots[0][:, i]), linestyle=":")[0] for i in range(3)]
lines_density, = axs[1, 0].plot(x, density_snapshots[0])
lines_velocity, = axs[1, 1].plot(x, velocity_snapshots[0])
lines_pressure, = axs[1, 2].plot(x, pressure_snapshots[0])

def update(frame):
    for i in range(3):
        lines_species[i].set_ydata(snapshots[frame][:, i])
        lines_reference[i].set_ydata(reference_snapshots[frame][:, i])
        lines_difference[i].set_ydata(np.abs(snapshots[frame][:, i] - reference_snapshots[frame][:, i]))

    lines_density.set_ydata(density_snapshots[frame])
    lines_velocity.set_ydata(velocity_snapshots[frame])
    lines_pressure.set_ydata(pressure_snapshots[frame])

    # Dynamically adjust y-limits
    for ax in axs.flat:
        ax.relim()  # Recalculate limits based on data
        ax.autoscale_view()  # Rescale view

    return lines_species + lines_reference + lines_difference + [lines_density, lines_velocity, lines_pressure]


ani = FuncAnimation(fig, update, frames=len(snapshots), blit=True)
ani.save('sodtube_chem.gif', writer='pillow', fps=5)
plt.show()

