import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import deepxde as dde

# Define dummy data for inference
x_train_dummy = (np.zeros((1, 3)), np.zeros((1, 1)))  # Placeholder input: IC (3 species), time (1 value)
y_train_dummy = np.zeros((1, 1, 3))  # Placeholder output: 3 species
data_dummy = dde.data.Triple(X_train=x_train_dummy, y_train=y_train_dummy, X_test=x_train_dummy, y_test=y_train_dummy)

# Define the parameters for the DeepONet (match training settings)
m = 3  # Input dimension of the branch network
dim_x = 1  # Input dimension of the trunk network
activation = f"LAAF-{5} tanh"
net = dde.nn.DeepONetCartesianProd(
    [m, 60, 60],  # Branch network architecture
    [dim_x, 60, 20],  # Trunk network architecture
    activation,
    "Glorot normal",
    num_outputs=3,
    multi_output_strategy="split_branch"
)

# Initialize the model
model = dde.Model(data_dummy, net)
model.compile("adam", lr=0.001)

# Load the trained DeepONet model
model_path = 'model_3species_td_multi/model-10000.ckpt'

model.restore(model_path, verbose=1)

# Define physical and simulation parameters
nx = 100  # Number of grid points
Lx = 10.0  # Domain length
x = np.linspace(0, Lx, nx)  # Spatial grid

dx = x[1] - x[0]  # Spatial resolution
total_time = 3.0  # Total simulation time
dt = 0.01  # Time step
nt = int(total_time / dt)  # Number of time steps

# Gas properties (density, velocity, pressure)
density = np.ones(nx)/np.linspace(1,2,nx)  #np.ones(nx)
velocity = np.zeros(nx)
pressure = density #np.ones(nx)

# Initialize chemical species (n1, n2, n3)
n_species = np.zeros((nx, 3))
n_species[:, 0] = 0.9*density  # Initial n1
n_species[:, 1] = 0.1*density  # Initial n2
n_species[:, 2] = 0.0*density  # Initial n3

# ODE system for reference solution
def ode_system(t, n):
    n1, n2, n3 = n
    conv = 5e0
    k1, k2, k3 = 0.8 - np.sin(conv * t), 0.5 + np.cos(conv * t), 0.2 + np.sin(2 * t)

    dn1 = conv * (- k1 * n1 - k3 * n1 * n3)
    dn2 = conv * (k1 * n1 - k2 * n2 + 2. * k3 * n1 * n3)
    dn3 = conv * (k2 * n2 - k3 * n1 * n3)

    return dn1, dn2, dn3

# Prepare data for animation
snapshots = []
reference_snapshots = []
diff_snapshots = []

# Helper function for chemical evolution using the trained model
def evolve_chemistry(n_species_cell, dt):
    t_space = np.expand_dims(np.array([0, dt]), axis=-1)
    IC = np.expand_dims(n_species_cell, axis=0)
    x_input = (IC, t_space)
    y_pred = model.predict(x_input)
    return y_pred[0, -1, :]  # Return the updated chemical state

# Time integration loop
for t in range(nt):

    n_species[:, 0] = n_species[:,0]/density  
    n_species[:, 1] = n_species[:,1]/density  
    n_species[:, 2] = n_species[:,2]/density  


    # Hydrodynamic step (simple explicit advection as placeholder)
    density[1:-1] -= dt / dx * (velocity[2:] * density[2:] - velocity[:-2] * density[:-2])
    velocity[1:-1] -= dt / dx * ((pressure[2:] - pressure[:-2]) / density[1:-1])
    pressure = density  # Simplified isothermal equation of state

    n_species[:, 0] = n_species[:,0]*density
    n_species[:, 1] = n_species[:,1]*density
    n_species[:, 2] = n_species[:,2]*density


    # Chemistry update using DeepONet and reference ODE solver
    reference_state = np.zeros_like(n_species)
    diff_state      = np.zeros_like(n_species) 
    for i in range(nx):
        # Update using DeepONet
        n_species[i, :] = evolve_chemistry(n_species[i, :], dt)

        # Compute reference solution
        initial_conditions = n_species[i, :]
        t_span = [0, dt]
        sol = solve_ivp(ode_system, t_span, initial_conditions, method='RK45')
        reference_state[i, :] = sol.y[:, -1]  # Take the final state at t = dt

        #abs difference
        diff_state[i, :] = np.abs(n_species[i, :] - reference_state[i, :])

    # Save snapshots for animation
    if t % 10 == 0:
        snapshots.append(n_species.copy())
        reference_snapshots.append(reference_state.copy())
        diff_snapshots.append(diff_state.copy())

# Set up the animation
fig, axs = plt.subplots(1, 3, figsize=(16, 6))
lines_emulated = [
    axs[0].plot(x, snapshots[0][:, i]/density, label=f"n{i+1} (Emulated)")[0] for i in range(3)
]

lines_reference = [
    axs[1].plot(x, reference_snapshots[0][:, i]/density, label=f"n{i+1} (Reference)")[0] for i in range(3)
]

lines_diff = [
    axs[2].plot(x, diff_snapshots[0][:, i]/density, label=f"n{i+1} (diff)")[0] for i in range(3)
]

for ax in axs:
    ax.set_xlabel("x")
    ax.set_ylabel("Concentration")
    ax.legend()
    ax.grid()

axs[0].set_title("Emulated Evolution of Chemical Species")
axs[1].set_title("Reference Evolution of Chemical Species")
axs[2].set_title("Errors of Chemical Species")

# Animation update function
def update(frame):
    for i, line in enumerate(lines_emulated):
        line.set_ydata(snapshots[frame][:, i])
    for i, line in enumerate(lines_reference):
        line.set_ydata(reference_snapshots[frame][:, i])
    for i, line in enumerate(lines_diff):
        line.set_ydata(diff_snapshots[frame][:, i])

    axs[0].set_title(f"Emulated Evolution (Time: {frame * dt * 10:.2f})")
    axs[1].set_title(f"Reference Evolution (Time: {frame * dt * 10:.2f})")
    axs[2].set_title(f"Diff Evolution (Time: {frame * dt * 10:.2f})")
    return lines_emulated + lines_reference + lines_diff

ani = FuncAnimation(fig, update, frames=len(snapshots), blit=True)

# Display the animation
plt.show()
ani.save('hydro_plus_chem.gif', writer='pillow', fps=5)
