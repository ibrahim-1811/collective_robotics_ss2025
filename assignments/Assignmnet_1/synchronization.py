import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Global parameters for the simulation
N = 250         # Number of fireflies
L = 50          # Cycle length (time steps): first half flash, second half off
T = 5000        # Total simulation time steps

def simulate_fireflies_task_A(r, T=T, positions=None, init_phases=None):
    """
    Simulate fireflies flashing for T time steps with a given vicinity distance r.
    
    Returns:
      flashing_counts: List of the number of fireflies flashing at each time step.
      avg_neighbors: Average number of neighbors per firefly (neighbors defined as others within distance r).
      positions, init_phases: The positions and initial phases used in the simulation.
    """
    if positions is None:
        positions = np.random.rand(N, 2)
    if init_phases is None:
        init_phases = np.random.randint(0, L, size=N)
    
    # Build a KDTree for efficient neighbor lookup.
    tree = KDTree(positions)
    # For each firefly, get indices of others within distance r.
    neighbors = [tree.query_ball_point(pos, r) for pos in positions]
    # Compute average number of neighbors (excluding self).
    avg_neighbors = np.mean([len(nb) - 1 for nb in neighbors])
    
    flashing_counts = []
    phases = init_phases.copy()

    for t in range(T):
        # Firefly flashes if its phase is less than L/2
        flashing = (phases < (L // 2))
        flashing_counts.append(np.sum(flashing))
        
        # Update phases by incrementing and wrapping around with modulo L.
        new_phases = (phases + 1) % L
        
        # For each firefly that has just started flashing (phase was 0), check its neighbors.
        for i in range(N):
            if phases[i] == 0:
                flashing_neighbors = np.sum(flashing[neighbors[i]])
                if flashing_neighbors > (len(neighbors[i]) / 2):
                    new_phases[i] = (new_phases[i] + 1) % L
        phases = new_phases

    return flashing_counts, avg_neighbors, positions, init_phases

# Define the vicinity distances
vicinities = [0.05, 0.1, 0.5, 1.4]

# Generate the same positions and initial phases for all simulations for a fair comparison.
positions_A = np.random.rand(N, 2)
init_phases_A = np.random.randint(0, L, size=N)

# Dictionaries to store simulation results for each vicinity value.
flashing_results = {}
avg_neighbors_dict = {}
for r in vicinities:
    flashing_counts, avg_nb, _, _ = simulate_fireflies_task_A(r, T=T, positions=positions_A, init_phases=init_phases_A)
    flashing_results[r] = flashing_counts
    avg_neighbors_dict[r] = avg_nb
    print(f"Vicinity r = {r}: Average Neighbors per Firefly = {avg_nb:.2f}")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharey=True)

# Flatten the array of axes to index them in a simple loop.
axs = axs.flatten()

# Loop over each vicinity and plot the corresponding flashing counts with flipped axes.
# Loop over each vicinity and plot the corresponding flashing counts.
for idx, r in enumerate(vicinities):
    axs[idx].plot(flashing_results[r][:150], color='blue', linewidth=1.5)  # only plot first 150 time steps
    axs[idx].set_xlim(0, 150)  # x-axis: time steps from 0 to 150
    axs[idx].set_ylim(0, N)    # y-axis: number of fireflies flashing (max = N = 250)
    axs[idx].set_title(f"r = {r}, Avg. Neighbors = {avg_neighbors_dict[r]:.2f}")
    axs[idx].set_xlabel("Time Steps")
    axs[idx].set_ylabel("Flashing Count")
    axs[idx].grid(True)

plt.tight_layout()
plt.show()