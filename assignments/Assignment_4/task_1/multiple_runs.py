import numpy as np
import matplotlib.pyplot as plt
import time

# --- Simulation Parameters ---
C = 1.0                  # Circumference of the ring
SPEED = 0.001            # Step size per time step
PERCEPTION_RANGE = 0.045 # Perception range for each locust
P_SWITCH_SPONTANEOUS = 0.015  # Probability of spontaneous direction switch
N_LOCUSTS = 20           # Number of locusts

# --- Experiment Parameters ---
N_RUNS = 1000            # Number of independent simulation runs
N_TIME_STEPS_PER_RUN = 500  # Number of time steps per run

# --- Transition Matrix ---
# transition_matrix[L_t, L_{t+1}] counts transitions from L_t to L_{t+1}
transition_matrix = np.zeros((N_LOCUSTS + 1, N_LOCUSTS + 1), dtype=int)

def simulate_step_and_get_counts(current_positions, current_directions):
    """
    Simulate one time step for all locusts.
    Returns updated positions, updated directions, L_t, and L_{t+1}.
    """
    L_t = np.sum(current_directions == -1)
    new_directions = current_directions.copy()

    for i in range(N_LOCUSTS):
        locust_pos = current_positions[i]
        locust_dir = current_directions[i]

        # Find neighbors within perception range (with ring wrap-around)
        neighbors = []
        for j in range(N_LOCUSTS):
            if i == j:
                continue
            dist = np.abs(current_positions[j] - locust_pos)
            dist_wrapped = min(dist, C - dist)
            if dist_wrapped <= PERCEPTION_RANGE:
                neighbors.append(j)

        # Decide if locust switches direction
        switched_by_neighbors = False
        if neighbors:
            neighbor_dirs = current_directions[neighbors]
            num_opposite = np.sum(neighbor_dirs == -locust_dir)
            num_same = np.sum(neighbor_dirs == locust_dir)
            if num_opposite > num_same:
                switched_by_neighbors = True

        switched_spontaneously = np.random.rand() < P_SWITCH_SPONTANEOUS

        if switched_by_neighbors or switched_spontaneously:
            new_directions[i] = -locust_dir

    L_t_plus_1 = np.sum(new_directions == -1)
    updated_positions = (current_positions + new_directions * SPEED) % C
    return updated_positions, new_directions, L_t, L_t_plus_1

# --- Main Simulation Loop ---
start_time = time.time()
print(f"Starting {N_RUNS} simulation runs of {N_TIME_STEPS_PER_RUN} steps each...")

for run_idx in range(N_RUNS):
    if (run_idx + 1) % 100 == 0:
        print(f"  Starting Run {run_idx + 1}/{N_RUNS}...")

    # Initialize locusts for this run
    current_positions = np.random.rand(N_LOCUSTS) * C
    current_directions = np.random.choice([-1, 1], size=N_LOCUSTS)

    # Simulate time steps for this run
    for t_step in range(N_TIME_STEPS_PER_RUN - 1):
        updated_positions, updated_directions, L_t, L_t_plus_1 = \
            simulate_step_and_get_counts(current_positions, current_directions)
        transition_matrix[L_t, L_t_plus_1] += 1
        current_positions = updated_positions
        current_directions = updated_directions

end_time = time.time()
print(f"Simulations completed in {end_time - start_time:.2f} seconds.")

# --- Plot Transition Matrix as Heatmap ---
plt.figure(figsize=(10, 8))
plt.imshow(transition_matrix, origin='lower', cmap='viridis', interpolation='nearest', aspect='auto')
plt.colorbar(label='Frequency of Transitions Observed During Simulation')
plt.xlabel('Number of Left-Going Locusts at t+1')
plt.ylabel('Number of Left-Going Locusts at t')
tick_positions = np.arange(N_LOCUSTS + 1)
plt.xticks(tick_positions)
plt.yticks(tick_positions)

# Add grid lines for clarity
ax = plt.gca()
ax.set_xticks(np.arange(-.5, N_LOCUSTS + 1, 1), minor=True)
ax.set_yticks(np.arange(-.5, N_LOCUSTS + 1, 1), minor=True)
ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5, alpha=0.5)
ax.tick_params(which="minor", size=0)
plt.savefig("task_B_transition_histogram.png")
plt.show()

# --- Transition Count Verification ---
total_transitions_recorded = np.sum(transition_matrix)
expected_transitions = N_RUNS * (N_TIME_STEPS_PER_RUN - 1)
print(f"\nTotal transitions recorded in matrix: {total_transitions_recorded}")
print(f"Expected total transitions: {expected_transitions}")
np.save("transition_matrix_A.npy", transition_matrix)
print(f"Transition matrix A saved to transition_matrix_A.npy")
