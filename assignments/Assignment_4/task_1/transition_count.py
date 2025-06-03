import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- Simulation Parameters ---
C = 1.0                   # Circumference of the ring
SPEED = 0.001             # Step size per time step
PERCEPTION_RANGE = 0.045  # Perception range for each locust
P_SWITCH_SPONTANEOUS = 0.015  # Probability of spontaneous direction switch
N_LOCUSTS = 20            # Number of locusts

# --- Experiment Parameters ---
N_RUNS = 1000             # Number of independent simulation runs
N_TIME_STEPS_PER_RUN = 500  # Number of time steps per run

# --- Transition Matrix A ---
# transition_matrix_A[L_t, L_{t+1}] counts transitions from L_t to L_{t+1}
transition_matrix_A = np.load("transition_matrix_A.npy")

# --- Build Markov Model from Transition Matrix ---
# 1. Count occurrences M[i]: number of times state i was an origin for a transition
M_i = np.sum(transition_matrix_A, axis=1)

# 2. Normalize A[i][j] by M[i] to get transition probabilities P_ij
P_ij = np.zeros_like(transition_matrix_A, dtype=float)
for i in range(N_LOCUSTS + 1):
    if M_i[i] > 0:
        P_ij[i, :] = transition_matrix_A[i, :] / M_i[i]
    else:
        # If state i was never observed as an origin, make it absorbing
        print(f"Warning: State L={i} was never an origin state (M[{i}]=0). Setting P[{i},{i}]=1.")
        P_ij[i, i] = 1.0

# --- Sample a Trajectory from the Markov Model ---
N_TIME_STEPS_MODEL = 500  # Length of the new trajectory
L_trajectory_model = np.zeros(N_TIME_STEPS_MODEL, dtype=int)

# Initial state L_0: Sampled from random initial directions
initial_directions_sample = np.random.choice([-1, 1], size=N_LOCUSTS)
L_trajectory_model[0] = np.sum(initial_directions_sample == -1)

for t in range(N_TIME_STEPS_MODEL - 1):
    current_L = L_trajectory_model[t]
    probabilities_for_next_L = P_ij[current_L, :]
    possible_next_states = np.arange(N_LOCUSTS + 1)
    prob_sum = np.sum(probabilities_for_next_L)
    if not np.isclose(prob_sum, 1.0) and prob_sum > 0:
        probabilities_for_next_L = probabilities_for_next_L / prob_sum
    elif prob_sum == 0:
        print(f"Error: Probabilities sum to 0 for L={current_L}. Forcing self-loop.")
        probabilities_for_next_L = np.zeros(N_LOCUSTS + 1)
        probabilities_for_next_L[current_L] = 1.0
    L_trajectory_model[t+1] = np.random.choice(possible_next_states, p=probabilities_for_next_L)

# --- Plot the Sampled Trajectory from the Markov Model ---
plt.figure(figsize=(12, 6))
plt.plot(range(N_TIME_STEPS_MODEL), L_trajectory_model, label='Sampled $L_t$ from Markov Model (Part C)')
plt.xlabel("Time Step")
plt.ylabel("Number of Left-Going Locusts ($L_t$)")
plt.title(f'Trajectory from Reduced Model (N={N_LOCUSTS})')
plt.ylim(0, N_LOCUSTS)
plt.grid(True)
plt.legend()
plt.savefig("task_C_markov_model_trajectory.png")
plt.show()

# --- For Comparison: Generate a Sample Trajectory from the Full ABM (Part A) ---
def simulate_step_and_get_counts_for_A(current_positions, current_directions):
    """
    Simulate one time step for all locusts.
    Returns updated positions, updated directions, L_t, and L_{t+1}.
    """
    L_t = np.sum(current_directions == -1)
    new_directions = current_directions.copy()
    for i in range(N_LOCUSTS):
        locust_pos = current_positions[i]
        locust_dir = current_directions[i]
        switched_by_neighbors = False
        neighbors = []
        for j in range(N_LOCUSTS):
            if i == j:
                continue
            dist = np.abs(current_positions[j] - locust_pos)
            dist_wrapped = min(dist, C - dist)
            if dist_wrapped <= PERCEPTION_RANGE:
                neighbors.append(j)
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

num_left_going_history_A = []
positions_A_sample = np.random.rand(N_LOCUSTS) * C
directions_A_sample = np.random.choice([-1, 1], size=N_LOCUSTS)

for t_a in range(N_TIME_STEPS_MODEL):
    num_left_going_history_A.append(np.sum(directions_A_sample == -1))
    temp_updated_pos, temp_updated_dir, _, _ = \
        simulate_step_and_get_counts_for_A(positions_A_sample, directions_A_sample)
    positions_A_sample = temp_updated_pos
    directions_A_sample = temp_updated_dir

plt.figure(figsize=(12, 6))
plt.plot(range(N_TIME_STEPS_MODEL), num_left_going_history_A, label='Sample $L_t$ from Full ABM (Part A type)')
plt.xlabel("Time Step")
plt.ylabel("Number of Left-Going Locusts ($L_t$)")
plt.title(f'Trajectory from Full Agent-Based Model (N={N_LOCUSTS}) - Example for Comparison')
plt.ylim(0, N_LOCUSTS)
plt.grid(True)
plt.legend()
plt.savefig("task_C_ABM_comparison_trajectory.png")
plt.show()