import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Global parameters for Task Two
N = 250         # Number of fireflies
L = 50          # Flash cycle length: flash for first L/2 steps, off for the remaining L/2 steps
T_main = 5000   # Time steps before the final cycle
T_total = T_main + L  # Total simulation steps (5000 + final 50 steps)

def simulate_fireflies_amplitude(r, T_total=T_total):
    """
    Simulate the firefly synchronization model for T_total time steps and compute the amplitude during the final cycle.
    
    The amplitude is defined as the difference between the maximum and minimum number
    of flashing fireflies during the last L time steps.
    
    Returns:
      amplitude: The computed amplitude (max flashing count - min flashing count) in the final cycle.
    """
    # Generate random positions (uniform in [0,1] x [0,1])
    positions = np.random.rand(N, 2)
    # Generate random initial phases (each between 0 and L-1) for the fireflies
    phases = np.random.randint(0, L, size=N)
    
    # Build a KDTree for efficient lookup of neighbors.
    tree = KDTree(positions)
    # Find neighbors for each firefly: all fireflies within distance r.
    neighbors = [tree.query_ball_point(pos, r) for pos in positions]
    
    # This will record the number of flashing fireflies at each time step.
    flashing_history = np.zeros(T_total, dtype=int)
    
    # Run the simulation for T_total time steps.
    for t in range(T_total):
        # A firefly flashes if its phase is less than L/2.
        flashing = (phases < (L // 2))
        flashing_history[t] = np.sum(flashing)
        
        # Update phases: increment by 1 with a modulo to keep within 0 to L-1.
        new_phases = (phases + 1) % L
        
        # For each firefly that has just started flashing (phase was 0),
        # check its neighbors and adjust if the majority are flashing.
        for i in range(N):
            if phases[i] == 0:
                flashing_neighbors = np.sum(flashing[neighbors[i]])
                if flashing_neighbors > (len(neighbors[i]) / 2):
                    new_phases[i] = (new_phases[i] + 1) % L
        phases = new_phases

    # Extract the final cycle: last L time steps from t = T_main to T_total - 1.
    final_cycle = flashing_history[T_main:T_total]
    # Compute amplitude as the difference between max and min flashing counts.
    amplitude = np.max(final_cycle) - np.min(final_cycle)
    return amplitude

# Define the vicinity distances for which the simulations will run.
r_values = np.arange(0.025, 1.4 + 0.0001, 0.025)
n_runs = 50  # Number of independent simulation runs per r

# List to hold the average amplitude for each r value.
avg_amplitudes = []

# For each vicinity distance, run the simulation n_runs times and average the amplitudes.
for r in r_values:
    amplitudes = []
    for run in range(n_runs):
        amp = simulate_fireflies_amplitude(r)
        amplitudes.append(amp)
    avg_amp = np.mean(amplitudes)
    avg_amplitudes.append(avg_amp)
    print(f"r = {r:.3f} --> Average Amplitude = {avg_amp:.2f}")

# Plot the average amplitude vs. the vicinity distance r.
plt.figure(figsize=(10, 6))
plt.plot(r_values, avg_amplitudes, marker='o', linestyle='-')
plt.title("Average Double-Amplitude vs Vicinity Distance r")
plt.xlabel("Vicinity Distance r")
plt.ylabel("Average Amplitude (Max - Min) in Final Cycle")
plt.grid(True)
plt.show()
