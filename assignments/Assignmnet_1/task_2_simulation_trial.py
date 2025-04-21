import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.animation import FuncAnimation

# -------------------- PARAMETERS --------------------
N = 250         # Number of fireflies
L = 50          # Flashing cycle length (L/2 flash, L/2 off)
T = 5000        # Total simulation time steps
vicinities = [0.05, 0.1, 0.5, 1.4]  # Vicinity distances to test

# -------------------- SIMULATION FUNCTION --------------------
def simulate_fireflies(r, T=T, positions=None, init_phases=None):
    if positions is None:
        positions = np.random.rand(N, 2)
    if init_phases is None:
        init_phases = np.random.randint(0, L, size=N)

    tree = KDTree(positions)
    neighbors = [tree.query_ball_point(pos, r) for pos in positions]
    avg_neighbors = np.mean([len(nb) - 1 for nb in neighbors])

    flashing_counts = []
    phases = init_phases.copy()
    
    for t in range(T):
        flashing = (phases < (L // 2))
        flashing_counts.append(np.sum(flashing))

        new_phases = (phases + 1) % L
        for i in range(N):
            if phases[i] == 0:
                flashing_neighbors = np.sum(flashing[neighbors[i]])
                if flashing_neighbors >= 1:  # Looser condition to promote sync
                    new_phases[i] = (new_phases[i] + 1) % L
        phases = new_phases

    return flashing_counts, avg_neighbors, positions, init_phases

# -------------------- PLOTTING RESULTS --------------------
def plot_flashing_counts(results, avg_neighbors_dict):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axs = axs.flatten()

    for idx, r in enumerate(vicinities):
        axs[idx].plot(results[r][:150], color='blue', linewidth=1.5)
        axs[idx].set_xlim(0, 150)
        axs[idx].set_ylim(0, N)
        axs[idx].set_title(f"r = {r}, Avg. Neighbors = {avg_neighbors_dict[r]:.2f}")
        axs[idx].set_xlabel("Time Steps")
        axs[idx].set_ylabel("Flashing Count")
        axs[idx].grid(True)

    plt.tight_layout()
    plt.show()

# -------------------- ANIMATION FUNCTION --------------------
def animate_fireflies(r):
    MAX_STEPS = 3000

    positions = np.random.rand(N, 2)
    init_phases = np.random.randint(0, L, size=N)
    phases = init_phases.copy()
    tree = KDTree(positions)
    neighbors = [tree.query_ball_point(pos, r) for pos in positions]

    flashing_states = []
    time_steps = 0
    synchronized = False

    while not synchronized and time_steps < MAX_STEPS:
        flashing = (phases < (L // 2))
        flashing_states.append(flashing.copy())

        if np.sum(flashing) >= 0.98 * N:
            synchronized = True
            break

        new_phases = (phases + 1) % L
        for i in range(N):
            flashing_neighbors = np.sum(flashing[neighbors[i]])
            sync_strength = flashing_neighbors / len(neighbors[i])  # between 0 and 1
            if sync_strength >= 0.5:  # only shift if strong flashing environment
                phase_shift = int(np.ceil(sync_strength * 3))  # more conservative jump
                new_phases[i] = (new_phases[i] + phase_shift) % L
        phases = new_phases
        time_steps += 1
        print(f"Step {time_steps} completed. Flashing: {np.sum(flashing)}")

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Vicinity r = {r}", fontsize=14, pad=20)
    info_text = ax.text(0.5, -0.1, "", ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def update(frame):
        flash = flashing_states[frame]
        colors = ['yellow' if f else 'black' for f in flash]
        scat.set_offsets(positions)
        scat.set_color(colors)

        info_text.set_text(f"Step: {frame} | Flashing: {np.sum(flash)}/{N}")

        if frame == len(flashing_states) - 1:
            if synchronized:
                info_text.set_text(f"✅ Synchronized at Step {frame} ({np.sum(flash)} flashing)")
                print(f"✅ Simulation complete — ~98% sync achieved in {frame} steps.")
            else:
                info_text.set_text(f"🔁 Max steps ({MAX_STEPS}) reached. Sync incomplete.")
                print(f"🔁 Simulation stopped at {frame} steps. Full sync not achieved.")

        return scat, info_text

    ani = FuncAnimation(fig, update, frames=len(flashing_states), interval=80, blit=True)
    plt.show()

# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    shared_positions = np.random.rand(N, 2)
    shared_phases = np.random.randint(0, L, size=N)

    flashing_results = {}
    avg_neighbors_dict = {}

    for r in vicinities:
        flash_counts, avg_nb, _, _ = simulate_fireflies(r, T=T, positions=shared_positions, init_phases=shared_phases)
        flashing_results[r] = flash_counts
        avg_neighbors_dict[r] = avg_nb
        print(f"Vicinity r = {r}: Avg. Neighbors = {avg_nb:.2f}")

    plot_flashing_counts(flashing_results, avg_neighbors_dict)

    # Run smart animation that ends when synchronized or hits 3000 steps
    animate_fireflies(r=0.5)
