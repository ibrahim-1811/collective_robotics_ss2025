import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
C = 1.0                  # Circumference of the ring
SPEED = 0.001            # Step size per time step
PERCEPTION_RANGE = 0.045 # How far a locust can sense neighbors
P_SWITCH_SPONTANEOUS = 0.015  # Probability of spontaneous direction switch
N_LOCUSTS = 20           # Number of locusts in the simulation
N_TIME_STEPS = 500       # Total simulation duration (also used for animation)
ANIMATION_INTERVAL = 50  # Milliseconds per animation frame

# --- Global State Variables ---
positions = np.random.rand(N_LOCUSTS) * C           # Initial positions on the ring
directions = np.random.choice([-1, 1], N_LOCUSTS)   # Initial directions: -1 (left), 1 (right)
num_left_going_history = []                         # Track number of left-going locusts over time
current_time_step = 0                               # Simulation time step counter

# --- Setup Figure and Subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Subplot 1: Visual simulation of locusts on the ring
ring_radius = C / (2 * np.pi)
ax1.set_xlim(-1.2 * ring_radius, 1.2 * ring_radius)
ax1.set_ylim(-1.2 * ring_radius, 1.2 * ring_radius)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title(f"Locusts on Ring (N={N_LOCUSTS})")
ring_circle = plt.Circle((0, 0), ring_radius, color='lightgrey', fill=False, linestyle='--')
ax1.add_artist(ring_circle)
left_locust_scatter = ax1.scatter([], [], color='red', label='Left-going')
right_locust_scatter = ax1.scatter([], [], color='blue', label='Right-going')
ax1.legend(loc='upper right')
time_text_ax1 = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, ha='left', va='top')


# Subplot 2: Number of left-going locusts over time
ax2.set_xlim(0, N_TIME_STEPS)
ax2.set_ylim(0, N_LOCUSTS)
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Number of Left-Going Locusts")
ax2.set_title("Left-Going Locusts Over Time")
ax2.grid(True)
line_left_count, = ax2.plot([], [], 'r-')

def update_simulation_step():
    """
    Advance the simulation by one time step:
    - Each locust checks its neighbors within PERCEPTION_RANGE.
    - If more neighbors are moving in the opposite direction, it switches.
    - Each locust may also switch direction spontaneously.
    - Positions are updated according to direction and SPEED.
    """
    global positions, directions, num_left_going_history, current_time_step

    if current_time_step >= N_TIME_STEPS:
        return

    # Record current number of left-going locusts
    num_left_going = np.sum(directions == -1)
    num_left_going_history.append(num_left_going)

    new_directions = directions.copy()

    for i in range(N_LOCUSTS):
        current_pos = positions[i]
        current_dir = directions[i]

        # Find neighbors within perception range (accounting for ring wrap-around)
        neighbors = []
        for j in range(N_LOCUSTS):
            if i == j:
                continue
            dist = np.abs(positions[j] - current_pos)
            dist_wrapped = min(dist, C - dist)
            if dist_wrapped <= PERCEPTION_RANGE:
                neighbors.append(j)

        switched_by_neighbors = False
        if neighbors:
            neighbor_dirs = directions[neighbors]
            num_opposite = np.sum(neighbor_dirs == -current_dir)
            num_same = np.sum(neighbor_dirs == current_dir)
            # Switch if more neighbors are going the opposite way
            if num_opposite > num_same:
                switched_by_neighbors = True

        # Spontaneous switching
        switched_spontaneously = np.random.rand() < P_SWITCH_SPONTANEOUS

        if switched_by_neighbors or switched_spontaneously:
            new_directions[i] = -current_dir

    directions[:] = new_directions

    # Update positions and wrap around the ring
    positions[:] = (positions + directions * SPEED) % C

    current_time_step += 1

def animate(frame_num):
    """
    Animation callback for FuncAnimation.
    Runs one simulation step and updates both subplots:
    - Subplot 1: Shows locusts on the ring, colored by direction.
    - Subplot 2: Plots the number of left-going locusts over time.
    """
    if current_time_step < N_TIME_STEPS:
        update_simulation_step()

    # Update Subplot 1: Locusts on Ring
    left_indices = np.where(directions == -1)[0]
    right_indices = np.where(directions == 1)[0]

    # Convert 1D positions to 2D for circular plot
    angles_left = (positions[left_indices] / C) * 2 * np.pi
    x_left = ring_radius * np.cos(angles_left)
    y_left = ring_radius * np.sin(angles_left)
    left_locust_scatter.set_offsets(np.c_[x_left, y_left])

    angles_right = (positions[right_indices] / C) * 2 * np.pi
    x_right = ring_radius * np.cos(angles_right)
    y_right = ring_radius * np.sin(angles_right)
    right_locust_scatter.set_offsets(np.c_[x_right, y_right])

    time_text_ax1.set_text(f'Time: {current_time_step}/{N_TIME_STEPS}')

    # Update Subplot 2: Number of left-going locusts
    if num_left_going_history:
        line_left_count.set_data(range(len(num_left_going_history)), num_left_going_history)
        ax2.set_xlim(0, max(N_TIME_STEPS, len(num_left_going_history)))

    return left_locust_scatter, right_locust_scatter, line_left_count, time_text_ax1

# --- Run the Animation ---
ani = animation.FuncAnimation(
    fig, animate, frames=N_TIME_STEPS,
    interval=ANIMATION_INTERVAL, blit=True, repeat=False
)

plt.tight_layout()
plt.savefig("task_A_ABM_locust_trajectory.png")
plt.show()

# --- If animation is interrupted, plot the full history ---
if len(num_left_going_history) < N_TIME_STEPS:
    while current_time_step < N_TIME_STEPS:
        update_simulation_step()

if num_left_going_history:
    print(f"\nSimulation finished. Total time steps recorded: {len(num_left_going_history)}")
    if not plt.fignum_exists(fig.number):
        fig_final, ax_final = plt.subplots(figsize=(10, 6))
        ax_final.plot(range(len(num_left_going_history)), num_left_going_history, 'r-')
        ax_final.set_xlabel("Time Step")
        ax_final.set_ylabel("Number of Left-Going Locusts")
        ax_final.set_title(f"Locust Simulation (N={N_LOCUSTS}) - Full Run")
        ax_final.grid(True)
        ax_final.set_ylim(0, N_LOCUSTS)
        plt.show()

print("\n--- Sample Final Locust States ---")
for i in range(min(5, N_LOCUSTS)):
    print(f"Locust {i}: Position={positions[i]:.3f}, Direction={'Left' if directions[i]==-1 else 'Right'}")