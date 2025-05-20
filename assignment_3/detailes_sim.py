import random
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from collections import deque

# --- SCRIPT MODE ---
# 'DATA_COLLECTION' : Runs multiple configurations and repetitions, saves to CSV, no live visualization.
# 'VISUALIZE_SINGLE_RUN' : Runs ONE specific configuration with live visualization.
MODE = 'DATA_COLLECTION' # <-- CHANGE THIS TO 'DATA_COLLECTION' FOR CSV OUTPUT

# --- General Simulation Parameters (Can be used by both modes) ---
# These are defaults; you might want to increase them for more robust data,
# matching closer to the research papers (e.g., GRID 100x100 or 500x500, STEPS 50000+).
GRID_WIDTH = 50
GRID_HEIGHT = 50
# Initial item density (e.g., 0.01 for 1%, 0.05 for 5%)
ITEM_DENSITY = 0.05
NUM_ITEMS_INITIAL = int(ITEM_DENSITY * GRID_WIDTH * GRID_HEIGHT)

# Agent interaction parameters
K_PLUS = 0.05  # Threshold for picking up (P_pick = (k+ / (k+ + f))^2)
K_MINUS = 0.3  # Threshold for dropping (P_drop = (f / (k- + f))^2)
NEIGHBORHOOD_RADIUS = 5 # Von Neumann radius for sensing local density 'f'

# --- Parameters for DATA_COLLECTION Mode ---
DC_NUM_NORMAL_AGENTS = 50
# List of numbers of anti-agents to test
DC_ANTI_AGENT_CONFIGURATIONS = [0, 5, 10, 15, 20, 25, 30, 40, 50]
DC_SIMULATION_STEPS = 20000   # Number of steps per simulation run
DC_NUM_REPETITIONS = 5        # Number of times to repeat each configuration
DC_OUTPUT_FILENAME = "results/simulation_data_robust.csv"

# --- Parameters for VISUALIZE_SINGLE_RUN Mode ---
VIZ_NUM_NORMAL_AGENTS = 20
VIZ_NUM_ANTI_AGENTS = 10    # Number of anti-agents for this single visual run
VIZ_SIMULATION_STEPS = 2000   # Shorter or longer, depending on patience
VIZ_UPDATE_RATE = 2         # Update visual every X steps (lower is slower but smoother)

# --- Agent Class (Unchanged) ---
class Agent:
    def __init__(self, agent_id, agent_type, grid_width, grid_height):
        self.id = agent_id
        self.type = agent_type
        self.x = random.randint(0, grid_width - 1)
        self.y = random.randint(0, grid_height - 1)
        self.is_laden = False
        self.grid_width = grid_width
        self.grid_height = grid_height

    def move(self):
        direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.x = (self.x + direction[0] + self.grid_width) % self.grid_width
        self.y = (self.y + direction[1] + self.grid_height) % self.grid_height

    def calculate_local_density_f(self, current_grid_state):
        neighborhood_cells_count = 0
        occupied_cells_count = 0
        for dx in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
            for dy in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
                if abs(dx) + abs(dy) <= NEIGHBORHOOD_RADIUS:
                    check_x = (self.x + dx + self.grid_width) % self.grid_width
                    check_y = (self.y + dy + self.grid_height) % self.grid_height
                    neighborhood_cells_count += 1
                    if current_grid_state[check_y, check_x] == 1:
                        occupied_cells_count += 1
        if neighborhood_cells_count == 0: return 0.0
        return occupied_cells_count / neighborhood_cells_count

    def pick_drop_decision(self, current_grid_state):
        local_f = self.calculate_local_density_f(current_grid_state)
        p_pick_std = (K_PLUS / (K_PLUS + local_f))**2 if (K_PLUS + local_f) > 0 else 0
        p_drop_std = (local_f / (K_MINUS + local_f))**2 if (K_MINUS + local_f) > 0 else 0
        
        p_pick_actual, p_drop_actual = p_pick_std, p_drop_std
        if self.type == 'anti':
            p_pick_actual = 1.0 - p_pick_std
            p_drop_actual = 1.0 - p_drop_std
            
        if not self.is_laden and current_grid_state[self.y, self.x] == 1:
            if random.random() < p_pick_actual:
                self.is_laden = True
                current_grid_state[self.y, self.x] = 0
        elif self.is_laden and current_grid_state[self.y, self.x] == 0:
            if random.random() < p_drop_actual:
                self.is_laden = False
                current_grid_state[self.y, self.x] = 1

# --- Cluster Analysis (Unchanged) ---
def find_biggest_cluster(current_grid_state):
    rows, cols = current_grid_state.shape
    visited = np.zeros((rows, cols), dtype=bool)
    max_cluster_size = 0
    for r in range(rows):
        for c in range(cols):
            if current_grid_state[r, c] == 1 and not visited[r, c]:
                current_cluster_size = 0
                q = deque()
                q.append((r, c))
                visited[r, c] = True
                current_cluster_size += 1
                while q:
                    curr_r, curr_c = q.popleft()
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and \
                           current_grid_state[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            current_cluster_size += 1
                max_cluster_size = max(max_cluster_size, current_cluster_size)
    return max_cluster_size

# --- Visualization Globals and Functions (Largely Unchanged) ---
fig_viz, ax_viz = None, None 

def setup_visualization():
    global fig_viz, ax_viz
    fig_viz, ax_viz = plt.subplots(figsize=(8,8))
    plt.ion() 
    fig_viz.show()

def visualize_grid_state(current_grid_display, agents_list, step_num, num_normal_val, num_anti_val):
    if not fig_viz: return

    ax_viz.clear() 
    display_grid_viz = np.copy(current_grid_display).astype(float)
    
    ax_viz.imshow(display_grid_viz, cmap='Greys', origin='lower', vmin=0, vmax=1)

    normal_agents_x = [agent.x for agent in agents_list if agent.type == 'normal']
    normal_agents_y = [agent.y for agent in agents_list if agent.type == 'normal']
    anti_agents_x = [agent.x for agent in agents_list if agent.type == 'anti']
    anti_agents_y = [agent.y for agent in agents_list if agent.type == 'anti']

    normal_colors = ['darkblue' if agent.is_laden else 'blue' for agent in agents_list if agent.type == 'normal']
    anti_colors = ['darkred' if agent.is_laden else 'red' for agent in agents_list if agent.type == 'anti']

    ax_viz.scatter(normal_agents_x, normal_agents_y, c=normal_colors, marker='o', s=60, label='Normal', edgecolors='black')
    ax_viz.scatter(anti_agents_x, anti_agents_y, c=anti_colors, marker='X', s=70, label='Anti', edgecolors='black')
    
    ax_viz.set_title(f"Step: {step_num}/{VIZ_SIMULATION_STEPS if MODE == 'VISUALIZE_SINGLE_RUN' else DC_SIMULATION_STEPS} | Items: {np.sum(current_grid_display)} | Normal: {num_normal_val} Anti: {num_anti_val}")
    ax_viz.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax_viz.set_ylim(-0.5, GRID_HEIGHT - 0.5)
    if len(agents_list) < 30:
         ax_viz.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    fig_viz.canvas.draw_idle()
    plt.pause(0.0001)

# --- Core Simulation Loop for a Single Run ---
def perform_single_simulation_run(num_normal, num_anti, total_steps, run_id, is_visualizing=False, viz_update_rate=10):
    """
    Performs a single simulation run with given parameters.
    Returns the size of the biggest cluster at the end.
    Optionally visualizes if is_visualizing is True.
    """
    if is_visualizing:
        print(f"--- Visualizing Run: {num_normal} Normal, {num_anti} Anti Agents ---")
    
    current_sim_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    items_placed = 0
    while items_placed < NUM_ITEMS_INITIAL:
        x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
        if current_sim_grid[y, x] == 0:
            current_sim_grid[y, x] = 1
            items_placed += 1

    all_agents_list = []
    agent_id_counter = 0
    for _ in range(num_normal):
        all_agents_list.append(Agent(agent_id_counter, 'normal', GRID_WIDTH, GRID_HEIGHT))
        agent_id_counter +=1
    for _ in range(num_anti):
        all_agents_list.append(Agent(agent_id_counter, 'anti', GRID_WIDTH, GRID_HEIGHT))
        agent_id_counter +=1
    random.shuffle(all_agents_list)

    if is_visualizing:
        visualize_grid_state(current_sim_grid, all_agents_list, 0, num_normal, num_anti)
        time.sleep(0.5)

    for step in range(total_steps):
        # Progress for non-visual or long visual runs
        if not is_visualizing and (step + 1) % (total_steps // 10 if total_steps >=10 else 1) == 0 :
             print(f"    Run {run_id}, Anti {num_anti}: Step {step + 1}/{total_steps}")
        elif is_visualizing and (step + 1) % (total_steps // 20 if total_steps >=20 else 1) == 0 :
             print(f"  Visualizing Step {step + 1}/{total_steps}")


        for agent in all_agents_list:
            agent.move()
            agent.pick_drop_decision(current_sim_grid)
        
        if is_visualizing and (step + 1) % viz_update_rate == 0:
            visualize_grid_state(current_sim_grid, all_agents_list, step + 1, num_normal, num_anti)
    
    if is_visualizing:
        visualize_grid_state(current_sim_grid, all_agents_list, total_steps, num_normal, num_anti)
    
    final_biggest_cluster = find_biggest_cluster(current_sim_grid)
    return final_biggest_cluster

# --- Main Execution ---
if __name__ == "__main__":
    if MODE == 'VISUALIZE_SINGLE_RUN':
        setup_visualization()
        perform_single_simulation_run(
            num_normal=VIZ_NUM_NORMAL_AGENTS,
            num_anti=VIZ_NUM_ANTI_AGENTS,
            total_steps=VIZ_SIMULATION_STEPS,
            run_id="Viz",
            is_visualizing=True,
            viz_update_rate=VIZ_UPDATE_RATE
        )
        print("Visualization finished. Close the plot window to exit.")
        plt.ioff()
        plt.show()

    elif MODE == 'DATA_COLLECTION':
        print(f"--- Starting Data Collection Mode ---")
        print(f"Grid: {GRID_WIDTH}x{GRID_HEIGHT}, Items: {NUM_ITEMS_INITIAL} ({ITEM_DENSITY*100}%)")
        print(f"Normal Agents: {DC_NUM_NORMAL_AGENTS}, K+:{K_PLUS}, K-:{K_MINUS}, HoodRad:{NEIGHBORHOOD_RADIUS}")
        print(f"Steps/Run: {DC_SIMULATION_STEPS}, Reps/Config: {DC_NUM_REPETITIONS}")
        print(f"Anti-Agent Configs: {DC_ANTI_AGENT_CONFIGURATIONS}")
        print(f"Output to: {DC_OUTPUT_FILENAME}")

        # Ensure results directory exists
        if '/' in DC_OUTPUT_FILENAME:
            import os
            os.makedirs(os.path.dirname(DC_OUTPUT_FILENAME), exist_ok=True)

        with open(DC_OUTPUT_FILENAME, 'w', newline='') as csvfile:
            fieldnames = ['num_anti_agents', 'repetition', 'biggest_cluster_size']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for num_anti_agents_current in DC_ANTI_AGENT_CONFIGURATIONS:
                print(f"  Running configuration: {num_anti_agents_current} Anti-Agents")
                for rep in range(DC_NUM_REPETITIONS):
                    print(f"    Repetition {rep + 1}/{DC_NUM_REPETITIONS}")
                    biggest_cluster_result = perform_single_simulation_run(
                        num_normal=DC_NUM_NORMAL_AGENTS,
                        num_anti=num_anti_agents_current,
                        total_steps=DC_SIMULATION_STEPS,
                        run_id=f"Rep{rep+1}",
                        is_visualizing=False # No live viz during data collection
                    )
                    writer.writerow({
                        'num_anti_agents': num_anti_agents_current,
                        'repetition': rep + 1,
                        'biggest_cluster_size': biggest_cluster_result
                    })
                    print(f"    Completed. Biggest cluster: {biggest_cluster_result}")
        print(f"--- Data Collection Complete. Results saved to {DC_OUTPUT_FILENAME} ---")
    else:
        print(f"Error: Unknown MODE '{MODE}'. Choose 'DATA_COLLECTION' or 'VISUALIZE_SINGLE_RUN'.")