import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools
import multiprocessing
import os
from collections import deque

# --- Experiment Configuration ---

GRID_WIDTH = 50
GRID_HEIGHT = 50
K_PLUS = 0.05
K_MINUS = 0.3
NEIGHBORHOOD_RADIUS = 5
NUM_NORMAL_AGENTS = 50
SIMULATION_STEPS = 100000
NUM_REPETITIONS = 5

ITEM_DENSITIES_TO_TEST = [0.05, 0.10, 0.20]
ANTI_AGENT_COUNTS_TO_TEST = [0, 5, 10, 15, 20, 25, 30, 40, 50]

RESULTS_DIR = "results_plots"
CSV_RESULTS_FILENAME = os.path.join(RESULTS_DIR, "experiment_summary_data.csv")

class Agent:
    """
    Represents an agent in the environment.
    Agents can be 'normal' or 'anti', and can pick up or drop items based on local density.
    """
    def __init__(self, agent_id, agent_type, grid_width, grid_height):
        self.id = agent_id
        self.type = agent_type
        self.x = random.randint(0, grid_width - 1)
        self.y = random.randint(0, grid_height - 1)
        self.is_laden = False
        self.grid_width = grid_width
        self.grid_height = grid_height

    def move(self):
        """Move agent to a random neighboring cell (with wrap-around)."""
        direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.x = (self.x + direction[0] + self.grid_width) % self.grid_width
        self.y = (self.y + direction[1] + self.grid_height) % self.grid_height

    def calculate_local_density_f(self, current_grid_state):
        """Calculate the fraction of occupied cells in the agent's neighborhood."""
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
        if neighborhood_cells_count == 0:
            return 0.0
        return occupied_cells_count / neighborhood_cells_count

    def pick_drop_decision(self, current_grid_state):
        """Decide whether to pick up or drop an item based on local density and agent type."""
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

def find_biggest_cluster(current_grid_state):
    """
    Find the size of the largest connected cluster of items using BFS.
    """
    rows, cols = current_grid_state.shape
    visited = np.zeros((rows, cols), dtype=bool)
    max_cluster_size = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if current_grid_state[r_idx, c_idx] == 1 and not visited[r_idx, c_idx]:
                current_cluster_size = 0
                q = deque()
                q.append((r_idx, c_idx))
                visited[r_idx, c_idx] = True
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

def run_single_scenario(params):
    """
    Runs a single simulation scenario based on the given parameters.
    Returns a dictionary with the scenario parameters and the resulting biggest cluster size.
    """
    item_density = params['item_density']
    num_anti_agents = params['num_anti_agents']
    repetition_id = params['repetition_id']
    grid_w = params.get('grid_width', GRID_WIDTH)
    grid_h = params.get('grid_height', GRID_HEIGHT)
    num_items = int(item_density * grid_w * grid_h)
    num_normal = params.get('num_normal_agents', NUM_NORMAL_AGENTS)
    steps = params.get('simulation_steps', SIMULATION_STEPS)

    # Initialize grid with items
    current_sim_grid = np.zeros((grid_h, grid_w), dtype=int)
    items_placed = 0
    while items_placed < num_items:
        x, y = random.randint(0, grid_w - 1), random.randint(0, grid_h - 1)
        if current_sim_grid[y, x] == 0:
            current_sim_grid[y, x] = 1
            items_placed += 1

    # Initialize agents
    all_agents_list = []
    agent_id_counter = 0
    for _ in range(num_normal):
        all_agents_list.append(Agent(agent_id_counter, 'normal', grid_w, grid_h))
        agent_id_counter += 1
    for _ in range(num_anti_agents):
        all_agents_list.append(Agent(agent_id_counter, 'anti', grid_w, grid_h))
        agent_id_counter += 1
    random.shuffle(all_agents_list)

    # Simulation loop
    for step in range(steps):
        for agent in all_agents_list:
            agent.move()
            agent.pick_drop_decision(current_sim_grid)
            
    final_biggest_cluster = find_biggest_cluster(current_sim_grid)
    
    return {
        'item_density': item_density,
        'num_anti_agents': num_anti_agents,
        'repetition_id': repetition_id,
        'biggest_cluster_size': final_biggest_cluster
    }

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Prepare all parameter combinations for the experiment
    all_scenario_params_list = []
    param_id_counter = 0
    for density in ITEM_DENSITIES_TO_TEST:
        for anti_count in ANTI_AGENT_COUNTS_TO_TEST:
            for rep in range(NUM_REPETITIONS):
                all_scenario_params_list.append({
                    'item_density': density,
                    'num_anti_agents': anti_count,
                    'repetition_id': rep + 1,
                    'param_combination_id': param_id_counter
                })
            param_id_counter += 1
    
    print(f"--- Starting Experiment Suite ---")
    print(f"Total scenarios to run (including repetitions): {len(all_scenario_params_list)}")
    print(f"Grid: {GRID_WIDTH}x{GRID_HEIGHT}, Normal Agents: {NUM_NORMAL_AGENTS}")
    print(f"Steps/Run: {SIMULATION_STEPS}, Reps/Config: {NUM_REPETITIONS}")
    print(f"Testing Item Densities: {ITEM_DENSITIES_TO_TEST}")
    print(f"Testing Anti-Agent Counts: {ANTI_AGENT_COUNTS_TO_TEST}")

    start_time = time.time()

    # Run scenarios in parallel using multiprocessing
    num_processes = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    print(f"Using {num_processes} worker processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_list = pool.map(run_single_scenario, all_scenario_params_list)
    
    end_time = time.time()
    print(f"--- All simulations completed in {end_time - start_time:.2f} seconds ---")

    # Aggregate results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(CSV_RESULTS_FILENAME, index=False)
    print(f"Raw simulation data saved to {CSV_RESULTS_FILENAME}")

    # --- Generate Plots ---
    print("Generating plots...")

    # Plot: Average Biggest Cluster Size vs. Num Anti-Agents, one plot per item density
    for density, group_df in results_df.groupby('item_density'):
        agg_df = group_df.groupby('num_anti_agents')['biggest_cluster_size'].agg(['mean', 'std']).reset_index()
        plt.figure(figsize=(10, 6))
        plt.errorbar(agg_df['num_anti_agents'], agg_df['mean'], yerr=agg_df['std'], fmt='-o', capsize=5, label=f"Density {density*100:.0f}%")
        plt.xlabel("Number of Anti-Agents")
        plt.ylabel("Average Biggest Cluster Size")
        plt.title(f"Effect of Anti-Agents on Clustering (Item Density: {density*100:.0f}%)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plot_filename = os.path.join(RESULTS_DIR, f"plot_density_{int(density*100)}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot: {plot_filename}")

    # Plot: All densities on one plot for comparison
    plt.figure(figsize=(12, 7))
    for density, group_df in results_df.groupby('item_density'):
        agg_df = group_df.groupby('num_anti_agents')['biggest_cluster_size'].agg(['mean', 'std']).reset_index()
        plt.errorbar(agg_df['num_anti_agents'], agg_df['mean'], yerr=agg_df['std'], fmt='-o', capsize=5, label=f"Density {density*100:.0f}%")
    
    plt.xlabel("Number of Anti-Agents")
    plt.ylabel("Average Biggest Cluster Size")
    plt.title("Effect of Anti-Agents on Clustering (Comparison of Item Densities)")
    plt.legend(title="Item Density")
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_filename_combined = os.path.join(RESULTS_DIR, "plot_all_densities_comparison.png")
    plt.savefig(plot_filename_combined)
    plt.close()
    print(f"Saved plot: {plot_filename_combined}")

    print(f"--- Plot generation complete. Plots saved in '{RESULTS_DIR}' directory. ---")