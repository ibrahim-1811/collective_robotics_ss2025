import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools # For creating parameter combinations
import multiprocessing
import os
from collections import deque

# --- Experiment Configuration ---
# These parameters define the scenarios you want to test.

# Grid and General Simulation Settings (can be large for robust results)
GRID_WIDTH = 50
GRID_HEIGHT = 50
K_PLUS = 0.05           # For P_pick
K_MINUS = 0.3           # For P_drop
NEIGHBORHOOD_RADIUS = 5
NUM_NORMAL_AGENTS = 50
SIMULATION_STEPS = 10000  # Steps per individual simulation run
NUM_REPETITIONS = 10      # Repetitions for each unique parameter combination

# Parameters to Vary for Experiments:
ITEM_DENSITIES_TO_TEST = [0.05, 0.10, 0.20] # e.g., 5%, 10%, 20%
ANTI_AGENT_COUNTS_TO_TEST = [0, 5, 10, 15, 20, 25, 30, 40, 50]

# Output settings
RESULTS_DIR = "results_plots" # Directory to save plots and CSV data
CSV_RESULTS_FILENAME = os.path.join(RESULTS_DIR, "experiment_summary_data.csv")

# --- Agent Class (Unchanged from previous versions) ---
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
        self.x = (self.x + self.direction[0] + self.grid_width) % self.grid_width
        self.y = (self.y + self.direction[1] + self.grid_height) % self.grid_height

    def calculate_local_density_f(self, current_grid_state):
        neighborhood_cells_count = 0
        occupied_cells_count = 0
        for dx in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
            for dy in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
                if abs(dx) + abs(dy) <= NEIGHBORHOOD_RADIUS: # Von Neumann
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
    for r_idx in range(rows): # renamed r to r_idx
        for c_idx in range(cols): # renamed c to c_idx
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

# --- Core Simulation Function for a Single Run (for multiprocessing) ---
def run_single_scenario(params):
    """
    Runs a single simulation scenario based on the given parameters.
    params is a dictionary containing all necessary parameters for the run.
    """
    item_density = params['item_density']
    num_anti_agents = params['num_anti_agents']
    repetition_id = params['repetition_id'] # For tracking
    
    # Extract other fixed params or pass them if they vary too
    grid_w = params.get('grid_width', GRID_WIDTH)
    grid_h = params.get('grid_height', GRID_HEIGHT)
    num_items = int(item_density * grid_w * grid_h)
    k_p = params.get('k_plus', K_PLUS)
    k_m = params.get('k_minus', K_MINUS)
    n_radius = params.get('neighborhood_radius', NEIGHBORHOOD_RADIUS)
    num_normal = params.get('num_normal_agents', NUM_NORMAL_AGENTS)
    steps = params.get('simulation_steps', SIMULATION_STEPS)

    # --- Initialize grid for this run ---
    current_sim_grid = np.zeros((grid_h, grid_w), dtype=int)
    items_placed = 0
    # Ensure random seed is different per process if not passed explicitly
    # For true independent randomness, this is usually fine as each process gets its own state.
    # If you need to control seeds for debugging, you could pass a seed in params.
    while items_placed < num_items:
        x, y = random.randint(0, grid_w - 1), random.randint(0, grid_h - 1)
        if current_sim_grid[y, x] == 0:
            current_sim_grid[y, x] = 1
            items_placed += 1

    # --- Initialize agents for this run ---
    all_agents_list = []
    agent_id_counter = 0
    for _ in range(num_normal):
        # For Agent class, ensure move method has access to self.grid_width and self.grid_height
        # The Agent class already uses self.grid_width, self.grid_height
        # Make sure agent's random move uses self.direction correctly
        temp_agent = Agent(agent_id_counter, 'normal', grid_w, grid_h)
        temp_agent.direction = (0,0) # Initialize if used in move
        all_agents_list.append(temp_agent)

        agent_id_counter +=1
    for _ in range(num_anti_agents):
        temp_agent = Agent(agent_id_counter, 'anti', grid_w, grid_h)
        temp_agent.direction = (0,0) # Initialize if used in move
        all_agents_list.append(temp_agent)
        agent_id_counter +=1
    random.shuffle(all_agents_list)

    # --- Simulation loop for this run ---
    for step in range(steps):
        for agent in all_agents_list:
            # Assign random direction for this step inside agent.move if not done already
            agent.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            agent.move() # Agent moves based on its current x,y and the chosen direction
            agent.pick_drop_decision(current_sim_grid) # Agent decides based on new pos
            
    final_biggest_cluster = find_biggest_cluster(current_sim_grid)
    
    # Print progress for long runs (will be interleaved due to multiprocessing)
    # print(f"Finished: Density {item_density*100:.0f}%, AntiAgents {num_anti_agents}, Rep {repetition_id} -> Cluster {final_biggest_cluster}")
    
    return {
        'item_density': item_density,
        'num_anti_agents': num_anti_agents,
        'repetition_id': repetition_id,
        'biggest_cluster_size': final_biggest_cluster
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate all parameter combinations for the experiments
    all_scenario_params_list = []
    param_id_counter = 0
    for density in ITEM_DENSITIES_TO_TEST:
        for anti_count in ANTI_AGENT_COUNTS_TO_TEST:
            for rep in range(NUM_REPETITIONS):
                all_scenario_params_list.append({
                    'item_density': density,
                    'num_anti_agents': anti_count,
                    'repetition_id': rep + 1,
                    'param_combination_id': param_id_counter # Unique ID for this specific set of varying params
                    # Fixed parameters can be accessed globally or passed if they were also varied
                })
            param_id_counter +=1
    
    print(f"--- Starting Experiment Suite ---")
    print(f"Total scenarios to run (including repetitions): {len(all_scenario_params_list)}")
    print(f"Grid: {GRID_WIDTH}x{GRID_HEIGHT}, Normal Agents: {NUM_NORMAL_AGENTS}")
    print(f"Steps/Run: {SIMULATION_STEPS}, Reps/Config: {NUM_REPETITIONS}")
    print(f"Testing Item Densities: {ITEM_DENSITIES_TO_TEST}")
    print(f"Testing Anti-Agent Counts: {ANTI_AGENT_COUNTS_TO_TEST}")
    print(f"This may take a significant amount of time depending on parameters...")

    start_time = time.time()

    # Use multiprocessing to run scenarios in parallel
    # Adjust number of processes as needed, os.cpu_count() is a good starting point
    num_processes = os.cpu_count() -1 if os.cpu_count() > 1 else 1 
    print(f"Using {num_processes} worker processes.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # `starmap` is good if your function takes multiple arguments unpacked from tuples
        # `map` is good if your function takes a single argument (like our `params` dict)
        results_list = pool.map(run_single_scenario, all_scenario_params_list)
    
    end_time = time.time()
    print(f"--- All simulations completed in {end_time - start_time:.2f} seconds ---")

    # Convert results to a Pandas DataFrame for easier aggregation and plotting
    results_df = pd.DataFrame(results_list)
    
    # Save raw results to CSV
    results_df.to_csv(CSV_RESULTS_FILENAME, index=False)
    print(f"Raw simulation data saved to {CSV_RESULTS_FILENAME}")

    # --- Generate Plots ---
    print("Generating plots...")

    # Plot 1: Average Biggest Cluster Size vs. Num Anti-Agents, one plot per item density
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

    # Plot 2: Average Biggest Cluster Size vs. Num Anti-Agents, all densities on one plot
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