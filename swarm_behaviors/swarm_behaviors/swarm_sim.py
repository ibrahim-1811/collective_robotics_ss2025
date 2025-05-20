import random
import numpy as np
import csv
from collections import deque # For BFS in cluster finding
import matplotlib.pyplot as plt
import time # For pausing

# --- Simulation Parameters (Adjusted for Visualization) ---
GRID_WIDTH = 30         # Smaller grid for faster display
GRID_HEIGHT = 30
NUM_ITEMS_INITIAL = int(0.1 * GRID_WIDTH * GRID_HEIGHT) # 10% density

K_PLUS = 0.05
K_MINUS = 0.3
NEIGHBORHOOD_RADIUS = 3 # Smaller radius for faster calculation during viz

NUM_NORMAL_AGENTS = 20

# --- VISUALIZATION SETTINGS ---
VISUALIZE = True  # Set to True to see the simulation, False to run without visuals
VISUALIZATION_UPDATE_RATE = 10  # Update visual every X steps
SINGLE_RUN_NUM_ANTI_AGENTS = 5 # Number of anti-agents for this single visual run

# --- Parameters for data collection (if VISUALIZE is False) ---
ANTI_AGENT_CONFIGURATIONS_DATA = [0, 5, 10, 15, 20] # Used if VISUALIZE is False
SIMULATION_STEPS_DATA = 5000  # For data collection
NUM_REPETITIONS_DATA = 3      # For data collection
OUTPUT_FILENAME = "simulation_results.csv"

# --- Simulation Parameters (used by the run) ---
# These will be set based on whether we are visualizing or collecting data
SIMULATION_STEPS = 10000 # Default for visualization
NUM_REPETITIONS = 5     # Default for visualization
ANTI_AGENT_CONFIGURATIONS = [SINGLE_RUN_NUM_ANTI_AGENTS] # Default for viz

# --- Grid Representation ---
grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

# --- Agent Class (Same as before) ---
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

    def calculate_local_density_f(self, current_grid):
        neighborhood_cells_count = 0
        occupied_cells_count = 0
        for dx in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
            for dy in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
                if abs(dx) + abs(dy) <= NEIGHBORHOOD_RADIUS:
                    check_x = (self.x + dx + self.grid_width) % self.grid_width
                    check_y = (self.y + dy + self.grid_height) % self.grid_height
                    neighborhood_cells_count += 1
                    if current_grid[check_y, check_x] == 1:
                        occupied_cells_count += 1
        if neighborhood_cells_count == 0: return 0.0
        return occupied_cells_count / neighborhood_cells_count

    def pick_drop_decision(self, current_grid):
        local_f = self.calculate_local_density_f(current_grid)
        p_pick_std = (K_PLUS / (K_PLUS + local_f))**2 if (K_PLUS + local_f) > 0 else 0
        p_drop_std = (local_f / (K_MINUS + local_f))**2 if (K_MINUS + local_f) > 0 else 0
        
        p_pick_actual, p_drop_actual = p_pick_std, p_drop_std
        if self.type == 'anti':
            p_pick_actual = 1.0 - p_pick_std
            p_drop_actual = 1.0 - p_drop_std
            
        if not self.is_laden and current_grid[self.y, self.x] == 1:
            if random.random() < p_pick_actual:
                self.is_laden = True
                current_grid[self.y, self.x] = 0
        elif self.is_laden and current_grid[self.y, self.x] == 0:
            if random.random() < p_drop_actual:
                self.is_laden = False
                current_grid[self.y, self.x] = 1

# --- Cluster Analysis (Same as before) ---
def find_biggest_cluster(current_grid):
    rows, cols = current_grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    max_cluster_size = 0
    for r in range(rows):
        for c in range(cols):
            if current_grid[r, c] == 1 and not visited[r, c]:
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
                           current_grid[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            current_cluster_size += 1
                max_cluster_size = max(max_cluster_size, current_cluster_size)
    return max_cluster_size

# --- Visualization Function ---
fig, ax = None, None # Global figure and axis for visualization

def setup_visualization():
    global fig, ax
    fig, ax = plt.subplots()
    plt.ion() # Turn on interactive mode
    fig.show()

def visualize_grid(current_grid, agents_list, step_num, num_anti_val):
    if not VISUALIZE or not fig:
        return

    ax.clear() # Clear previous frame
    
    # Display items (1 = item, 0 = empty)
    # We can use a colormap: 0 maps to white, 1 maps to black
    display_grid = np.copy(current_grid).astype(float) # Make a copy for display
    
    ax.imshow(display_grid, cmap='Greys', origin='lower', vmin=0, vmax=1)

    # Display agents
    normal_agents_x = [agent.x for agent in agents_list if agent.type == 'normal']
    normal_agents_y = [agent.y for agent in agents_list if agent.type == 'normal']
    anti_agents_x = [agent.x for agent in agents_list if agent.type == 'anti']
    anti_agents_y = [agent.y for agent in agents_list if agent.type == 'anti']

    # Agent colors and laden status
    # Normal: blue (laden: darkblue), Anti: red (laden: darkred)
    normal_colors = ['darkblue' if agent.is_laden else 'blue' for agent in agents_list if agent.type == 'normal']
    anti_colors = ['darkred' if agent.is_laden else 'red' for agent in agents_list if agent.type == 'anti']

    ax.scatter(normal_agents_x, normal_agents_y, c=normal_colors, marker='o', s=50, label='Normal Agents')
    ax.scatter(anti_agents_x, anti_agents_y, c=anti_colors, marker='x', s=50, label='Anti Agents')
    
    ax.set_title(f"Step: {step_num}, Items: {np.sum(current_grid)}, Anti-Agents: {num_anti_val}")
    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(-0.5, GRID_HEIGHT - 0.5)
    # ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0)) # legend can make it crowded
    
    plt.draw()
    plt.pause(0.001) # Pause briefly to allow the plot to update

# --- Main Simulation ---
def run_simulation():
    global SIMULATION_STEPS, NUM_REPETITIONS, ANTI_AGENT_CONFIGURATIONS

    if VISUALIZE:
        SIMULATION_STEPS = 20000  # Shorter for visualization
        NUM_REPETITIONS = 1
        ANTI_AGENT_CONFIGURATIONS = [SINGLE_RUN_NUM_ANTI_AGENTS]
        setup_visualization()
        print(f"--- Running SINGLE VISUALIZATION with {SINGLE_RUN_NUM_ANTI_AGENTS} anti-agents ---")
    else:
        SIMULATION_STEPS = SIMULATION_STEPS_DATA
        NUM_REPETITIONS = NUM_REPETITIONS_DATA
        ANTI_AGENT_CONFIGURATIONS = ANTI_AGENT_CONFIGURATIONS_DATA
        print(f"--- Running DATA COLLECTION for {len(ANTI_AGENT_CONFIGURATIONS_DATA)} configurations ---")

    # Prepare CSV file (only if not visualizing, or if you want data from viz run too)
    if not VISUALIZE: # Or always write if you want data from viz runs
        csv_file_exists = False
        try:
            with open(OUTPUT_FILENAME, 'r') as f_check:
                csv_file_exists = True
        except FileNotFoundError:
            pass
        
        with open(OUTPUT_FILENAME, 'a' if csv_file_exists else 'w', newline='') as csvfile:
            fieldnames = ['num_anti_agents', 'repetition', 'biggest_cluster_size', 'total_items_end']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not csv_file_exists:
                writer.writeheader()

            core_simulation_loop(writer)
    else: # Visualization mode, don't write to CSV by default for this example
        core_simulation_loop(None) # Pass None for writer

    if VISUALIZE:
        print("Visualization finished. Close the plot window to exit.")
        plt.ioff() # Turn off interactive mode
        plt.show() # Keep window open until manually closed
    else:
        print(f"Simulations complete. Results appended/saved to {OUTPUT_FILENAME}")


def core_simulation_loop(csv_writer):
    for num_anti in ANTI_AGENT_CONFIGURATIONS:
        print(f"  Testing with {num_anti} anti-agents...")
        for rep in range(NUM_REPETITIONS):
            print(f"    Repetition {rep + 1}/{NUM_REPETITIONS}")
            
            current_grid_sim = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
            items_placed = 0
            while items_placed < NUM_ITEMS_INITIAL:
                x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
                if current_grid_sim[y, x] == 0:
                    current_grid_sim[y, x] = 1
                    items_placed += 1

            agents = []
            agent_id_counter = 0
            for _ in range(NUM_NORMAL_AGENTS):
                agents.append(Agent(agent_id_counter, 'normal', GRID_WIDTH, GRID_HEIGHT))
                agent_id_counter +=1
            for _ in range(num_anti):
                agents.append(Agent(agent_id_counter, 'anti', GRID_WIDTH, GRID_HEIGHT))
                agent_id_counter +=1
            random.shuffle(agents)

            if VISUALIZE: # Initial frame
                 visualize_grid(current_grid_sim, agents, 0, num_anti)
                 time.sleep(1)


            for step in range(SIMULATION_STEPS):
                if (step + 1) % (SIMULATION_STEPS // 20) == 0 and SIMULATION_STEPS >= 20 : # Print progress
                        print(f"      Step {step + 1}/{SIMULATION_STEPS}")
                
                for agent_obj in agents: # Renamed to avoid conflict
                    agent_obj.move()
                    agent_obj.pick_drop_decision(current_grid_sim)
                
                if VISUALIZE and (step + 1) % VISUALIZATION_UPDATE_RATE == 0:
                    visualize_grid(current_grid_sim, agents, step + 1, num_anti)
            
            if VISUALIZE: # Final frame
                visualize_grid(current_grid_sim, agents, SIMULATION_STEPS, num_anti)
                print(f"    Finished Visual Repetition {rep + 1}. Biggest cluster: {find_biggest_cluster(current_grid_sim)}")
            
            if csv_writer: # Only write if a writer is provided
                biggest_cluster = find_biggest_cluster(current_grid_sim)
                total_items_end = np.sum(current_grid_sim)
                print(f"    Finished Data Repetition {rep + 1}. Biggest cluster: {biggest_cluster}, Total Items: {total_items_end}")
                csv_writer.writerow({
                    'num_anti_agents': num_anti,
                    'repetition': rep + 1,
                    'biggest_cluster_size': biggest_cluster,
                    'total_items_end': total_items_end
                })

if __name__ == "__main__":
    run_simulation()