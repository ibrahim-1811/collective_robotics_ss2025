import random
import numpy as np
import matplotlib.pyplot as plt
import time # For pausing
from collections import deque # For BFS in cluster finding (still useful to see biggest cluster in console)

# --- Simulation Parameters (Adjusted for Visualization) ---
GRID_WIDTH = 30         # Smaller grid for faster display
GRID_HEIGHT = 30
NUM_ITEMS_INITIAL = int(0.1 * GRID_WIDTH * GRID_HEIGHT) # 10% density

K_PLUS = 0.05
K_MINUS = 0.3
NEIGHBORHOOD_RADIUS = 7 # Smaller radius for faster calculation during viz

NUM_NORMAL_AGENTS = 10
NUM_ANTI_AGENTS_TO_VISUALIZE = 0 # Number of anti-agents for this visual run

SIMULATION_STEPS = 10000  # Number of steps for the visual simulation
VISUALIZATION_UPDATE_RATE = 50  # Update visual every X steps

# --- Grid Representation ---
# 0: empty cell
# 1: cell with an item
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

    def calculate_local_density_f(self, current_grid_state): # Renamed parameter
        neighborhood_cells_count = 0
        occupied_cells_count = 0
        for dx in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
            for dy in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
                if abs(dx) + abs(dy) <= NEIGHBORHOOD_RADIUS:
                    check_x = (self.x + dx + self.grid_width) % self.grid_width
                    check_y = (self.y + dy + self.grid_height) % self.grid_height
                    neighborhood_cells_count += 1
                    if current_grid_state[check_y, check_x] == 1: # Use renamed parameter
                        occupied_cells_count += 1
        if neighborhood_cells_count == 0: return 0.0
        return occupied_cells_count / neighborhood_cells_count

    def pick_drop_decision(self, current_grid_state): # Renamed parameter
        local_f = self.calculate_local_density_f(current_grid_state) # Use renamed parameter
        p_pick_std = (K_PLUS / (K_PLUS + local_f))**2 if (K_PLUS + local_f) > 0 else 0
        p_drop_std = (local_f / (K_MINUS + local_f))**2 if (K_MINUS + local_f) > 0 else 0
        
        p_pick_actual, p_drop_actual = p_pick_std, p_drop_std
        if self.type == 'anti':
            p_pick_actual = 1.0 - p_pick_std
            p_drop_actual = 1.0 - p_drop_std
            
        if not self.is_laden and current_grid_state[self.y, self.x] == 1: # Use renamed parameter
            if random.random() < p_pick_actual:
                self.is_laden = True
                current_grid_state[self.y, self.x] = 0 # Use renamed parameter
        elif self.is_laden and current_grid_state[self.y, self.x] == 0: # Use renamed parameter
            if random.random() < p_drop_actual:
                self.is_laden = False
                current_grid_state[self.y, self.x] = 1 # Use renamed parameter

# --- Cluster Analysis (Same as before, for console output) ---
def find_biggest_cluster(current_grid_state): # Renamed parameter
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

# --- Visualization Function ---
fig, ax = None, None 

def setup_visualization():
    global fig, ax
    fig, ax = plt.subplots(figsize=(8,8)) # Make figure a bit bigger
    plt.ion() 
    fig.show()

def visualize_grid_state(current_grid_display, agents_list, step_num, num_anti_val): # Renamed
    if not fig: # Check if figure exists
        return

    ax.clear() 
    display_grid_viz = np.copy(current_grid_display).astype(float)
    
    ax.imshow(display_grid_viz, cmap='Greys', origin='lower', vmin=0, vmax=1)

    normal_agents_x = [agent.x for agent in agents_list if agent.type == 'normal']
    normal_agents_y = [agent.y for agent in agents_list if agent.type == 'normal']
    anti_agents_x = [agent.x for agent in agents_list if agent.type == 'anti']
    anti_agents_y = [agent.y for agent in agents_list if agent.type == 'anti']

    normal_colors = ['darkblue' if agent.is_laden else 'blue' for agent in agents_list if agent.type == 'normal']
    anti_colors = ['darkred' if agent.is_laden else 'red' for agent in agents_list if agent.type == 'anti']

    ax.scatter(normal_agents_x, normal_agents_y, c=normal_colors, marker='o', s=60, label='Normal Agents', edgecolors='black')
    ax.scatter(anti_agents_x, anti_agents_y, c=anti_colors, marker='X', s=70, label='Anti Agents', edgecolors='black')
    
    ax.set_title(f"Step: {step_num}, Items: {np.sum(current_grid_display)}, Anti-Agents: {num_anti_val}")
    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(-0.5, GRID_HEIGHT - 0.5)
    # Optional: Add legend if not too crowded
    if len(agents_list) < 30 : # Only show legend if not too many agents
         ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    fig.canvas.draw_idle() # Use draw_idle for better responsiveness
    plt.pause(0.0001) 

# --- Main Simulation ---
def run_visual_simulation():
    setup_visualization()
    print(f"--- Running VISUALIZATION with {NUM_ANTI_AGENTS_TO_VISUALIZE} anti-agents ---")
    
    # Initialize grid
    current_simulation_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    items_placed = 0
    while items_placed < NUM_ITEMS_INITIAL:
        x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
        if current_simulation_grid[y, x] == 0:
            current_simulation_grid[y, x] = 1
            items_placed += 1

    # Initialize agents
    all_agents = []
    agent_id_counter = 0
    for _ in range(NUM_NORMAL_AGENTS):
        all_agents.append(Agent(agent_id_counter, 'normal', GRID_WIDTH, GRID_HEIGHT))
        agent_id_counter +=1
    for _ in range(NUM_ANTI_AGENTS_TO_VISUALIZE):
        all_agents.append(Agent(agent_id_counter, 'anti', GRID_WIDTH, GRID_HEIGHT))
        agent_id_counter +=1
    random.shuffle(all_agents)

    # Initial frame
    visualize_grid_state(current_simulation_grid, all_agents, 0, NUM_ANTI_AGENTS_TO_VISUALIZE)
    time.sleep(1) # Pause for a second to see initial state

    # Simulation loop
    for step in range(SIMULATION_STEPS):
        if (step + 1) % (SIMULATION_STEPS // 20) == 0 and SIMULATION_STEPS >= 20:
            print(f"  Step {step + 1}/{SIMULATION_STEPS}")
        
        for agent in all_agents:
            agent.move()
            agent.pick_drop_decision(current_simulation_grid)
        
        if (step + 1) % VISUALIZATION_UPDATE_RATE == 0:
            visualize_grid_state(current_simulation_grid, all_agents, step + 1, NUM_ANTI_AGENTS_TO_VISUALIZE)
    
    # Final frame and stats
    visualize_grid_state(current_simulation_grid, all_agents, SIMULATION_STEPS, NUM_ANTI_AGENTS_TO_VISUALIZE)
    biggest_cluster = find_biggest_cluster(current_simulation_grid)
    print(f"Visualization finished. Biggest cluster: {biggest_cluster}")
    
    plt.ioff() 
    plt.show() # Keep window open

if __name__ == "__main__":
    run_visual_simulation()