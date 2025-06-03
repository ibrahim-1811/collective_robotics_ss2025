import random
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque

# --- Simulation Parameters ---
GRID_WIDTH = 30
GRID_HEIGHT = 30

ITEM_DENSITY_CONFIG = 0.1  
NUM_ITEMS_INITIAL = int(ITEM_DENSITY_CONFIG * GRID_WIDTH * GRID_HEIGHT)

K_PLUS = 0.05  # k+ from the paper
K_MINUS = 0.3  # k- from the paper
NEIGHBORHOOD_RADIUS = 7  # Neighborhood radius for local density calculation

NUM_NORMAL_AGENTS = 10 
NUM_ANTI_AGENTS_TO_VISUALIZE = 5  

SIMULATION_STEPS = 100000 
VISUALIZATION_UPDATE_RATE = 200  

# --- Grid Representation ---
# 0: empty cell, 1: cell with an item
grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

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
        """
        Calculate the fraction of occupied cells in the agent's neighborhood.
        """
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

    # ** CORRECTED METHOD NAME **
    def pick_drop_decision(self, current_grid_state):
        """
        Decide whether to pick up or drop an item based on local density and agent type.
        """
        local_f = self.calculate_local_density_f(current_grid_state)
        # Using K_PLUS and K_MINUS in the formulas
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

# --- Visualization Setup ---
fig, ax = None, None

def setup_visualization(): 
    global fig, ax
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()
    fig.show()

def visualize_grid_state(current_grid_display, agents_list, step_num, num_normal_agents_val, num_anti_val): 
    if not fig:
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

    total_agents = num_normal_agents_val + num_anti_val
    ax.set_title(f"Step: {step_num}, Items: {np.sum(current_grid_display)}, Normal: {num_normal_agents_val}, Anti: {num_anti_val}")
    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(-0.5, GRID_HEIGHT - 0.5)
    if total_agents < 30: 
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    fig.canvas.draw_idle()
    plt.pause(0.0001)

def run_visual_simulation():
    setup_visualization()
    print(f"--- Running VISUALIZATION: Normal Agents: {NUM_NORMAL_AGENTS}, Anti-agents: {NUM_ANTI_AGENTS_TO_VISUALIZE} ---")
    print(f"Item Density: {ITEM_DENSITY_CONFIG*100:.1f}%, Total Items: {NUM_ITEMS_INITIAL}")


    current_simulation_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    items_placed = 0
    while items_placed < NUM_ITEMS_INITIAL:
        x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
        if current_simulation_grid[y, x] == 0:
            current_simulation_grid[y, x] = 1
            items_placed += 1

    all_agents = []
    agent_id_counter = 0
    for _ in range(NUM_NORMAL_AGENTS):
        all_agents.append(Agent(agent_id_counter, 'normal', GRID_WIDTH, GRID_HEIGHT))
        agent_id_counter += 1
    for _ in range(NUM_ANTI_AGENTS_TO_VISUALIZE):
        all_agents.append(Agent(agent_id_counter, 'anti', GRID_WIDTH, GRID_HEIGHT))
        agent_id_counter += 1
    random.shuffle(all_agents)

    visualize_grid_state(current_simulation_grid, all_agents, 0, NUM_NORMAL_AGENTS, NUM_ANTI_AGENTS_TO_VISUALIZE)
    time.sleep(1)

    for step in range(SIMULATION_STEPS):
        if (step + 1) % (SIMULATION_STEPS // 100) == 0 and SIMULATION_STEPS >= 100: 
            print(f"  Step {step + 1}/{SIMULATION_STEPS}")

        for agent in all_agents:
            agent.move()
            agent.pick_drop_decision(current_simulation_grid) 

        if (step + 1) % VISUALIZATION_UPDATE_RATE == 0:
            visualize_grid_state(current_simulation_grid, all_agents, step + 1, NUM_NORMAL_AGENTS, NUM_ANTI_AGENTS_TO_VISUALIZE)

    visualize_grid_state(current_simulation_grid, all_agents, SIMULATION_STEPS, NUM_NORMAL_AGENTS, NUM_ANTI_AGENTS_TO_VISUALIZE)
    biggest_cluster = find_biggest_cluster(current_simulation_grid)
    print(f"Visualization finished. Biggest cluster: {biggest_cluster}")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_visual_simulation()