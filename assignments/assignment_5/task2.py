import numpy as np
import matplotlib.pyplot as plt
import time

# --- Task 2: Density-dependent global switching ---

# The Locust and LocustSimulation classes are reused from Task 1.
class Locust:
    """Represents a single locust with position and direction."""
    def __init__(self, circumference):
        self.C = circumference
        self.pos = np.random.rand() * self.C
        self.direction = np.random.choice([-1, 1])

    def update_position(self, speed):
        self.pos += self.direction * speed
        self.pos %= self.C

class LocustSimulation:
    """Manages the simulation of the locust swarm."""
    def __init__(self, N, C, r, P, speed):
        self.N = N
        self.C = C
        self.r = r
        self.P_spontaneous_switch = P
        self.speed = speed
        self.locusts = [Locust(C) for _ in range(N)]
        self.L_t = self.count_left_goers()

    def count_left_goers(self):
        return sum(1 for locust in self.locusts if locust.direction == -1)

    def step(self):
        """Executes one time step of the simulation."""
        for i, locust in enumerate(self.locusts):
            if np.random.rand() < self.P_spontaneous_switch:
                locust.direction *= -1
            else:
                neighbors = []
                for j, other in enumerate(self.locusts):
                    if i == j: continue
                    dist = abs(locust.pos - other.pos)
                    dist = min(dist, self.C - dist)
                    if dist < self.r:
                        neighbors.append(other)
                if neighbors:
                    left_neighbors = sum(1 for n in neighbors if n.direction == -1)
                    right_neighbors = len(neighbors) - left_neighbors
                    if left_neighbors > right_neighbors:
                        locust.direction = -1
                    elif right_neighbors > left_neighbors:
                        locust.direction = 1
            locust.update_position(self.speed)
        self.L_t = self.count_left_goers()

def get_zone(L, N):
    """Determines the current zone based on L and N as per the assignment. [source: 28]"""
    if L > 0.7 * N: return 'A' # Strong right-going majority
    if L < 0.3 * N: return 'C' # Strong left-going majority
    return 'B' # Transition/undecided zone

def run_task2():
    """Main function to run simulation and analyze switching for Task 2."""
    print("\n--- Running Task 2: Density-dependent switching ---")
    start_time = time.time()
    
    # Parameters from task 1
    C = 0.5
    r = 0.045
    P = 0.15
    speed = 0.01
    sim_duration = 10000  # Run for at least 5000 steps to observe switches [source: 33]

    swarm_sizes_N = np.arange(20, 151, 10)  # Test N in [20, 150] [source: 26]
    avg_switch_times = []
    num_switches_observed = []

    for N in swarm_sizes_N:
        print(f"  Testing swarm size N = {N}...")
        sim = LocustSimulation(N, C, r, P, speed)
        switch_durations = []
        
        timer = 0
        came_from = None  # Tracks which zone (A or C) we started from [source: 29]

        for t in range(sim_duration):
            sim.step()
            L = sim.L_t
            current_zone = get_zone(L, N)
            
            # Logic to start, reset, or record the timer for a switch
            if came_from is None:
                if current_zone in ['A', 'C']:
                    came_from = current_zone
            elif current_zone == 'B':
                timer += 1 # Start/continue counting when in zone B [source: 29]
            elif current_zone == came_from:
                timer = 0 # Reset if we fall back to the original zone [source: 31]
            else: # We must have entered the opposite zone for a successful switch
                if (came_from == 'A' and current_zone == 'C') or \
                   (came_from == 'C' and current_zone == 'A'):
                    if timer > 0:
                        switch_durations.append(timer) # Store the time [source: 30]
                # Reset for the next switch attempt
                timer = 0
                came_from = current_zone
        
        if switch_durations:
            avg_time = np.mean(switch_durations)
        else:
            avg_time = np.nan # Use NaN (Not a Number) if no switches were observed
            
        avg_switch_times.append(avg_time)
        num_switches_observed.append(len(switch_durations))

    print("Simulations complete. Processing results...")

    # Plotting results [source: 35, 36]
    plt.figure(figsize=(10, 6))
    plt.plot(swarm_sizes_N, avg_switch_times, 'bo-', label='Average Switch Time')
    plt.title('Average Global Switch Time vs. Swarm Size', fontsize=14)
    plt.xlabel('Swarm Size (N)', fontsize=12)
    plt.ylabel('Average Time Steps Between Switches', fontsize=12)
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(swarm_sizes_N, num_switches_observed, 'ro-', label='Number of Switches')
    plt.title('Number of Observed Global Switches vs. Swarm Size', fontsize=14)
    plt.xlabel('Swarm Size (N)', fontsize=12)
    plt.ylabel(f'Switches Observed in {sim_duration} Steps', fontsize=12)
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()
    
    end_time = time.time()
    print(f"--- Task 2 finished in {end_time - start_time:.2f} seconds ---")


# --- Execute Task 2 ---
if __name__ == '__main__':
    run_task2()
