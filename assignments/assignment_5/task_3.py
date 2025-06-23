import numpy as np
import matplotlib.pyplot as plt
import time

# --- Task 3: Foraging ---

class ForagingRobot:
    """A robot for a foraging task with a state-based controller. [source: 39]"""
    def __init__(self, x, y, home_zone_pos, arena_size):
        self.pos = np.array([x, y], dtype=float)
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.speed = 0.1
        self.turn_speed = 0.2
        
        self.state = 'SEARCHING' # Initial state
        self.carrying_object = None # To link robot with the object it carries
        
        self.home_zone_pos = np.array(home_zone_pos, dtype=float)
        self.home_zone_radius = 1.5
        self.arena_size = arena_size

    def controller(self, objects):
        """Implements the robot's state-based controller and outputs status. [source: 43]"""
        # --- Sensor Simulation ---
        # Bumper sensor to detect nearby objects [source: 40]
        sensed_object = None
        if not self.carrying_object:
            for obj in objects:
                if not obj.is_carried and np.linalg.norm(self.pos - obj.pos) < 0.5:
                    sensed_object = obj
                    break

        # Light sensors to find home (simulated via position) [source: 42]
        is_in_home_zone = np.linalg.norm(self.pos - self.home_zone_pos) < self.home_zone_radius
        
        vec_to_home = self.home_zone_pos - self.pos
        angle_to_home = np.arctan2(vec_to_home[1], vec_to_home[0])
        # Calculate smallest angle difference to turn efficiently
        angle_diff = (angle_to_home - self.angle + np.pi) % (2 * np.pi) - np.pi

        # --- State Machine Logic ---
        if self.state == 'SEARCHING':
            if sensed_object:
                self.state = 'TRANSPORTING'
                self.carrying_object = sensed_object
                sensed_object.is_carried = True
            else:
                # Random walk to find objects
                self.angle += np.random.uniform(-0.1, 0.1)
        
        elif self.state == 'TRANSPORTING':
            if is_in_home_zone:
                # Drop object in the home zone
                self.carrying_object.is_home = True
                self.carrying_object.is_carried = False
                self.carrying_object = None
                self.state = 'SEARCHING'
                # Turn around to leave home zone and reduce clutter
                self.angle += np.pi
            else:
                # Navigate towards light/home (phototaxis)
                if angle_diff > 0.1: self.angle += self.turn_speed
                elif angle_diff < -0.1: self.angle -= self.turn_speed

        # --- Actuator Control & Physics Simulation ---
        self.pos[0] += self.speed * np.cos(self.angle)
        self.pos[1] += self.speed * np.sin(self.angle)
        
        # If carrying an object, it moves with the robot
        if self.carrying_object:
            self.carrying_object.pos = self.pos
        
        # Handle Arena Boundaries by turning around
        if not (0.1 < self.pos[0] < self.arena_size[0] - 0.1) or \
           not (0.1 < self.pos[1] < self.arena_size[1] - 0.1):
            self.pos = np.clip(self.pos, 0.1, self.arena_size[0] - 0.1)
            self.angle += np.pi

class ForagingObject:
    """Represents a collectible object."""
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.is_carried = False
        self.is_home = False # Final state to count collected objects

def run_foraging_experiment(num_robots, duration):
    """Runs a single foraging simulation and returns the number of collected objects."""
    arena_size = (20, 20)
    home_zone_pos = (2, 2)
    num_objects = 50

    robots = [ForagingRobot(np.random.uniform(5, 15), np.random.uniform(5, 15), home_zone_pos, arena_size) for _ in range(num_robots)]
    objects = [ForagingObject(np.random.uniform(5, 15), np.random.uniform(5, 15)) for _ in range(num_objects)]
    
    for _ in range(duration):
        # Update each robot's controller
        for robot in robots:
            robot.controller(objects)
            
    # Measure performance: count objects successfully brought home [source: 44]
    collected_count = sum(1 for obj in objects if obj.is_home)
    return collected_count

def run_task3():
    """Main function to run foraging experiments and plot results."""
    print("\n--- Running Task 3: Foraging ---")
    start_time = time.time()
    
    swarm_sizes = np.arange(1, 11)  # Test N from 1 to 10 [source: 44]
    performance = []
    sim_duration = 3000  # Fixed period of time
    num_runs_per_size = 5 # Average over several runs for smoother results
    
    for n in swarm_sizes:
        print(f"  Testing swarm size N = {n}...")
        runs = [run_foraging_experiment(n, sim_duration) for _ in range(num_runs_per_size)]
        perf = np.mean(runs)
        performance.append(perf)
        
    print("Simulations complete. Processing results...")

    # Plot swarm performance over swarm size [source: 44]
    plt.figure(figsize=(10, 6))
    plt.plot(swarm_sizes, performance, 'go-', linewidth=2, markersize=8, label='Swarm Performance')
    plt.title('Swarm Foraging Performance vs. Swarm Size', fontsize=14)
    plt.xlabel('Swarm Size (N)', fontsize=12)
    plt.ylabel(f'Objects Collected in {sim_duration} Steps (Avg)', fontsize=12)
    plt.xticks(swarm_sizes)
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()

    end_time = time.time()
    print(f"--- Task 3 finished in {end_time - start_time:.2f} seconds ---")


# --- Execute Task 3 ---
if __name__ == '__main__':

    run_task3()
