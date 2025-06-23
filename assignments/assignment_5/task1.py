import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

# --- Task 1: Urn Model for Locust Scenario ---

class Locust:
    """Represents a single locust with position and direction."""
    def __init__(self, circumference):
        self.C = circumference
        self.pos = np.random.rand() * self.C
        self.direction = np.random.choice([-1, 1])  # -1: left, 1: right

    def update_position(self, speed):
        self.pos += self.direction * speed
        self.pos %= self.C  # Apply periodic boundary conditions

class LocustSimulation:
    """Manages the simulation of the locust swarm."""
    def __init__(self, N, C, r, P, speed):
        # Store parameters as defined in the assignment [source: 3]
        self.N = N
        self.C = C
        self.r = r
        self.P_spontaneous_switch = P
        self.speed = speed
        self.locusts = [Locust(C) for _ in range(N)]
        self.L_t = self.count_left_goers()

    def count_left_goers(self):
        """Counts the number of locusts moving left."""
        return sum(1 for locust in self.locusts if locust.direction == -1)

    def step(self):
        """Executes one time step of the simulation."""
        for i, locust in enumerate(self.locusts):
            # Spontaneous switch with probability P [source: 3]
            if np.random.rand() < self.P_spontaneous_switch:
                locust.direction *= -1
            else:
                # Social interaction based on neighbors in perception range r [source: 3]
                neighbors = []
                for j, other in enumerate(self.locusts):
                    if i == j: continue
                    dist = abs(locust.pos - other.pos)
                    # Account for periodic boundary on the circumference C [source: 3]
                    dist = min(dist, self.C - dist)
                    if dist < self.r:
                        neighbors.append(other)

                if neighbors:
                    # Align with the majority direction of neighbors
                    left_neighbors = sum(1 for n in neighbors if n.direction == -1)
                    right_neighbors = len(neighbors) - left_neighbors
                    if left_neighbors > right_neighbors:
                        locust.direction = -1
                    elif right_neighbors > left_neighbors:
                        locust.direction = 1
            
            locust.update_position(self.speed)
        
        # Update the count of left-goers for the new state [source: 4]
        self.L_t = self.count_left_goers()


def run_task1():
    """Main function to run simulation, analyze, and plot for Task 1."""
    print("--- Running Task 1: Urn Model ---")
    start_time = time.time()
    
    # Simulation Parameters from the assignment [source: 3]
    N = 50
    C = 0.5
    r = 0.045
    P = 0.15
    speed = 0.01
    
    # Data Collection Parameters
    warm_up_steps = 100  # [source: 8]
    measure_steps = 20   # [source: 9]
    num_samples = 500  # [source: 12]

    deltaSum = np.zeros(N + 1)
    count = np.zeros(N + 1)

    print(f"Collecting {num_samples} samples. This may take a few minutes...")
    for i in range(num_samples):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{num_samples} samples collected...")
        
        sim = LocustSimulation(N, C, r, P, speed)
        
        # Warm-up period to let the system self-organize [source: 8]
        for _ in range(warm_up_steps):
            sim.step()
        
        # Measurement period [source: 9]
        L_t_minus_1 = sim.L_t
        for _ in range(measure_steps):
            sim.step()
            L_t = sim.L_t
            deltaL = L_t - L_t_minus_1 # Calculate change in left-goers [source: 9]
            
            # Store data for each time step [source: 10, 11]
            if 0 <= L_t_minus_1 <= N:
                deltaSum[L_t_minus_1] += deltaL
                count[L_t_minus_1] += 1
            
            L_t_minus_1 = L_t

    print("Data collection complete. Processing results...")
    
    # Calculate average delta_L, avoiding division by zero for unobserved states
    L_values = np.arange(N + 1)
    valid_indices = count > 0
    L_observed = L_values[valid_indices]
    deltaL_avg = deltaSum[valid_indices] / count[valid_indices]
    
    # A. Plot the measured data for delta_L(L) [source: 14]
    plt.figure(figsize=(10, 6))
    plt.plot(L_observed, deltaL_avg, 'o', markersize=4, label='Measured Data $\Delta L(L)$')
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.title('A. Average Change in Left-Goers vs. Number of Left-Goers', fontsize=14)
    plt.xlabel('Number of Left-Goers (L)', fontsize=12)
    plt.ylabel('Average Change ($\Delta L$)', fontsize=12)
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()

    # B. Fit the swarm urn model to the data [source: 15]
    # Convert data to fractions s = L/N [source: 17]
    s_observed = L_observed / N
    delta_s_observed = deltaL_avg / N
    
    # Define the model function to fit, with scaling constant c [source: 17]
    def urn_model(s, c, phi):
        P_FB = phi * np.sin(np.pi * s)  # Probability of positive feedback [source: 15]
        return 4 * c * (P_FB - 0.5) * (s - 0.5) # Expected change in fraction [source: 17]

    # Fit the model, with bounds on phi [0, 1] [source: 18]
    popt, _ = curve_fit(urn_model, s_observed, delta_s_observed, bounds=([0, 0], [np.inf, 1.0]))
    c_fit, phi_fit = popt
    print(f"\nFitted Parameters:\n  Scaling constant (c) = {c_fit:.4f}\n  Feedback intensity (phi) = {phi_fit:.4f}")

    # C. Plot data together with the fitted function and P_FB [source: 19]
    s_model = np.linspace(0, 1, 200)
    delta_s_model = urn_model(s_model, c_fit, phi_fit)
    
    plt.figure(figsize=(10, 6))
    plt.plot(s_observed, delta_s_observed, 'o', markersize=4, label='Measured Data ($\Delta s(s)$)')
    plt.plot(s_model, delta_s_model, 'r-', linewidth=2, label=f'Fitted Urn Model (c={c_fit:.2f}, $\phi$={phi_fit:.2f})')
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.title('C. Measured Data vs. Fitted Urn Model', fontsize=14)
    plt.xlabel('Fraction of Left-Goers (s = L/N)', fontsize=12)
    plt.ylabel('Average Change in Fraction ($\Delta s$)', fontsize=12)
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()
    
    # Plot the resulting probability of positive feedback P_FB [source: 19]
    P_FB_model = phi_fit * np.sin(np.pi * s_model)
    plt.figure(figsize=(10, 6))
    plt.plot(s_model, P_FB_model, 'g-', linewidth=2)
    plt.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='P=0.5 (No Bias)')
    plt.title('C. Resulting Probability of Positive Feedback ($P_{FB}$)', fontsize=14)
    plt.xlabel('Fraction of Left-Goers (s = L/N)', fontsize=12)
    plt.ylabel('Probability of Positive Feedback ($P_{FB}$)', fontsize=12)
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()
    
    end_time = time.time()
    print(f"--- Task 1 finished in {end_time - start_time:.2f} seconds ---")


# --- Execute Task 1 ---
if __name__ == '__main__':
    run_task1()
