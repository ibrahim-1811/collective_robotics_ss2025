import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# --- Task 1A: Plot Poisson Distribution ---
print("--- Task 1A ---")

# Function to calculate Poisson probability for a given λ and i
def poisson_probability(lmbda, i):
    return (np.exp(-lmbda) * (lmbda ** i)) / factorial(i)

x_values = range(15)  # Range of X values for the plot
alpha_values = [0.01, 0.1, 0.5, 1]  # Different α values to compare

plt.figure()
for alpha in alpha_values:
    probabilities = [poisson_probability(alpha, i) for i in x_values]
    plt.plot(x_values, probabilities, label=f'α = {alpha}')

plt.xlabel('Number of Arrivals (X)')
plt.ylabel('Probability')
plt.title('Poisson Distribution')
plt.legend()
plt.grid()
plt.savefig('output/task_1_A_poisson_distribution.png')  # Save the figure
plt.show()

# --- Task 1B: Sample from Poisson Distribution ---
print("\n--- Task 1B ---")

# Function to sample from a Poisson distribution with parameter α
def sample_poisson(alpha, num_samples):
    """
    Samples numbers of incoming data from a Poisson distribution using the poisson_probability function.

    Parameters:
    - alpha: The rate parameter (λ = α).
    - num_samples: Number of samples to generate.

    Returns:
    - A list of sampled values.
    """
    # Define a reasonable range for X values
    x_values = range(15)  # Adjust the range as needed
    # Calculate probabilities using poisson_probability
    probabilities = [poisson_probability(alpha, i) for i in x_values]
    # Normalize probabilities to ensure they sum to 1
    probabilities = np.array(probabilities) / sum(probabilities)

    # Sample from the distribution using the calculated probabilities
    samples = np.random.choice(x_values, size=num_samples, p=probabilities)
    return samples

# Example usage
alpha = 0.1  # Set α value
num_samples = 1000  # Number of samples to generate

# Generate samples
samples = sample_poisson(alpha, num_samples)

print(f"Generated {num_samples} samples for α = {alpha}:")
print(samples[:20])  # Print the first 20 samples as an example

# Example 2
alpha = 0.5  # Set α value
num_samples = 1000  # Number of samples to generate

# Generate samples
samples = sample_poisson(alpha, num_samples)

print(f"Generated {num_samples} samples for α = {alpha}:")
print(samples[:20])  # Print the first 20 samples as an example


# --- Task 1C: Single Simulation Run ---
print("\n--- Task 1C ---")

# Function to simulate a data center queue
def simulate_data_center(alpha, processing_duration, time_steps):
    waiting_list = 0  # Number of items in the waiting list
    processing_steps_left = 0  # Steps left for the current item
    list_lengths = []  
    
    for _ in range(time_steps):
        arrivals = sample_poisson(alpha, 1)[0]  # New arrivals at this time step
        waiting_list += arrivals  # Add arrivals to the waiting list

        # Process the current item if processing is ongoing
        if processing_steps_left > 0:
            processing_steps_left -= 1
        # Start processing a new item if the waiting list is not empty
        elif waiting_list > 0:
            waiting_list -= 1
            processing_steps_left = processing_duration - 1

        # Record the current waiting list length
        list_lengths.append(waiting_list)

    # Return the average waiting list length over all time steps
    return sum(list_lengths) / len(list_lengths)

alpha_c = 0.1  # Arrival rate
processing_duration_c = 4  # Processing duration for each item
time_steps_c = 2000  # Total simulation time steps

# Run the simulation and print the average waiting list length
avg_length_c = simulate_data_center(alpha_c, processing_duration_c, time_steps_c)
print(f"Average waiting list length: {avg_length_c:.4f}")

# --- Task 1D: Multiple Simulations (Processing Duration = 4) ---
print("\n--- Task 1D ---")

alphas_d = np.arange(0.005, 0.255, 0.005)  # Range of α values
num_samples_d = 200  # Number of simulation runs for each α
processing_duration_d = 4  # Processing duration
time_steps_d = 2000  # Total simulation time steps
average_lengths_d = []  # To store average waiting list lengths for each α

# Run simulations for each α and calculate the average waiting list length
for alpha in alphas_d:
    run_lengths = [simulate_data_center(alpha, processing_duration_d, time_steps_d) for _ in range(num_samples_d)]
    average_lengths_d.append(sum(run_lengths) / len(run_lengths))
    print(f"α = {alpha:.3f}: Avg Length = {average_lengths_d[-1]:.4f}")

# Plot the results for Task 1D
plt.figure()
plt.plot(alphas_d, average_lengths_d, marker='.')
plt.title('Average Waiting List Length (Processing Duration = 4)')
plt.xlabel('Arrival Rate (α)')
plt.ylabel('Average Waiting List Length')
plt.grid()
plt.savefig('output/task_1_D_average_lengths.png')  # Save the figure
plt.show()

# --- Task 1E: Multiple Simulations (Processing Duration = 2) ---
print("\n--- Task 1E ---")

alphas_e = np.arange(0.005, 0.505, 0.005)  # Range of α values
processing_duration_e = 2  # Processing duration
average_lengths_e = []  # To store average waiting list lengths for each α

# Run simulations for each α and calculate the average waiting list length
for alpha in alphas_e:
    run_lengths = [simulate_data_center(alpha, processing_duration_e, time_steps_d) for _ in range(num_samples_d)]
    average_lengths_e.append(sum(run_lengths) / len(run_lengths))
    print(f"α = {alpha:.3f}: Avg Length = {average_lengths_e[-1]:.4f}")

# Plot the results for Task 1E
plt.figure()
plt.plot(alphas_e, average_lengths_e, marker='.')
plt.title('Average Waiting List Length (Processing Duration = 2)')
plt.xlabel('Arrival Rate (α)')
plt.ylabel('Average Waiting List Length')
plt.grid()
plt.savefig('output/task_1_E_average_lengths.png')  # Save the figure
plt.show()

# --- Comparison Plot ---
print("\n--- Comparison Plot ---")

# Plot the comparison of average waiting list lengths for different processing durations
plt.figure()
plt.plot(alphas_d, average_lengths_d, marker='.', label='Processing Duration = 4')
plt.plot(alphas_e, average_lengths_e, marker='.', label='Processing Duration = 2')
plt.title('Comparison of Average Waiting List Length')
plt.xlabel('Arrival Rate (α)')
plt.ylabel('Average Waiting List Length')
plt.legend()
plt.grid()
plt.savefig('output/task_1_comparison_plot.png')  # Save the figure
plt.show()