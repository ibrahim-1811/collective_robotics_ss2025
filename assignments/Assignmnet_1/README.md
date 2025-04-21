# Collective Robotics - Task Sheet 1 Solution

**Group Members:** 
- Mohammad Ibrahim Memon
- Ujjwal Patil
---

## Dependencies

The solutions are implemented in **Python 3** and require the following libraries:

- NumPy
- Matplotlib

You can install these dependencies using pip:

```bash
pip install numpy matplotlib
```

---

## How to Compile & Start

1. Navigate to the directory containing the scripts in your terminal.

2. Run the script for Task 1:

   ```bash
   python task_1_scaling.py
   ```

3. Run the script for Task 2:

   ```bash
   python task_2_A.py
   python task_2_B.py
   ```

The scripts will execute the simulations and generate the required plots as image files in the `output` directory. Console output will provide specific results and progress information.

---

## Script Explanations

### Task 1: Scaling of a Data Center (`task1_scaling.py` or relevant section)

This part simulates a simple data center model based on queuing theory, analyzing the waiting list length under varying arrival rates ($\alpha$) and processing durations.

- **1A**: Calculates and plots the Poisson probability distribution\
  $P(X=i)=\frac{e^{-\lambda}\lambda^{i}}{i!}$ (with $\lambda=\alpha$, $\Delta t=1$) for $\alpha \in {0.01, 0.1, 0.5, 1}$.\
  *Saved as:* `task_1_A_poisson_distribution.png`

- **1B**: Implements a function `sample_poisson(alpha)` to sample incoming data numbers from the Poisson distribution. *Used internally.*

- **1C**: Implements the simulation model.\
  Runs a single simulation (2000 time steps) with $\alpha=0.1$ and processing duration of 4 steps.\
  *Prints the average waiting list length to the console.*

- **1D**: Averages the waiting list length over 200 independent runs (2000 steps each) for\
  $\alpha \in [0.005, 0.25]$ (step 0.005) with processing duration 4.\
  *Saved as:* `task_1_D_average_lengths.png`

- **1E**: Repeats 1D with processing duration 2 steps and $\alpha \in [0.005, 0.5]$ (step 0.005).\
  *Saved as:* `task_1_E_average_lengths.png`

- **Comparison Plot**: Compares the results of **1D** and **1E**.\
  *Saved as:* `task_1_comparison_plot.png`

---

### Task 2: Synchronization of a Swarm (`task2_synchronization.py` or relevant section)

This part simulates a model of firefly synchronization based on local neighbor interactions. Fireflies are randomly placed in a 1x1 square and adjust their flashing cycle based on neighbors within a vicinity distance $r$. The cycle length is $L = 50$.

- **2A**: Implements the model for $N = 250$ fireflies.\
  Calculates and prints the average number of neighbors per firefly for $r \in {0.05, 0.1, 0.5, 1.4}$.

  Runs simulations for 5000 time steps for each $r$ and plots the number of currently flashing fireflies over time.\
  *Saved as:* `task2a_flashing_r0.05.png`, `task2a_flashing_r0.1.png`, etc.\
  *Vertical axis:* [0, N] (or [0, 150] as specified)

- **2B**: Extends the model to calculate synchronization amplitude.\
  Determines the min/max number of flashing fireflies in the last cycle (last 50 steps).\
  Calculates `amplitude = (max - min) / 2`.

  Averages this amplitude over 50 independent runs (5000 steps each) for vicinities\
  $r \in [0.025, 1.4]$ (step 0.025).\
  *Plots average amplitude vs. vicinity $r.*\
  *Saved as:* `task2b_amplitude_vs_vicinity.png`\
  *Includes console output with insights about good vicinity choices.*

---

