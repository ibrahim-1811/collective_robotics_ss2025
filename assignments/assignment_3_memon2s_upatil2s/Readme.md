# Buffon's Needle Simulation and Analysis
This project simulates Buffon's Needle experiment to estimate the value of π and explores the probabilistic nature of the results across multiple trials and densities. The project is structured in modular Python scripts and includes visualizations for detailed analysis.

## Project Structure

```
assignment_3/
├── assignment_3_task_1.py
├── detailes_sim.py
├── parallel_experiment.py
├── visualize_woodchunk.py
├── results_plots/
│   ├── experiment_summary_data.csv
│   ├── plot_all_densities_comparison.png
│   ├── plot_density_5.png
│   ├── plot_density_10.png
│   ├── plot_density_20.png
├── standard deviation over the number of trials n.png
├── ratio over the number of trials n.png
├── probability of many experiments.png

```
How to Run
Install requirements (if any):
```
pip install matplotlib numpy
```
Run basic simulation:
```
python3 assignment_3_task_1.py

```
Run parallel experiments:
```
python3 parallel_experiment.py
```

Visualize results:
```
python3 visualize_woodchunk.py
```


## Results
### experiment_summary_data.csv
Contains aggregated results from parallel simulations.

Includes fields like number of trials, estimated π, standard deviation, etc.

plot_all_densities_comparison.png
Purpose: Compares results across different needle densities.

Insight: Highlights how π estimation varies with density and how convergence stabilizes.

plot_density_5.png, plot_density_10.png, plot_density_20.png

Purpose: Individual plots for each density scenario.

Insight: Provide zoomed-in views of performance for densities 5, 10, and 20.

