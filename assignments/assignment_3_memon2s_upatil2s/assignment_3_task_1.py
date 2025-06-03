#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math


b = 0.7               # Needle length
s = 1.0               # Line spacing
true_pi = math.pi     # True value of pi
true_P = (2 * b) / (s * true_pi)  # Theoretical intersection probability

#Buffon's Needle experiment

def simulate_buffon(n, b, s):
    theta = np.random.uniform(0, np.pi / 2, size=n)   # random angles between 0 and pi/2
    d = np.random.uniform(0, s / 2, size=n)            # random distances from needle center to nearest line
    hits = d <= (b / 2) * np.sin(theta)                # check if each needle hits a line
    hit_count = np.sum(hits)                           # count total hits
    return hit_count / n                               # return estimated probability


#A.

P_estimate = simulate_buffon(100000, b, s)             # run large experiment for good estimate
pi_estimate = (2 * b) / (s * P_estimate)               # estimate pi from result


#B.
n_values = list(range(10, 1001, 10))                   # from 10 to 1000
num_experiments = 1000                                 # number of experiments per n
std_devs = []                                          # to store std values

for n in n_values:
    results = [simulate_buffon(n, b, s) for _ in range(num_experiments)]
    std_devs.append(np.std(results))


#C.
n_small = list(range(1, 101))  # trial numbers from 1 to 100
avg_probs = []
ci_low = []
ci_high = []

for n in n_small:
    p_list = [simulate_buffon(n, b, s) for _ in range(num_experiments)]
    mean_p = np.mean(p_list)
    avg_probs.append(mean_p)

    # Confidence interval: P̂ ± 1.96 * sqrt(P̂(1 − P̂)/n)
    margin = 1.96 * np.sqrt(mean_p * (1 - mean_p) / n)
    ci_low.append(mean_p - margin)
    ci_high.append(mean_p + margin)

# D. 

outside_ratios = []

for n in n_values:
    count_outside = 0
    for _ in range(num_experiments):
        p = simulate_buffon(n, b, s)
        margin = 1.96 * np.sqrt(p * (1 - p) / n)
        low = p - margin
        high = p + margin
        if true_P < low or true_P > high:
            count_outside += 1
    outside_ratios.append(count_outside / num_experiments)


print (P_estimate, pi_estimate)

# Figure 1
plt.figure(1)
plt.plot(n_values, std_devs, marker='o', label='Std Dev')
plt.title("Standard Deviation of Estimated Probabilities")
plt.grid(True)
plt.legend()


# Figure 2
plt.figure(2)
plt.plot(n_small, avg_probs, label='P̂')
plt.fill_between(n_small, ci_low, ci_high, alpha=0.4, label='95% CI')
plt.title("Estimated Probability and 95% CI (n ≤ 100)")
plt.grid(True)
plt.legend()


# Figure 3
plt.figure(3)
plt.plot(n_values, outside_ratios, marker='s', label='Outside Ratio')
plt.title("Ratio of Experiments Where True P is Outside 95% CI")
plt.grid(True)
plt.legend()



plt.show()



