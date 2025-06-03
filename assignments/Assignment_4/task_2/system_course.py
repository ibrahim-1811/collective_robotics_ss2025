import numpy as np
import matplotlib.pyplot as plt

# Task 1 parameters
alpha_r, alpha_p, tau_a = 0.6, 0.2, 2.0
ns0, m0 = 1.0, 1.0
dt, t_max = 0.01, 50.0
N = int(t_max / dt) + 1   # 5001 steps

# Time vector
t = np.linspace(0, t_max, N)

# Preallocate arrays for n_s and m
ns = np.full(N, ns0)
m  = np.full(N, m0)

# Delay in steps
delay = int(tau_a / dt)   # 2.0 / 0.01 = 200

# Forward Euler integration with a 2 s delay
for i in range(1, N):
    prev_ns = ns[i - 1]
    # If i < 200, use history ns = 1; else read ns from 2 s ago
    delayed_ns = ns0 if i < delay else ns[i - delay]
    # Compute derivatives
    dns = -alpha_r * prev_ns * (prev_ns + 1) \
          + alpha_r * delayed_ns * (delayed_ns + 1)
    dm  = -alpha_p * prev_ns * m[i - 1]
    # Euler update
    ns[i] = prev_ns + dns * dt
    m[i]  = m[i - 1] + dm * dt

# Plot results side by side
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t, ns, color='blue')
plt.title("n_s(t) over (0, 50]")
plt.xlabel("t")
plt.ylabel("n_s")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, m, color='orange')
plt.title("m(t) over (0, 50]")
plt.xlabel("t")
plt.ylabel("m")
plt.grid(True)

plt.tight_layout()
plt.show()

# Interpretation printed to console
print("Interpretation:")
print("  n_s(t) remains at 1.0 because, for t<2, n_s(t-2)=1, so the avoid-out and avoid-in terms cancel.")
print("  m(t) decays roughly as exp(-0.2 t) from 1 to near 0 by t=50.")
