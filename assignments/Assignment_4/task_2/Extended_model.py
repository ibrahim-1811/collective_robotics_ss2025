import numpy as np
import matplotlib.pyplot as plt

# Task 2 parameters
alpha_r, alpha_p = 0.6, 0.2
tau_a, tau_h = 2.0, 15.0
ns0, nh0, m0 = 1.0, 0.0, 1.0
dt, t_max = 0.01, 160.0
N = int(t_max / dt) + 1   # 160/0.01 + 1 = 16001 steps

# Time vector
t = np.linspace(0, t_max, N)

# Delay lengths in steps
delay_a = int(tau_a / dt)   # 200
delay_h = int(tau_h / dt)   # 1500

# ------ First run: no reset ------
ns = np.zeros(N)
nh = np.zeros(N)
m  = np.zeros(N)
ns[0], nh[0], m[0] = ns0, nh0, m0

for i in range(1, N):
    ns_cur, nh_cur, m_cur = ns[i - 1], nh[i - 1], m[i - 1]
    # Avoidance delay
    ns_delay_avoid = ns0 if i < delay_a else ns[i - delay_a]
    # Homing return delay
    if i < delay_h:
        homing_delay_rate = 0.0
    else:
        homing_delay_rate = alpha_p * ns[i - delay_h] * m[i - delay_h]
    # Pickup rate
    pickup_rate = alpha_p * ns_cur * m_cur
    # Derivatives
    avoid_out = alpha_r * ns_cur * (ns_cur + 1)
    avoid_in  = alpha_r * ns_delay_avoid * (ns_delay_avoid + 1)
    dns_dt = -avoid_out + avoid_in - pickup_rate + homing_delay_rate
    dnh_dt = pickup_rate - homing_delay_rate
    dm_dt  = -pickup_rate
    # Euler update
    ns[i] = ns_cur + dns_dt * dt
    nh[i] = nh_cur + dnh_dt * dt
    m[i]  = m_cur + dm_dt * dt

# Plot first run
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, ns, label='n_s(t)')
plt.plot(t, nh, label='n_h(t)')
plt.title("First Run (No Reset): n_s and n_h over (0, 160]")
plt.xlabel("t")
plt.ylabel("Population")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, m, color='orange', label='m(t)')
plt.title("First Run (No Reset): m(t) over (0, 160]")
plt.xlabel("t")
plt.ylabel("m")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("task2_first_run.png")  # also save figure
plt.show()

# ------ Second run: reset at t = 80 ------
ns2 = np.zeros(N)
nh2 = np.zeros(N)
m2  = np.zeros(N)
ns2[0], nh2[0], m2[0] = ns0, nh0, m0

for i in range(1, N):
    # At t ≈ 80, reset m to 0.5
    if abs(t[i] - 80.0) < dt / 2:
        m2[i - 1] = 0.5

    ns_cur2, nh_cur2, m_cur2 = ns2[i - 1], nh2[i - 1], m2[i - 1]
    ns_delay_avoid2 = ns0 if i < delay_a else ns2[i - delay_a]
    if i < delay_h:
        homing_delay_rate2 = 0.0
    else:
        homing_delay_rate2 = alpha_p * ns2[i - delay_h] * m2[i - delay_h]
    pickup_rate2 = alpha_p * ns_cur2 * m_cur2
    avoid_out2 = alpha_r * ns_cur2 * (ns_cur2 + 1)
    avoid_in2  = alpha_r * ns_delay_avoid2 * (ns_delay_avoid2 + 1)
    dns2_dt = -avoid_out2 + avoid_in2 - pickup_rate2 + homing_delay_rate2
    dnh2_dt = pickup_rate2 - homing_delay_rate2
    dm2_dt  = -pickup_rate2
    ns2[i] = ns_cur2 + dns2_dt * dt
    nh2[i] = nh_cur2 + dnh2_dt * dt
    m2[i]  = m_cur2 + dm2_dt * dt

# Plot second run
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, ns2, label='n_s(t)')
plt.plot(t, nh2, label='n_h(t)')
plt.axvline(80, color='gray', linestyle='--', label='Reset at t=80')
plt.title("Second Run (Reset at t=80): n_s and n_h")
plt.xlabel("t")
plt.ylabel("Population")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, m2, color='orange', label='m(t)')
plt.scatter([80], [0.5], color='red', label='m reset to 0.5')
plt.title("Second Run (Reset at t=80): m(t)")
plt.xlabel("t")
plt.ylabel("m")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("task2_second_run.png")
plt.show()
