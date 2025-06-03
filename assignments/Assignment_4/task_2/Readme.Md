# Assignment 4 – Task 2_1 & Task 2_2

This repository simulates collective robot behaviors using delay differential equations. Two Python scripts are provided:

---

## Files

### `system_course.py`

* Use the rate equations model of searching and avoiding.

* Plots the number of searching robots, *n\_s(t)*, and the number of pucks, *m(t)*, for *t* in (0, 50].
* Calculates the temporal course of this system
* **Usage:**

  ```bash
  python3 system_course.py
  ```

### `Extended_model.py`

* Simulates the extended “search + avoid + homing” model with delays:

  * Avoid delay: 2 seconds
  * Homing delay: 15 seconds
* **First run:** No reset of *m*, simulate for *t* in (0, 160].
* **Second run:** At *t* = 80, reset *m(t)* to 0.5, then continue to *t* = 160.
* Produces two plots:

  * `task2_first_run.png`  – $n\_s$, $n\_h$, *m* without reset
  * `task2_second_run.png` – $n\_s$, $n\_h$, *m* with reset at *t* = 80
* **Usage:**

  ```bash
  python3 Extended_model.py
  ```

---

## Task 1 – “Search + Avoid” Model

### Equations

Block‐format versions in LaTeX:

$$
\begin{aligned}
\frac{d\,n_s}{dt} &= -\,0.6\,n_s(t)\,\bigl[n_s(t) + 1\bigr]
  \;+\; 0.6\,n_s\bigl(t - 2\bigr)\,\bigl[n_s\bigl(t - 2\bigr) + 1\bigr], \\
\frac{d\,m}{dt}   &= -\,0.2\,n_s(t)\,m(t).
\end{aligned}
$$

* $n\_s(t)$ = number of searching robots at time $t$
* $m(t)$    = number of pucks at time $t$

### Initial Conditions

* $n\_s(0) = 1$
* $m(0)   = 1$
* For all $t \le 0$, assume $n\_s(t) = 1$.

Hence, for $0 < t < 2$, we set $n\_s(t - 2) = 1$.

### Behavior

* When $n\_s(t) = 1$, the “avoid” terms exactly cancel:

  $$
  -0.6 \cdot 1 \cdot (1 + 1) \;+\; 0.6 \cdot 1 \cdot (1 + 1) \;=\; 0,
  $$

  so $\frac{d,n\_s}{dt} = 0$. Therefore, $n\_s(t)$ remains at 1.0 for $0 < t \le 50$.

* The puck equation reduces to

  $$
  \frac{d\,m}{dt} = -0.2\,m,
  $$

  whose solution is $m(t) = e^{-0.2,t}$. By $t = 50$, $m(50) \approx e^{-10} \approx 0$.

### Output

* **task_2_A_plot.png**

  * Left:  $n\_s(t)$ vs $t$ (flat at 1)
  * Right: $m(t)$ vs $t$ (exponential decay)

---

## Task 2_B – “Search + Avoid + Homing” Model

### Equations

Block‐format versions in LaTeX:

$$
\begin{aligned}
\frac{d\,n_s}{dt} &=
  -\,0.6\,n_s(t)\,\bigl[n_s(t) + 1\bigr]
  \;+\; 0.6\,n_s\bigl(t - 2\bigr)\,\bigl[n_s\bigl(t - 2\bigr) + 1\bigr]
  \;-\; 0.2\,n_s(t)\,m(t)
  \;+\; 0.2\,n_s\bigl(t - 15\bigr)\,m\bigl(t - 15\bigr), \\
\frac{d\,n_h}{dt} &=
  +\,0.2\,n_s(t)\,m(t)
  \;-\; 0.2\,n_s\bigl(t - 15\bigr)\,m\bigl(t - 15\bigr), \\
\frac{d\,m}{dt}   &= -\,0.2\,n_s(t)\,m(t).
\end{aligned}
$$

* $n\_s(t)$ = number of searching robots
* $n\_h(t)$ = number of homing robots
* $m(t)$    = number of pucks

### Initial Conditions

* $n\_s(0) = 1$
* $n\_h(0) = 0$
* $m(0)   = 1$
* For all $t \le 0$: $n\_s(t) = 1$, $n\_h(t)=0$, $m(t)=1$.

### Delay Implementation

* Avoid delay (2 s) → $2 / 0.01 = 200$ steps
* Homing delay (15 s) → $15 / 0.01 = 1500$ steps

For $t < 2$, set $n\_s(t - 2) = 1$.
For $t < 15$, homing return term = 0.

---

### First Run (No Reset)

* Simulate $t$ from 0 to 160 in increments of 0.01.
* Each pickup event moves robots from $n\_s \to n\_h$ and removes a puck from $m$.
* After 15 s, homing robots return from $n\_h \to n\_s$.
* $m(t)$ decays from 1 to 0 by $t \approx 50$.
* $n\_h(t)$ produces a large pulse from $t=0$ to $t \approx 15$, then smaller pulses until $t \approx 50$, after which $n\_h(t)=0$.
* $n\_s(t)$ dips during pickups and rebounds when homing robots return. After $t \approx 50$, $n\_s(t)=1$ again.

**Output:**

* `task2_B_first_run.png`

  * Top: $n\_s(t)$ (blue) and $n\_h(t)$ (red) vs $t$
  * Bottom: $m(t)$ vs $t$

---

### Second Run (Reset at $t = 80$)

* At $t = 80$, set $m(80) = 0.5$ (second batch of pucks).
* Continue simulating until $t = 160$.
* A second homing pulse appears from $t = 80$ to $t \approx 95$.
* $m(t)$ jumps from 0 to 0.5 at $t=80$, then decays to 0 by $t \approx 120$.

**Output:**

* `task2_B_second_run.png`

  * Top: $n\_s(t)$ (blue), $n\_h(t)$ (red), vertical line at $t=80$
  * Bottom: $m(t)$ (orange) with a red marker at $(80,,0.5)$

---

## Interpretation

* **Task 1:**

  * The avoid‐out and avoid‐in terms exactly cancel when $n\_s = 1$, so $n\_s(t)$ remains at 1.
  * The puck equation becomes $\frac{dm}{dt} = -0.2,m$, so $m(t) = e^{-0.2,t}$.

* **Task 2:**

  * Homing introduces a 15-s delay. Each pickup moves a fraction of $n\_s \to n\_h$ and removes pucks. After 15 s, homing robots rejoin $n\_s$.
  * Resetting $m$ at $t = 80$ injects a second homing wave.

---

## How to Run

1. Install dependencies:

   ```
   pip install numpy matplotlib
   ```

2. **Task 1**:

   ```
   python task1.py
   ```

   Generates and displays `task1_plot.png`.

3. **Task 2**:

   ```
   python task2.py
   ```

   Generates and displays `task2_B_first_run.png` and `task2_B_second_run.png`.

---
