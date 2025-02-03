# import numpy as np
# import time

# def rk4(f, y0, t0, t_end, dt):
#     t_values = np.arange(t0, t_end, dt)
#     y_values = np.zeros((len(t_values), len(y0)))
#     y_values[0] = y0

#     for i in range(1, len(t_values)):
#         t = t_values[i - 1]
#         y = y_values[i - 1]

#         k1 = dt * f(t, y)
#         k2 = dt * f(t + dt / 2, y + k1 / 2)
#         k3 = dt * f(t + dt / 2, y + k2 / 2)
#         k4 = dt * f(t + dt, y + k3)

#         y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

#     return t_values, y_values

# # equation: dy/dt = -2y + sin(t))
# def f(t, y):
#     return np.array([-2 * y[0] + np.sin(t)])
# y0 = np.array([1.0])
# t0 = 0.0
# t_end = 10.0
# dt = 0.1
# start_time = time.time()
# t_values, y_values = rk4(f, y0, t0, t_end, dt)
# runtime = time.time() - start_time

# print(f"Runtime: {runtime:.6f} seconds")
# print("\nFirst few time points and corresponding y values:")
# for i in range(min(5, len(t_values))):
#     print(f"t = {t_values[i]:.2f}, y = {y_values[i][0]:.6f}")

# print("\nLast few time points and corresponding y values:")
# for i in range(max(0, len(t_values)-5), len(t_values)):
#     print(f"t = {t_values[i]:.2f}, y = {y_values[i][0]:.6f}")

import numpy as np
import time

def adaptive_rk4(f, y0, t0, t_end, dt_initial, tol):
    t = t0
    y = np.array(y0, dtype=float) 
    dt = dt_initial
    t_values = [t]
    y_values = [y.copy()]

    steps = 0
    max_steps = 10000

    print("Starting integration...")

    while t < t_end and steps < max_steps:
        print(f"Current t: {t}, dt: {dt}")

        k1 = dt * f(t, y)
        k2 = dt * f(t + dt/2, y + k1/2)
        k3 = dt * f(t + dt/2, y + k2/2)
        k4 = dt * f(t + dt, y + k3)

        y_new = y + (k1 + 2*k2 + 2*k3 + k4)/6
        error_estimate = np.linalg.norm(y_new - y)

        print(f"Error estimate: {error_estimate}")

        if error_estimate < tol:
            t += dt
            y = y_new.copy()
            t_values.append(t)
            y_values.append(y.copy())
            print(f"Step accepted. New t: {t}")

        dt *= 0.9 * (tol/error_estimate)**0.2
        dt = max(dt, dt_initial/10)

        steps += 1

    if steps >= max_steps:
        print("Warning: Maximum steps reached!")

    return np.array(t_values), np.array(y_values)

def f(t, y):
    return np.array([-2 * y[0] + np.sin(t)])

y0 = np.array([1.0])
t0 = 0.0
t_end = 10.0
dt_initial = 0.1
tol = 1e-6

print("Starting simulation...")
start_time = time.time()
t_values, y_values = adaptive_rk4(f, y0, t0, t_end, dt_initial, tol)
runtime = time.time() - start_time

print(f"\nSimulation completed!")
print(f"Runtime: {runtime:.6f} seconds")
print(f"Number of points: {len(t_values)}")

if len(t_values) > 0:
    print("\nFirst few values:")
    for i in range(min(5, len(t_values))):
        print(f"t = {t_values[i]:.3f}, y = {y_values[i][0]:.6f}")
