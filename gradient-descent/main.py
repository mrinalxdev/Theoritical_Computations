import numpy as np
import matplotlib.pyplot as plt

# Objective function: f(x1, x2) = (x1 - 3)^2 + (x2 + 2)^2
def objective_function(x):
    return (x[0] - 3)**2 + (x[1] + 2)**2

def gradient(x):
    df_dx1 = 2 * (x[0] - 3)
    df_dx2 = 2 * (x[1] + 2)
    return np.array([df_dx1, df_dx2])

def gradient_descent_with_path(starting_point, learning_rate, num_iterations):
    x = np.array(starting_point, dtype=float)
    points = [x.copy()]  # storing the path

    for i in range(num_iterations):
        grad = gradient(x)
        x = x - learning_rate * grad
        points.append(x.copy())

        # optional will do it after dinner
        if i % 10 == 0:
            print(f"Iteration {i}: x = {x}, f(x) = {objective_function(x)}")

    return x, points

def plot_optimization_path(objective_function, points):

    x_vals = np.linspace(-5, 5, 400)
    y_vals = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)


    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_function([X[i, j], Y[i, j]])

    # plotting the contour
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.7)


    points = np.array(points)
    for i in range(len(points) - 1):
        plt.arrow(
            points[i, 0], points[i, 1],
            points[i+1, 0] - points[i, 0], points[i+1, 1] - points[i, 1],
            head_width=0.1, head_length=0.1, fc='blue', ec='blue'
        )

    plt.scatter(points[0, 0], points[0, 1], color='red', s=100, label="Starting Point")
    plt.text(points[0, 0], points[0, 1], " Start", fontsize=12, color='red')

    plt.scatter(points[-1, 0], points[-1, 1], color='green', s=100, label="Optimal Point")
    plt.text(points[-1, 0], points[-1, 1], " Optimal", fontsize=12, color='green')
    plt.xlabel("x1", fontsize=14)
    plt.ylabel("x2", fontsize=14)
    plt.title("Gradient Descent Optimization Path", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.colorbar(label="Objective Function Value")
    plt.show()

starting_point = [0.0, 0.0]
learning_rate = 0.1
num_iterations = 50

optimal_x, points = gradient_descent_with_path(starting_point, learning_rate, num_iterations)


print(f"Optimal x: {optimal_x}")
print(f"Minimum value of f(x): {objective_function(optimal_x)}")
plot_optimization_path(objective_function, points)
