import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (Log Loss)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (-1/m) * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
    return cost

# Gradient Descent
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * X.T @ (h - y)
        theta -= learning_rate * gradient  # updating the theta
        cost = compute_cost(X, y, theta)   # computing the cost
        cost_history.append(cost)

        if i % 100 == 0: 
            print(f"Iteration {i}: Cost = {cost}")

    return theta, cost_history

def predict(X, theta, threshold=0.5):
    probabilities = sigmoid(X @ theta)
    return (probabilities >= threshold).astype(int)

def generate_large_dataset_3d(num_samples=30000):
    X1 = np.random.normal(loc=2, scale=1, size=num_samples)  # Feature 1
    X2 = np.random.normal(loc=3, scale=1.5, size=num_samples)  # Feature 2
    X3 = np.random.normal(loc=4, scale=2, size=num_samples)  # Feature 3

    # If X1 + X2 + X3 > 7.5, label = 1; otherwise, label = 0
    y = (X1 + X2 + X3 > 7.5).astype(int)

    X = np.column_stack((np.ones(num_samples), X1, X2, X3))

    return X, y

def plot_3d_decision_boundary(X, y, theta):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[y == 0][:, 1], X[y == 0][:, 2], X[y == 0][:, 3], color='red', label='Class 0', alpha=0.5)
    ax.scatter(X[y == 1][:, 1], X[y == 1][:, 2], X[y == 1][:, 3], color='blue', label='Class 1', alpha=0.5)

    # a meshgrid for the decision boundary
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

    # Calculate the corresponding z values for the decision boundary
    zz = -(theta[0] + theta[1] * xx + theta[2] * yy) / theta[3]

    # Plot the decision boundary
    ax.plot_surface(xx, yy, zz, color='green', alpha=0.5, label='Decision Boundary')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Decision Boundary')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    X, y = generate_large_dataset_3d(num_samples=30000)

    theta = np.zeros(X.shape[1])  # Initial weights (including bias)

    learning_rate = 0.01
    num_iterations = 1000

    theta, cost_history = gradient_descent(X, y, theta, learning_rate, num_iterations)

    predictions = predict(X, theta)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("Final weights:", theta)

    plot_3d_decision_boundary(X[:1000], y[:1000], theta)

    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Over Iterations')
    plt.show()
