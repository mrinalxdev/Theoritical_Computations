import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
        self.x_mean = 0.0
        self.y_mean = 0.0

    def fit(self, x, y):
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi*yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi**2 for xi in x)

        self.x_mean = sum_x / n
        self.y_mean = sum_y / n

        denominator = n * sum_x2 - sum_x**2
        if denominator == 0:
            raise ValueError("Cannot compute coefficients - denominator is zero")

        self.slope = (n * sum_xy - sum_x * sum_y) / denominator
        self.intercept = (sum_y - self.slope * sum_x) / n

    def predict(self, x):
        return [self.slope * xi + self.intercept for xi in x]

    def calculate_errors(self, y_true, y_pred):
        n = len(y_true)
        sse = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred))
        mse = sse / n
        rmse = mse ** 0.5
        mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

        ss_total = sum((yt - self.y_mean)**2 for yt in y_true)
        r_squared = 1 - (sse / ss_total) if ss_total != 0 else 0

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r_squared
        }

    def plot_regression(self, x, y):
        plt.scatter(x, y, color='blue', label='Actual Data')
        plt.plot(x, self.predict(x), color='red', label='Regression Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression Fit')
        plt.legend()
        plt.grid(True)
        plt.show()

x_data = [1, 2, 3, 4, 5, 6, 7]
y_data = [1, 3, 2, 5, 4, 6, 5]

model = LinearRegression()
model.fit(x_data, y_data)

predictions = model.predict(x_data)

errors = model.calculate_errors(y_data, predictions)

print("=== Linear Regression Results ===")
print(f"Slope (m): {model.slope:.4f}")
print(f"Intercept (b): {model.intercept:.4f}")
print("\nRegression Equation:")
print(f"y = {model.slope:.2f}x + {model.intercept:.2f}")

print("\n=== Error Metrics ===")
for metric, value in errors.items():
    print(f"{metric}: {value:.4f}")

model.plot_regression(x_data, y_data)

new_x = 8
prediction = model.slope * new_x + model.intercept
print(f"\nPrediction for x={new_x}: {prediction:.2f}")

# todo : write some edge cases after dinner
test_x = [10, 20, 30, 40, 50]
test_y = [15, 25, 35, 45, 55]
model.fit(test_x, test_y)
print("\n=== Perfect Fit Test Case ===")
print(f"Slope: {model.slope:.2f} (Expected: 1.00)")
print(f"R²: {model.calculate_errors(test_y, model.predict(test_x))['R²']:.2f} (Expected: 1.00)")
