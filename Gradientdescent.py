import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionEstimator:
    def __init__(self, N, true_slope, true_intercept, noise_std=1.3, seed=42):
        """Initialize dataset parameters"""
        self.N = N
        self.true_slope = true_slope
        self.true_intercept = true_intercept
        self.noise_std = noise_std
        self.seed = seed
        self.x, self.y = self._generate_data()

    def _generate_data(self):
        """Generate synthetic linear data with noise"""
        np.random.seed(self.seed)
        x = np.linspace(-5, 5, self.N)
        noise = np.random.normal(0, self.noise_std, self.N)
        y = self.true_slope * x + self.true_intercept + noise
        return x, y

    def _compute_loss(self, slope):
        """Compute absolute loss for a given slope"""
        y_pred = slope * self.x + self.true_intercept
        return np.mean(np.abs(self.y - y_pred))

    def linear_search_best_m1(self, m1_range):
        """Brute force search for the best slope"""
        min_error = float('inf')
        best_slope = None
        errors = []

        print("\nRunning Brute Force Search...")

        for m1 in m1_range:
            error = self._compute_loss(m1)
            errors.append(error)
            if error < min_error:
                min_error = error
                best_slope = m1  # Save best slope
        
        print(f"[INFO] Brute Force Best Slope Found: {best_slope:.4f}")
        return best_slope, errors

    def gradient_descent(self, learning_rate=0.005, iterations=500):
        """Gradient Descent to estimate best slope"""
        np.random.seed(self.seed)
        m1 = np.random.uniform(3.5, 4.8)  # Slightly randomized starting point
        loss_history = []

        print("\nRunning Gradient Descent...")
        for i in range(iterations):
            y_pred = m1 * self.x + self.true_intercept
            gradient = -np.dot(self.x, (self.y - y_pred)) / len(self.y)
            m1 -= learning_rate * gradient
            loss = np.mean((self.y - y_pred) ** 2)
            loss_history.append(loss)

            if i % 100 == 0:  # Debugging print every 100 iterations
                print(f"Iteration {i}: Slope={m1:.4f}, Loss={loss:.5f}")

        print(f"[INFO] Gradient Descent Best Slope Found: {m1:.4f}")
        return m1, loss_history

    def plot_graphs(self, m1_range, errors, best_m1_brute, best_m1_gd, loss_history):
        """Generate visualization for the model"""

        # Scatter plot of generated data
        plt.figure(figsize=(6, 4))
        plt.scatter(self.x, self.y, color='blue', marker='o', alpha=0.6, label="Data Points")
        plt.plot(self.x, self.true_slope * self.x + self.true_intercept, linestyle='dashed', color='black', label="True Line")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Generated Data")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

        # Error vs. slope (Brute Force Search)
        plt.figure(figsize=(6, 4))
        plt.plot(m1_range, errors, color='red', linewidth=2)
        plt.axvline(best_m1_brute, color='black', linestyle='dotted', label="Best Slope (Brute Force)")
        plt.xlabel("Slope (m1)")
        plt.ylabel("Error")
        plt.title("Error vs. Slope")
        plt.legend()
        plt.grid(alpha=0.4)
        plt.show()

        # Loss reduction over iterations
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(loss_history)), loss_history, color='green', linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Gradient Descent Loss Over Iterations")
        plt.grid(alpha=0.4)
        plt.show()

        # Comparison of best fit lines
        plt.figure(figsize=(6, 4))
        plt.scatter(self.x, self.y, color='blue', alpha=0.5, label="Data")
        plt.plot(self.x, best_m1_brute * self.x + self.true_intercept, color='red', label="Brute Force Fit")
        plt.plot(self.x, best_m1_gd * self.x + self.true_intercept, color='green', linestyle="dashed", label="Gradient Descent Fit")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Comparison of Estimated Fits")
        plt.legend()
        plt.grid(alpha=0.4)
        plt.show()


# Main Execution
if __name__ == "__main__":
    print("\n[INFO] Initializing Linear Regression Model...")
    model = LinearRegressionEstimator(N=150, true_slope=4.2, true_intercept=2.1)

    print("\n[INFO] Starting Brute Force Search...")
    m1_range = np.linspace(2, 6, 200)
    best_m1_brute, errors = model.linear_search_best_m1(m1_range)

    print("\n[INFO] Starting Gradient Descent...")
    best_m1_gd, loss_history = model.gradient_descent()

    print("\n[INFO] Plotting Results...")
    model.plot_graphs(m1_range, errors, best_m1_brute, best_m1_gd, loss_history)

    # Final Results
    print("\n======== Final Results ========")
    print(f"Brute Force Best Slope: {best_m1_brute:.4f}")
    print(f"Gradient Descent Best Slope: {best_m1_gd:.4f}")
    print("================================")
