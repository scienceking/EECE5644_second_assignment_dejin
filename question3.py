# This code is for Question 3
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Number of landmarks and measurement noise standard deviation
K_values = [1, 2, 3, 4]
sigma_r = 0.3  # Measurement noise standard deviation
sigma_x = sigma_y = 0.25  # Prior distribution standard deviation


# Function to generate landmark positions
def generate_landmarks(K):
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))


# Generate noisy distance measurements
def generate_measurements(true_pos, landmarks, sigma_r):
    distances = np.linalg.norm(landmarks - true_pos, axis=1)
    measurements = distances + np.random.normal(0, sigma_r, size=distances.shape)
    return np.clip(measurements, 0, None)  # Avoid negative measurement values


# Objective function calculation
def objective_function(x, y, measurements, landmarks, sigma_r, sigma_x, sigma_y):
    pred_distances = np.linalg.norm(landmarks - np.array([x, y]), axis=1)
    error_term = np.sum((measurements - pred_distances) ** 2) / (2 * sigma_r ** 2)
    prior_term = (x ** 2) / (2 * sigma_x ** 2) + (y ** 2) / (2 * sigma_y ** 2)
    return error_term + prior_term


# Function to plot contour plot
def plot_contour(K, true_pos):
    landmarks = generate_landmarks(K)
    measurements = generate_measurements(true_pos, landmarks, sigma_r)

    # Grid range [-2, 2], generate grid points
    x_vals = np.linspace(-2, 2, 100)
    y_vals = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Calculate objective function value for each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_function(X[i, j], Y[i, j], measurements, landmarks, sigma_r, sigma_x, sigma_y)

    # Plot contour
    plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], color='red', label='Landmarks (o)')
    plt.scatter(*true_pos, color='blue', marker='+', s=100, label='True Position (+)')
    plt.title(f'Contour plot for K = {K}')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Set true vehicle position
true_pos = np.array([0.5, 0.5])

# For each K value, plot contour
for K in K_values:
    plot_contour(K, true_pos)
