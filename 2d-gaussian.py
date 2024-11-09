import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

plt.rcParams['figure.figsize'] = 8, 8


def generate_and_plot(kx, mu):
    distr = multivariate_normal(cov=kx, mean=mu, seed=1000)
    data = distr.rvs(size=5000)
    plt.grid()
    plt.plot(data[:, 0], data[:, 1], 'o', c='lime', markeredgewidth=0.5, markeredgecolor='black')

    plt.title(r'Random samples from a 2D-Gaussian distribution')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.axis('equal')


# Define the mean and covariance matrix
Kx1 = np.array([[2.0, 0.0],
               [0.0, 4.0]])
mu1 = np.array([0, 0])

Kx2 = np.array([[2.0, 1.0],
               [1.0, 2.0]])
mu2 = np.array([0, 0])

Kx3 = np.array([[2.0, 1.0],
               [1.0, 2.0]])
mu3 = np.array([1, 2])

Kx = Kx3
mu = mu3

random_seed = 10

generate_and_plot(Kx, mu)

# Get inverse matrix of Kxx
Kx_inv = np.linalg.inv(Kx)

# Generate circle theta
theta = np.linspace(0, 2 * np.pi, 100)

# Generate circle by theta
circle_points = np.array([np.cos(theta), np.sin(theta)])

# Generating points of an ellipse by a quadratic matrix transformation
ellipse_points = np.linalg.cholesky(Kx_inv).dot(circle_points)

# Plot ellipse
plt.plot(ellipse_points[0, :] + mu[0], ellipse_points[1, :] + mu[1], color='red')

plt.show()
