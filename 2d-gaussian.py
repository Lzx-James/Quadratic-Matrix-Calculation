# Importing the necessary modules
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
Kx = np.array([[2.0, 1.0],
               [1.0, 4.0]])
mu = np.array([0, 0])
random_seed = 10

generate_and_plot(Kx, mu)
plt.show()
