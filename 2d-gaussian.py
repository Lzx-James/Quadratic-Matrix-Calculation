# Importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm
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
               [1.0, 2.0]])
mu = np.array([1, 2])
random_seed = 10

generate_and_plot(Kx, mu)

a = np.sqrt(1)  # Principle axes
b = np.sqrt(3)
theta = np.radians(-45)

# 生成椭圆的点
t = np.linspace(0, 2 * np.pi, 100)
x = a * np.cos(t)
y = b * np.sin(t)

# 旋转椭圆
x_rotated = x * np.cos(theta) - y * np.sin(theta) + mu[0]
y_rotated = x * np.sin(theta) + y * np.cos(theta) + mu[1]

# 绘制
plt.plot(x_rotated, y_rotated, label='斜椭圆', color='red')

plt.show()
