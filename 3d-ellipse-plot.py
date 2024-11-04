import numpy as np
import matplotlib.pyplot as plt

# Ellipse sphere 1
A1 = np.array([[2, 1, -1],
               [1, 4, 1],
               [-1, 1, 4]
               ])

# Ellipse sphere 2
A2 = np.array([[5, 0, -np.sqrt(2)],
               [0, 2, 0],
               [-np.sqrt(2), 0, 4],
               ])

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Generate sphere points
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

# Cholesky transmission for quadratic matrix
L = np.linalg.cholesky(A1)

# Generating points of an ellipse by a quadratic matrix transformation
ellipse_points = np.zeros((3, x_sphere.size))
for i in range(x_sphere.shape[0]):
    for j in range(x_sphere.shape[1]):
        point = np.array([x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]])
        ellipse_points[:, i * x_sphere.shape[1] + j] = L.dot(point)


# Plot ellipse sphere
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(ellipse_points[0, :].reshape(x_sphere.shape),
                ellipse_points[1, :].reshape(y_sphere.shape),
                ellipse_points[2, :].reshape(z_sphere.shape),
                color='cyan', alpha=0.6)

plt.show()
