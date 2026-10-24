import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(c, max_iter):
    z = 0
    n = np.zeros(c.shape, dtype=int)
    mask = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        # Apply the Mandelbrot formula: z = z^2 + c
        z = z**2 + c
        # Find points that escape to infinity
        escaped = np.abs(z) > 2
        # Record iteration number for escaped points
        n[escaped & mask] = i
        mask[escaped] = False

    return n


# Define the grid dimensions and resolution
xmin, xmax, xnum = -2.0, 0.5, 1000
ymin, ymax, ynum = -1.25, 1.25, 1000

# Create a coordinate grid of complex numbers
r1 = np.linspace(xmin, xmax, xnum)
r2 = np.linspace(ymin, ymax, ynum)
x, y = np.meshgrid(r1, r2)
c = x + 1j * y

# Generate the image matrix
fractal_image = mandelbrot(c, max_iter=100)

# Display the fractal using a color map
plt.figure(figsize=(10, 10))
plt.imshow(fractal_image, extent=(
    xmin, xmax, ymin, ymax), cmap='twilight_shifted')
plt.axis('off')
plt.show()
