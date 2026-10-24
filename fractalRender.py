import fractals
import matplotlib.pyplot as plt
import numpy as np


def render_fractal(fractal_type, max_itera):

    if fractal_type == "Barnsley Fern":
        xmin, xmax, ymin, ymax = -2.2, 2.7, 0.0, 10.0
    elif fractal_type == "Sierpinski Triangle":
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, np.sqrt(3) / 2
    else:
        xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0


    xnum, ynum = 1000, 1000

    r1 = np.linspace(xmin, xmax, xnum)
    r2 = np.linspace(ymin, ymax, ynum)
    x, y = np.meshgrid(r1, r2)
    c = x + 1j * y

    plt.figure(figsize=(10, 10))

    if fractal_type == "Burning ship":
        fractal_image = fractals.burningShip(c, max_itera)
    elif fractal_type == "Mandel Brot":
        fractal_image = fractals.mandelBrot(c, max_itera)
    elif fractal_type == "Newton's Fractal":
        fractal_image = fractals.newtonsFractal(c, max_itera)
    elif fractal_type == "Nova Fractal":
        fractal_image = fractals.novaFractal(c, max_itera)
    elif fractal_type == "Barnsley Fern":
        fractal_image = fractals.barnsleyFern(c, max_itera)
    elif fractal_type == "Sierpinski Triangle":
        fractal_image = fractals.sierpinskiTriangle(c, max_itera)
    else:
        raise ValueError(f"Unknown fractal type: {fractal_type}")

    plt.imshow(fractal_image, extent=(xmin, xmax, ymin, ymax),
                cmap="twilight_shifted")

    plt.axis("off")
    plt.title(fractal_type, fontsize=16)
    plt.show()
