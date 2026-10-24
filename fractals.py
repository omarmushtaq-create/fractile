import numpy as np

def mandelBrot(c, max_iter):
    z = 0
    n = np.zeros(c.shape, dtype=int)
    mask = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        z = z**2 + c
        escaped = np.abs(z) > 2
        n[escaped & mask] = i
        mask[escaped] = False

    return n


def burningShip(c, max_iter):
    z = 0
    n = np.zeros(c.shape, dtype=int)
    mask = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        z = (np.abs(z.real) + 1j * np.abs(z.imag))**2 + c
        escaped = np.abs(z) > 2
        n[escaped & mask] = i
        mask[escaped] = False

    return n


def newtonsFractal(c, max_iter, tol=1e-6):
    z = np.asarray(c, dtype=np.complex128).copy()
    n = np.zeros(c.shape, dtype=int)
    mask = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        if not np.any(mask):
            break

        z_current = z[mask]
        z_next = z_current - (z_current**3 - 1) / (3 * z_current**2)
        z[mask] = z_next

        converged = np.abs(z_next - z_current) < tol

        global_converged = np.zeros(c.shape, dtype=bool)
        global_converged[mask] = converged

        n[global_converged & mask] = i
        mask[global_converged] = False

    return n


def barnsleyFern(c, max_iter):
    """Generate a rasterized Barnsley Fern using an iterated function system.

    ``max_iter`` controls the number of points sampled in thousands.  Returning
    an image with the same shape as ``c`` keeps this function compatible with
    the other fractal renderers.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    height, width = c.shape
    sample_count = int(max_iter) * 1000
    rng = np.random.default_rng()
    choices = rng.random(sample_count)

    x = 0.0
    y = 0.0
    points_x = np.empty(sample_count)
    points_y = np.empty(sample_count)

    for i, choice in enumerate(choices):
        if choice < 0.01:
            x, y = 0.0, 0.16 * y
        elif choice < 0.86:
            x, y = 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6
        elif choice < 0.93:
            x, y = 0.2 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6
        else:
            x, y = -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44
        points_x[i] = x
        points_y[i] = y

    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, np.sqrt(3) / 2
    columns = np.clip(((points_x - xmin) / (xmax - xmin) * (width - 1)).astype(int), 0, width - 1)
    rows = np.clip(((points_y - ymin) / (ymax - ymin) * (height - 1)).astype(int), 0, height - 1)

    image = np.zeros((height, width), dtype=float)
    np.add.at(image, (height - 1 - rows, columns), 1)
    return np.log1p(image)


def novaFractal(c, max_iter, R=1.0, tol=1e-6):
    c = np.asarray(c, dtype=np.complex128)
    z = np.ones_like(c)                     # Nova starts at z0 = 1
    # points that never converge get max_iter
    n = np.full(c.shape, max_iter, dtype=int)
    mask = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        if not np.any(mask):
            break

        z_current = z[mask]
        c_current = c[mask]

        with np.errstate(divide="ignore", invalid="ignore"):
            z_next = z_current - R * \
                (z_current**3 - 1) / (3 * z_current**2) + c_current

        z[mask] = z_next

        # boolean, as it should be
        converged = np.abs(z_next - z_current) < tol

        global_converged = np.zeros(c.shape, dtype=bool)
        global_converged[mask] = converged

        n[global_converged] = i
        mask[global_converged] = False

    return n



def sierpinskiTriangle(c, max_iter):
    """Generate a rasterized Sierpinski Triangle using an iterated function system.

    ``max_iter`` controls the number of points sampled in thousands.  Returning
    an image with the same shape as ``c`` keeps this function compatible with
    the other fractal renderers.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    height, width = c.shape
    sample_count = int(max_iter) * 1000
    rng = np.random.default_rng()
    choices = rng.integers(0, 3, size=sample_count)

    x = 0.0
    y = 0.0
    points_x = np.empty(sample_count)
    points_y = np.empty(sample_count)

    for i, choice in enumerate(choices):
        if choice == 0:
            x, y = 0.5 * x, 0.5 * y
        elif choice == 1:
            x, y = 0.5 * x + 0.5, 0.5 * y
        else:
            x, y = 0.5 * x + 0.25, 0.5 * y + np.sqrt(3) / 4
        points_x[i] = x
        points_y[i] = y

    xmin, xmax = points_x.min(), points_x.max()
    ymin, ymax = points_y.min(), points_y.max()
    columns = np.clip(((points_x - xmin) / (xmax - xmin) * (width - 1)).astype(int), 0, width - 1)
    rows = np.clip(((points_y - ymin) / (ymax - ymin) * (height - 1)).astype(int), 0, height - 1)

    image = np.zeros((height, width), dtype=float)
    np.add.at(image, (height - 1 - rows, columns), 1)
    return np.log1p(image)
