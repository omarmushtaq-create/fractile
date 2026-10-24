import numpy as np


def _iterate(z, c, step, is_done, max_iter, fill=0):
    """Iterate ``z = step(z, c)`` but only on points that are still active.

    Points are removed from the working arrays as soon as ``is_done`` says
    they have escaped or converged, so the per-iteration cost shrinks as the
    image resolves instead of staying at the full grid size.  Returns a flat
    array holding the iteration each point finished on (``fill`` if it never
    did).
    """
    out = np.full(z.size, fill, dtype=np.int32)
    idx = np.arange(z.size)

    with np.errstate(all="ignore"):
        for i in range(1, max_iter + 1):
            if idx.size == 0:
                break
            z_new = step(z, c)
            done = is_done(z, z_new)
            if done.any():
                out[idx[done]] = i
                alive = ~done
                idx, z, c = idx[alive], z_new[alive], c[alive]
            else:
                z = z_new
    return out


def mandelBrot(c, max_iter):
    c = np.asarray(c, dtype=np.complex128)
    x, y = c.real, c.imag

    # Points inside the main cardioid or the period-2 bulb never escape, so
    # skip them entirely.  This removes a large share of the work when the
    # whole set is in view.
    q = (x - 0.25) ** 2 + y * y
    inside = (q * (q + (x - 0.25)) <= 0.25 * y *
              y) | ((x + 1) ** 2 + y * y <= 0.0625)

    flat_c = c.ravel()
    todo = ~inside.ravel()
    out = np.zeros(c.size, dtype=np.int32)
    out[todo] = _iterate(
        np.zeros(int(todo.sum()), dtype=np.complex128),
        flat_c[todo],
        step=lambda z, c: z * z + c,
        # |z| > 2 compared as |z|^2 > 4 to avoid a square root per point.
        is_done=lambda old, new: new.real ** 2 + new.imag ** 2 > 4.0,
        max_iter=max_iter,
    )
    return out.reshape(c.shape)


def burningShip(c, max_iter):
    c = np.asarray(c, dtype=np.complex128)

    def step(z, c):
        x = np.abs(z.real)
        y = np.abs(z.imag)
        return (x * x - y * y + c.real) + 1j * (2 * x * y + c.imag)

    out = _iterate(
        np.zeros(c.size, dtype=np.complex128),
        c.ravel(),
        step=step,
        is_done=lambda old, new: new.real ** 2 + new.imag ** 2 > 4.0,
        max_iter=max_iter,
    )
    return out.reshape(c.shape)


def newtonsFractal(c, max_iter, tol=1e-6):
    c = np.asarray(c, dtype=np.complex128)
    tol2 = tol * tol

    def is_done(old, new):
        d = new - old
        # Stop on convergence, and also drop points that hit 0 or blow up
        # (NaN/inf) so they do not stay active forever.
        return (d.real ** 2 + d.imag ** 2 < tol2) | ~np.isfinite(new)

    out = _iterate(
        c.ravel().copy(),
        c.ravel(),
        # Newton's update for f(z) = z^3 - 1, simplified: z - (z^3-1)/(3z^2)
        step=lambda z, c: (2 * z + 1 / (z * z)) / 3,
        is_done=is_done,
        max_iter=max_iter,
        fill=max_iter,
    )
    return out.reshape(c.shape)


def novaFractal(c, max_iter, R=1.0, tol=1e-6):
    c = np.asarray(c, dtype=np.complex128)
    tol2 = tol * tol

    def is_done(old, new):
        d = new - old
        converged = d.real ** 2 + d.imag ** 2 < tol2
        diverged = new.real ** 2 + new.imag ** 2 > 1e12
        return converged | diverged | ~np.isfinite(new)

    out = _iterate(
        np.ones(c.size, dtype=np.complex128),  # Nova starts at z0 = 1
        c.ravel(),
        # z - R * (z^3 - 1) / (3 z^2) + c, simplified
        step=lambda z, c: z - R * (z - 1 / (z * z)) / 3 + c,
        is_done=is_done,
        max_iter=max_iter,
        fill=max_iter,
    )
    return out.reshape(c.shape)


def _ifs_image(step, shape, bounds, max_iter, chains=2000, burn_in=20, max_steps=3000):
    """Rasterise an iterated function system with many parallel chains.

    ``step(x, y, rng)`` advances every chain by one random transform.  Only
    points that land inside ``bounds`` are kept, and sampling continues until
    ``max_iter * 1000`` of them have been collected (or ``max_steps`` is hit).
    That means zooming in keeps gathering points for the visible window
    instead of showing a handful of stray dots.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    height, width = shape
    xmin, xmax, ymin, ymax = bounds
    scale_x = width / (xmax - xmin)
    scale_y = height / (ymax - ymin)
    target = int(max_iter) * 1000
    rng = np.random.default_rng()

    x = np.zeros(chains)
    y = np.zeros(chains)
    chunks = []
    collected = 0

    for n in range(max_steps):
        x, y = step(x, y, rng)
        if n < burn_in:
            continue
        visible = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
        if not visible.any():
            continue
        cols = np.minimum(
            ((x[visible] - xmin) * scale_x).astype(np.intp), width - 1)
        rows = np.minimum(
            ((y[visible] - ymin) * scale_y).astype(np.intp), height - 1)
        chunks.append(rows * width + cols)
        collected += cols.size
        if collected >= target:
            break

    if chunks:
        counts = np.bincount(np.concatenate(chunks), minlength=height * width)
    else:
        counts = np.zeros(height * width)
    # Row 0 is the bottom of the image (the renderer uses origin="lower").
    # Log scaling reveals the faint detail.
    return np.log1p(counts.reshape(height, width).astype(float))


def barnsleyFern(c, max_iter, bounds=(-2.2, 2.7, 0.0, 10.0)):
    """Barnsley Fern. ``max_iter`` is the number of points sampled, in thousands."""

    def step(x, y, rng):
        r = rng.random(x.size)
        conds = [r < 0.01, r < 0.86, r < 0.93]
        new_x = np.select(
            conds,
            [np.zeros_like(x), 0.85 * x + 0.04 * y, 0.2 * x - 0.26 * y],
            default=-0.15 * x + 0.28 * y,
        )
        new_y = np.select(
            conds,
            [0.16 * y, -0.04 * x + 0.85 * y + 1.6, 0.23 * x + 0.22 * y + 1.6],
            default=0.26 * x + 0.24 * y + 0.44,
        )
        return new_x, new_y

    return _ifs_image(step, c.shape, bounds, max_iter)


def sierpinskiTriangle(c, max_iter, bounds=(0.0, 1.0, 0.0, np.sqrt(3) / 2)):
    """Sierpinski Triangle. ``max_iter`` is the number of points sampled, in thousands."""
    vertices_x = np.array([0.0, 1.0, 0.5])
    vertices_y = np.array([0.0, 0.0, np.sqrt(3) / 2])

    def step(x, y, rng):
        # Move halfway toward a randomly chosen vertex.
        v = rng.integers(0, 3, size=x.size)
        return (x + vertices_x[v]) / 2, (y + vertices_y[v]) / 2

    return _ifs_image(step, c.shape, bounds, max_iter)
