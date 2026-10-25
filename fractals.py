import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception as error:  # Falls back to the NumPy implementations below.
    # Numba can fail to import for reasons other than ImportError (for
    # example a NumPy version mismatch), so catch everything and say why.
    print(
        f"Numba unavailable ({type(error).__name__}: {error}); using NumPy fallback.")
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        return lambda func: func

    prange = range

# Parameters for the fractals that need a fixed constant. Change these to
# explore different Julia and Phoenix shapes.
JULIA_C = complex(-0.8, 0.156)
PHOENIX_P = 0.5667
PHOENIX_Q = -0.5
# fixed at 3 (z^3 + c); the kernel is written for the cubic
MULTIBROT_POWER = 3


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


def _mandelbrot_np(c, max_iter):
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


def _burning_ship_np(c, max_iter):
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


def _newton_np(c, max_iter, tol=1e-6):
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


def _nova_np(c, max_iter, R=1.0, tol=1e-6):
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


# ---------------------------------------------------------------------------
# Numba kernels: one compiled loop per pixel, rows run in parallel, and each
# pixel stops the moment it escapes or converges.  fastmath is left off for
# Newton/Nova because they rely on NaN/inf checks.
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _mandelbrot_nb(c, max_iter):
    h, w = c.shape
    out = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        for k in range(w):
            cr = c[r, k].real
            ci = c[r, k].imag
            # Skip the main cardioid and period-2 bulb: they never escape.
            q = (cr - 0.25) ** 2 + ci * ci
            if q * (q + (cr - 0.25)) <= 0.25 * ci * ci or (cr + 1.0) ** 2 + ci * ci <= 0.0625:
                continue
            zr = 0.0
            zi = 0.0
            for i in range(1, max_iter + 1):
                zr2 = zr * zr
                zi2 = zi * zi
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
                if zr * zr + zi * zi > 4.0:
                    out[r, k] = i
                    break
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _burning_ship_nb(c, max_iter):
    h, w = c.shape
    out = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        for k in range(w):
            cr = c[r, k].real
            ci = c[r, k].imag
            zr = 0.0
            zi = 0.0
            for i in range(1, max_iter + 1):
                x = abs(zr)
                y = abs(zi)
                zr = x * x - y * y + cr
                zi = 2.0 * x * y + ci
                if zr * zr + zi * zi > 4.0:
                    out[r, k] = i
                    break
    return out


@njit(parallel=True, cache=True)
def _newton_nb(c, max_iter, tol2):
    h, w = c.shape
    out = np.full((h, w), max_iter, dtype=np.int32)
    for r in prange(h):
        for k in range(w):
            zr = c[r, k].real
            zi = c[r, k].imag
            for i in range(1, max_iter + 1):
                a = zr * zr - zi * zi          # z^2 = a + bi
                b = 2.0 * zr * zi
                d = a * a + b * b
                if d == 0.0:                   # landed on 0: update undefined
                    out[r, k] = i
                    break
                # z - (z^3 - 1) / (3 z^2) == (2z + 1/z^2) / 3
                nzr = (2.0 * zr + a / d) / 3.0
                nzi = (2.0 * zi - b / d) / 3.0
                dr = nzr - zr
                di = nzi - zi
                zr = nzr
                zi = nzi
                if dr * dr + di * di < tol2:
                    out[r, k] = i
                    break
    return out


@njit(parallel=True, cache=True)
def _nova_nb(c, max_iter, R, tol2):
    h, w = c.shape
    out = np.full((h, w), max_iter, dtype=np.int32)
    for r in prange(h):
        for k in range(w):
            cr = c[r, k].real
            ci = c[r, k].imag
            zr = 1.0                           # Nova starts at z0 = 1
            zi = 0.0
            for i in range(1, max_iter + 1):
                a = zr * zr - zi * zi
                b = 2.0 * zr * zi
                d = a * a + b * b
                if d == 0.0:
                    out[r, k] = i
                    break
                nzr = zr - R * (zr - a / d) / 3.0 + cr
                nzi = zi - R * (zi + b / d) / 3.0 + ci
                dr = nzr - zr
                di = nzi - zi
                zr = nzr
                zi = nzi
                if dr * dr + di * di < tol2 or zr * zr + zi * zi > 1e12:
                    out[r, k] = i
                    break
    return out


def mandelBrot(c, max_iter):
    if HAVE_NUMBA:
        return _mandelbrot_nb(np.ascontiguousarray(c, dtype=np.complex128), int(max_iter))
    return _mandelbrot_np(c, max_iter)


def burningShip(c, max_iter):
    if HAVE_NUMBA:
        return _burning_ship_nb(np.ascontiguousarray(c, dtype=np.complex128), int(max_iter))
    return _burning_ship_np(c, max_iter)


def newtonsFractal(c, max_iter, tol=1e-6):
    if HAVE_NUMBA:
        return _newton_nb(np.ascontiguousarray(c, dtype=np.complex128), int(max_iter), tol * tol)
    return _newton_np(c, max_iter, tol)


def novaFractal(c, max_iter, R=1.0, tol=1e-6):
    if HAVE_NUMBA:
        return _nova_nb(np.ascontiguousarray(c, dtype=np.complex128), int(max_iter), float(R), tol * tol)
    return _nova_np(c, max_iter, R, tol)


# ---------------------------------------------------------------------------
# Additional escape-time fractals: NumPy fallbacks, Numba kernels, dispatchers.
# ---------------------------------------------------------------------------
def _escaped(old, new):
    return new.real ** 2 + new.imag ** 2 > 4.0


def _julia_np(c, max_iter):
    c = np.asarray(c, dtype=np.complex128)
    out = _iterate(c.ravel().copy(), np.full(c.size, JULIA_C, dtype=np.complex128),
                   lambda z, k: z * z + k, _escaped, max_iter)
    return out.reshape(c.shape)


def _tricorn_np(c, max_iter):
    c = np.asarray(c, dtype=np.complex128)
    out = _iterate(np.zeros(c.size, dtype=np.complex128), c.ravel(),
                   lambda z, k: np.conj(z) ** 2 + k, _escaped, max_iter)
    return out.reshape(c.shape)


def _multibrot_np(c, max_iter):
    c = np.asarray(c, dtype=np.complex128)
    out = _iterate(np.zeros(c.size, dtype=np.complex128), c.ravel(),
                   lambda z, k: z * z * z + k, _escaped, max_iter)
    return out.reshape(c.shape)


def _phoenix_np(c, max_iter):
    # Phoenix needs the previous value too, so it keeps its own active set.
    c = np.asarray(c, dtype=np.complex128)
    z = c.ravel().copy()
    prev = np.zeros_like(z)
    out = np.zeros(z.size, dtype=np.int32)
    idx = np.arange(z.size)
    with np.errstate(all="ignore"):
        for i in range(1, max_iter + 1):
            if idx.size == 0:
                break
            z_new = z * z + PHOENIX_P + PHOENIX_Q * prev
            done = z_new.real ** 2 + z_new.imag ** 2 > 4.0
            if done.any():
                out[idx[done]] = i
                alive = ~done
                idx, prev, z = idx[alive], z[alive], z_new[alive]
            else:
                prev, z = z, z_new
    return out.reshape(c.shape)


@njit(parallel=True, fastmath=True, cache=True)
def _julia_nb(c, max_iter, jr, ji):
    h, w = c.shape
    out = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        for k in range(w):
            zr = c[r, k].real
            zi = c[r, k].imag
            for i in range(1, max_iter + 1):
                t = zr * zr - zi * zi + jr
                zi = 2.0 * zr * zi + ji
                zr = t
                if zr * zr + zi * zi > 4.0:
                    out[r, k] = i
                    break
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _tricorn_nb(c, max_iter):
    h, w = c.shape
    out = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        for k in range(w):
            cr = c[r, k].real
            ci = c[r, k].imag
            zr = 0.0
            zi = 0.0
            for i in range(1, max_iter + 1):
                t = zr * zr - zi * zi + cr
                zi = -2.0 * zr * zi + ci   # conjugate before squaring
                zr = t
                if zr * zr + zi * zi > 4.0:
                    out[r, k] = i
                    break
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _multibrot_nb(c, max_iter):
    h, w = c.shape
    out = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        for k in range(w):
            cr = c[r, k].real
            ci = c[r, k].imag
            zr = 0.0
            zi = 0.0
            for i in range(1, max_iter + 1):
                a = zr * zr - zi * zi          # z^2 = a + bi
                b = 2.0 * zr * zi
                t = a * zr - b * zi + cr       # z^3 + c
                zi = a * zi + b * zr + ci
                zr = t
                if zr * zr + zi * zi > 4.0:
                    out[r, k] = i
                    break
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _phoenix_nb(c, max_iter, p, q):
    h, w = c.shape
    out = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        for k in range(w):
            zr = c[r, k].real
            zi = c[r, k].imag
            pr = 0.0
            pi = 0.0
            for i in range(1, max_iter + 1):
                nr = zr * zr - zi * zi + p + q * pr
                ni = 2.0 * zr * zi + q * pi
                pr = zr
                pi = zi
                zr = nr
                zi = ni
                if zr * zr + zi * zi > 4.0:
                    out[r, k] = i
                    break
    return out


def _as_complex(c):
    return np.ascontiguousarray(c, dtype=np.complex128)


def juliaSet(c, max_iter):
    if HAVE_NUMBA:
        return _julia_nb(_as_complex(c), int(max_iter), JULIA_C.real, JULIA_C.imag)
    return _julia_np(c, max_iter)


def tricorn(c, max_iter):
    if HAVE_NUMBA:
        return _tricorn_nb(_as_complex(c), int(max_iter))
    return _tricorn_np(c, max_iter)


def multibrot(c, max_iter):
    if HAVE_NUMBA:
        return _multibrot_nb(_as_complex(c), int(max_iter))
    return _multibrot_np(c, max_iter)


def phoenix(c, max_iter):
    if HAVE_NUMBA:
        return _phoenix_nb(_as_complex(c), int(max_iter), PHOENIX_P, PHOENIX_Q)
    return _phoenix_np(c, max_iter)


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


def sierpinskiCarpet(c, max_iter, bounds=(0.0, 1.0, 0.0, 1.0)):
    """Sierpinski Carpet. ``max_iter`` is the number of points sampled, in thousands."""
    offsets = np.array([(i / 3, j / 3) for i in range(3)
                       for j in range(3) if (i, j) != (1, 1)])

    def step(x, y, rng):
        # Shrink to a third and drop into one of the 8 outer squares.
        k = rng.integers(0, 8, size=x.size)
        return x / 3 + offsets[k, 0], y / 3 + offsets[k, 1]

    return _ifs_image(step, c.shape, bounds, max_iter)


def dragonCurve(c, max_iter, bounds=(-0.4, 1.25, -0.4, 0.75)):
    """Heighway Dragon. ``max_iter`` is the number of points sampled, in thousands."""

    def step(x, y, rng):
        first = rng.random(x.size) < 0.5
        return (np.where(first, 0.5 * x - 0.5 * y, -0.5 * x - 0.5 * y + 1.0),
                np.where(first, 0.5 * x + 0.5 * y, 0.5 * x - 0.5 * y))

    return _ifs_image(step, c.shape, bounds, max_iter)
