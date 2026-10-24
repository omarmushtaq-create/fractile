import matplotlib.pyplot as plt
import numpy as np

import fractals

# 1.0 renders one fractal pixel per screen pixel. Lower it (e.g. 0.5) for a
# much faster, slightly softer image on slow machines.
RENDER_SCALE = 1.0
# Wait this long after the last zoom/pan change before recomputing.
REFRESH_DELAY_MS = 150
# Mouse-wheel zoom: each notch scales the view by this factor.
SCROLL_ZOOM = 0.8

GENERATORS = {
    "Burning ship": lambda c, n, bounds: fractals.burningShip(c, n),
    "Mandel Brot": lambda c, n, bounds: fractals.mandelBrot(c, n),
    "Newton's Fractal": lambda c, n, bounds: fractals.newtonsFractal(c, n),
    "Nova Fractal": lambda c, n, bounds: fractals.novaFractal(c, n),
    "Barnsley Fern": fractals.barnsleyFern,
    "Sierpinski Triangle": fractals.sierpinskiTriangle,
}

# Each fractal gets a coordinate window that contains its interesting features.
INITIAL_BOUNDS = {
    "Barnsley Fern": (-2.2, 2.7, 0.0, 10.0),
    "Sierpinski Triangle": (0.0, 1.0, 0.0, np.sqrt(3) / 2),
}
DEFAULT_BOUNDS = (-2.0, 2.0, -2.0, 2.0)


def render_fractal(fractal_type, max_itera):
    if fractal_type not in GENERATORS:
        raise ValueError(f"Unknown fractal type: {fractal_type}")
    generator = GENERATORS[fractal_type]
    initial_bounds = INITIAL_BOUNDS.get(fractal_type, DEFAULT_BOUNDS)

    figure, axes = plt.subplots(figsize=(10, 10))
    axes.set_title(fractal_type, fontsize=16)
    axes.set_axis_off()

    # origin="lower" makes row 0 of the array the bottom (ymin) of the view,
    # which matches how the complex grid is built.  The old origin="upper"
    # flipped the image vertically, so zooms landed on the mirrored spot.
    image = axes.imshow(np.zeros((2, 2)), extent=initial_bounds,
                        cmap="twilight_shifted", origin="lower",
                        aspect="equal", interpolation="nearest")
    axes.set_xlim(initial_bounds[:2])
    axes.set_ylim(initial_bounds[2:])

    def generate_image(bounds):
        # Size the grid to the axes actually on screen (not the whole canvas),
        # so no pixels are computed that are never shown.
        axes.apply_aspect()
        box = axes.get_window_extent()
        width = max(int(box.width * RENDER_SCALE), 1)
        height = max(int(box.height * RENDER_SCALE), 1)

        xmin, xmax, ymin, ymax = bounds
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        c = x[None, :] + 1j * y[:, None]
        return generator(c, max_itera, bounds)

    updating = False

    def refresh():
        nonlocal updating
        bounds = (*axes.get_xlim(), *axes.get_ylim())
        data = generate_image(bounds)
        # set_extent can touch the axes limits; ignore the events it fires.
        updating = True
        try:
            image.set_data(data)
            image.set_extent(bounds)
            # Colour limits are fixed at the first draw, so rescale them or
            # zoomed views (especially the fern) can look blank.
            image.set_clim(data.min(), data.max())
        finally:
            updating = False
        figure.canvas.draw_idle()

    # One-shot debounce timer: any burst of zoom/pan/resize events results in
    # a single recompute after things go quiet.
    refresh_timer = figure.canvas.new_timer(interval=REFRESH_DELAY_MS)
    refresh_timer.single_shot = True
    refresh_timer.add_callback(refresh)

    def schedule_refresh(*_args):
        if updating:
            return
        refresh_timer.stop()
        refresh_timer.start()

    def on_scroll(event):
        if event.inaxes is not axes or event.xdata is None:
            return
        factor = SCROLL_ZOOM if event.button == "up" else 1 / SCROLL_ZOOM
        x0, x1 = axes.get_xlim()
        y0, y1 = axes.get_ylim()
        axes.set_xlim(event.xdata - (event.xdata - x0) * factor,
                      event.xdata + (x1 - event.xdata) * factor)
        axes.set_ylim(event.ydata - (event.ydata - y0) * factor,
                      event.ydata + (y1 - event.ydata) * factor)
        figure.canvas.draw_idle()

    refresh()  # initial image, before the callbacks are connected
    axes.callbacks.connect("xlim_changed", schedule_refresh)
    axes.callbacks.connect("ylim_changed", schedule_refresh)
    figure.canvas.mpl_connect("resize_event", schedule_refresh)
    figure.canvas.mpl_connect("scroll_event", on_scroll)
    plt.show()
