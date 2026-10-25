# Fractile

An interactive fractal viewer written in Python with twelve fractals to explore. Pick one, then zoom and pan around it. The image is recomputed for whatever part you are looking at.

A browser demo (`index.html`) is also included. It has the original six fractals only; the Python app has all twelve.

## Fractals

| Option | Fractal | Type |
| --- | --- | --- |
| 1 | Mandelbrot | Escape time |
| 2 | Barnsley Fern | Iterated function system |
| 3 | Burning Ship | Escape time |
| 4 | Newton's Fractal (z^3 - 1) | Convergence time |
| 5 | Nova Fractal | Convergence time |
| 6 | Sierpinski Triangle | Iterated function system |
| 7 | Julia Set | Escape time |
| 8 | Tricorn | Escape time |
| 9 | Multibrot (z^3 + c) | Escape time |
| 10 | Phoenix | Escape time |
| 11 | Sierpinski Carpet | Iterated function system |
| 12 | Dragon Curve | Iterated function system |

- **Julia Set:** the partner of the Mandelbrot set. Every point of the Mandelbrot set corresponds to a different Julia shape. This one uses a fixed constant.
- **Tricorn:** like the Mandelbrot set, but with the complex conjugate taken at each step, which gives three-fold symmetry.
- **Multibrot:** the Mandelbrot formula with a cube instead of a square, giving a two-lobed shape.
- **Phoenix:** a Julia-style fractal whose next value also depends on the one before it, giving flowing, feathered shapes.
- **Sierpinski Carpet:** the square version of the Sierpinski Triangle.
- **Dragon Curve:** the Heighway dragon, a space-filling curve made of two transformations.

## Requirements

- Python 3.10 or newer (developed on Python 3.13)
- numpy
- matplotlib
- numba (optional, but strongly recommended for speed)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If Numba fails to install, the program still works. It prints a short message and uses the slower NumPy code. See Troubleshooting below.

## Usage

```bash
python main.py
```

You will be asked for two things:

1. The fractal number (1 to 12). The menu lists them all.
2. The number of iterations. Press Enter to use the default of 100.

### What "iterations" means

- Mandelbrot, Burning Ship, Newton, Nova, Julia, Tricorn, Multibrot and Phoenix: the maximum number of iterations per pixel. Higher values show more detail near the edges of the set but take longer.
- Barnsley Fern, Sierpinski Triangle, Sierpinski Carpet and Dragon Curve: the number of points to plot, in thousands (100 means 100,000 points).

### Controls

| Action | Result |
| --- | --- |
| Scroll wheel | Zoom in or out around the mouse cursor |
| Toolbar zoom tool | Drag a rectangle to zoom to it |
| Toolbar pan tool | Drag to move the view |
| Toolbar home button | Return to the starting view |

After you stop zooming or panning, a quick low-resolution preview appears, followed by the full-resolution image.

## Project structure

| File | Purpose |
| --- | --- |
| `main.py` | Menu and input handling |
| `fractalRender.py` | matplotlib window, zoom and pan handling, preview and full renders |
| `fractals.py` | The fractal calculations (Numba kernels plus NumPy fallbacks) |
| `requirements.txt` | Python dependencies |

## Settings

These constants are at the top of `fractalRender.py`:

| Constant | Default | Effect |
| --- | --- | --- |
| `RENDER_SCALE` | `1.0` | Render resolution relative to the screen. Lower it (for example `0.5`) for faster, softer images. |
| `PREVIEW_SCALE` | `0.25` | Resolution of the quick preview shown while zooming. Set to `1.0` to disable the preview. |
| `REFRESH_DELAY_MS` | `150` | How long to wait after the last zoom or pan before redrawing. |
| `SCROLL_ZOOM` | `0.8` | How much each scroll step zooms. Smaller values zoom faster. |

## How it works

- Escape-time fractals (Mandelbrot, Burning Ship, Julia, Tricorn, Multibrot, Phoenix) iterate each point of the visible region until it escapes, and colour it by how long that took.
- Newton and Nova colour each point by how many steps it takes for the iteration to settle.
- The Fern, Sierpinski Triangle, Sierpinski Carpet and Dragon Curve are drawn by running many random chains of transformations and counting how many points land in each pixel.
- On every zoom or pan, the visible region is recomputed so detail stays sharp.

## Fractal parameters

A few fractals use fixed constants, defined near the top of `fractals.py`. Change them and rerun to get different shapes:

| Constant | Default | Used by |
| --- | --- | --- |
| `JULIA_C` | `-0.8 + 0.156j` | Julia Set. Try `-0.4 + 0.6j` or `0.285 + 0.01j`. |
| `PHOENIX_P` | `0.5667` | Phoenix |
| `PHOENIX_Q` | `-0.5` | Phoenix |

## Troubleshooting

**Numba will not install (Intel Macs).** Recent Numba and llvmlite releases do not ship prebuilt packages for Intel Macs on the newest Python versions, so pip tries to compile LLVM and fails. Try an older Python (3.13) in a fresh virtual environment, and install with:

```bash
pip install --only-binary=:all: numba
```

This fails immediately if no prebuilt package exists, instead of trying to build one. If it fails, the program still runs without Numba.

**Pylance or VS Code says `numba` could not be resolved.** Select the interpreter from your virtual environment (Command Palette, then "Python: Select Interpreter").

**The first run is slow.** Numba compiles the fractal code the first time it runs. The compiled result is cached, so later launches start quickly.

## Limitations

- Zooming deeper than about 1e-13 hits the limit of 64-bit floating point precision and the image turns blocky.
- Very deep zooms on the point-based fractals (Barnsley Fern, Sierpinski Triangle and Carpet, Dragon Curve) can look sparse, because few random points land in a tiny window. Increasing the iteration count helps.