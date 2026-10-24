import fractalRender

FRACTALS = {
    "1": "Mandel Brot",
    "2": "Barnsley Fern",
    "3": "Burning ship",
    "4": "Newton's Fractal",
    "5": "Nova Fractal",
    "6": "Sierpinski Triangle",
}

DEFAULT_ITERATIONS = 100


def ask_iterations():
    # An empty input uses the default; anything else must be a positive integer.
    while True:
        text = input(
            f"Enter the number of iterations (default is {DEFAULT_ITERATIONS}): ").strip()
        if not text:
            return DEFAULT_ITERATIONS
        if text.isdigit() and int(text) >= 1:
            return int(text)
        print("Please enter a whole number of at least 1.")


if __name__ == "__main__":
    # Keep input handling in this file and delegate rendering to fractalRender.
    choice = input(
        "Enter the fractal type (Mandelbrot [1], Barnsley Fern [2], Burning ship [3], "
        "Newton's Fractal [4], Nova Fractal [5], Sierpinski Triangle [6]): ").strip()

    if choice in FRACTALS:
        fractalRender.render_fractal(FRACTALS[choice], ask_iterations())
    else:
        print("Invalid input. Please enter a valid option.")
