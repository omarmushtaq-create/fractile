import fractalRender

# Menu number -> (name shown in the menu, name used by the renderer).
FRACTALS = {
    "1": "Mandel Brot",
    "2": "Barnsley Fern",
    "3": "Burning ship",
    "4": "Newton's Fractal",
    "5": "Nova Fractal",
    "6": "Sierpinski Triangle",
    "7": "Julia Set",
    "8": "Tricorn",
    "9": "Multibrot",
    "10": "Phoenix",
    "11": "Sierpinski Carpet",
    "12": "Dragon Curve",
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
    print("Choose a fractal:")
    for number, name in FRACTALS.items():
        print(f"  [{number}] {name}")
    choice = input("Enter the fractal number: ").strip()

    if choice in FRACTALS:
        fractalRender.render_fractal(FRACTALS[choice], ask_iterations())
    else:
        print("Invalid input. Please enter a valid option.")
