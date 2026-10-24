import fractalRender

if __name__ == "__main__":

    input_fractal = input(
        "Enter the fractal type (Mandelbrot [1], Barnsley Fern [2], Burning ship [3], Newton's Fractal [4], Nova Fractal [5], Sierpinski Triangle [6]): ")

    input_iterations = input("Enter the number of iterations (default is 100): ")
    if input_fractal == "1":
        fractalRender.render_fractal("Mandel Brot", int(input_iterations) if input_iterations else 100)
    elif input_fractal == "2":
        fractalRender.render_fractal("Barnsley Fern", int(
            input_iterations) if input_iterations else 100)
    elif input_fractal == "3":
        fractalRender.render_fractal("Burning ship", int(input_iterations) if input_iterations else 100)
    elif input_fractal == "4":
        fractalRender.render_fractal("Newton's Fractal", int(input_iterations) if input_iterations else 100)
    elif input_fractal == "5":
        fractalRender.render_fractal("Nova Fractal", int(input_iterations) if input_iterations else 100)
    elif input_fractal == "6":
        fractalRender.render_fractal("Sierpinski Triangle", int(input_iterations) if input_iterations else 100)
    else:
        print("Invalid input. Please enter a valid option.")