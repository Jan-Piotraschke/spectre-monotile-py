import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import spectre


# Argument Parser Setup
def setup_parser():
    parser = argparse.ArgumentParser(description="Fourier Epicycles Animation")
    parser.add_argument(
        "image_path", nargs="?", default=None, help="Path to the input image (optional)"
    )
    parser.add_argument(
        "--num_components",
        type=int,
        default=180,
        help="Number of Fourier components to include (default: 180)",
    )
    parser.add_argument(
        "--a", type=float, default=1.0, help="Value for parameter a (default: 1.0)"
    )
    parser.add_argument(
        "--b", type=float, default=1.0, help="Value for parameter b (default: 1.0)"
    )
    parser.add_argument(
        "--curve_strength",
        type=float,
        default=0.5,
        help="Strength of the curve (default: 0.5)",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Save the animation as a video file (default: False)",
    )
    return parser


# Image Processing Functions
def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    # Threshold the image to get a binary image
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("Error: No contours found in the image.")
        sys.exit(1)

    # Assuming the largest contour is the desired shape
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    x_points = contour[:, 0]
    y_points = -contour[:, 1]  # Invert y-coordinates for Matplotlib
    return x_points, y_points


# Fourier Transform Setup
def setup_fourier_transform(x_points, y_points, num_components):
    # Create complex representation of the points
    points = x_points + 1j * y_points

    # Number of samples along the contour path
    N = len(points)

    # Number of Fourier components to include
    num_components = min(num_components, N)

    # Compute the Fourier coefficients
    coefficients = np.fft.fft(points) / N
    # Frequencies corresponding to the Fourier coefficients
    freqs = np.fft.fftfreq(N, d=1 / N)  # Frequencies in cycles per unit time
    omega = 2 * np.pi * freqs  # Angular frequencies

    # Shift zero frequency component to the center
    coefficients = np.fft.fftshift(coefficients)
    omega = np.fft.fftshift(omega)

    # Sort the coefficients by magnitude (optional, for better visualization)
    indices = np.argsort(-np.abs(coefficients))
    coefficients = coefficients[indices]
    omega = omega[indices]

    return coefficients, omega, N, num_components


# Animation Setup
def init_animation():
    line.set_data([], [])
    path_line.set_data([], [])
    return line, path_line


def update_animation(frame, coefficients, omega, N, num_components):
    t_current = frame * (2 * np.pi / N)
    x, y = 0, 0
    xs, ys = [], []
    global circles
    # Remove previous circles
    for c in circles:
        c.remove()
    circles = []

    x_prev, y_prev = 0, 0
    # Loop over Fourier components<
    for n in range(num_components):
        coef, freq = coefficients[n], omega[n]
        x_prev, y_prev = x, y
        # Update position using Euler's formula
        x += np.real(coef * np.exp(1j * freq * t_current))
        y += np.imag(coef * np.exp(1j * freq * t_current))
        # Draw circle representing this Fourier component
        radius = np.abs(coef)
        circle = plt.Circle(
            (x_prev, y_prev), radius, fill=False, color="gray", alpha=0.3
        )
        ax.add_patch(circle)
        circles.append(circle)

        xs.append(x_prev)
        ys.append(y_prev)

    xs.append(x)
    ys.append(y)
    line.set_data(xs, ys)
    xdata.append(x)
    ydata.append(y)
    path_line.set_data(xdata, ydata)

    return line, path_line, *circles


# Main Execution Logic
def main():
    parser = setup_parser()
    args = parser.parse_args()

    if args.image_path:
        x_points, y_points = process_image(args.image_path)
    else:
        x_points, y_points = spectre.generate_monotile(
            args.a, args.b, args.curve_strength
        )

    # Normalize points
    x_points = (x_points - np.mean(x_points)) / np.max(
        np.abs(x_points - np.mean(x_points))
    )
    y_points = (y_points - np.mean(y_points)) / np.max(
        np.abs(y_points - np.mean(y_points))
    )

    coefficients, omega, N, num_components = setup_fourier_transform(
        x_points, y_points, args.num_components
    )

    # Setup plot
    global fig, ax, line, path_line, circles, xdata, ydata
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    (line,) = ax.plot([], [], "b-")
    (path_line,) = ax.plot([], [], "r-")
    circles = []
    xdata, ydata = [], []

    ani = FuncAnimation(
        fig,
        update_animation,
        frames=N,
        init_func=init_animation,
        fargs=(coefficients, omega, N, num_components),
        blit=True,
        interval=20,
    )

    if args.save:
        # Calculate fps for 10-second animation
        fps = N / 10
        ani.save("fourier_epicycles.mp4", writer="ffmpeg", fps=fps)
    else:
        plt.show()


if __name__ == "__main__":
    main()
