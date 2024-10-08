import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

# Load and preprocess the image
# Replace 'your_image.png' with the path to your image file
img = cv2.imread("your_image.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# Threshold the image to get a binary image
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Check if any contour is found
if not contours:
    raise ValueError("No contours found in the image.")

# Assuming the largest contour is the desired shape
contour = max(contours, key=cv2.contourArea)

# Reshape and extract contour points
contour = contour.reshape(-1, 2)
x_points = contour[:, 0]
y_points = contour[:, 1]

# Invert y-coordinates to match Matplotlib's coordinate system
y_points = -y_points

# Normalize points to range [-1, 1] for better visualization
x_points = (x_points - np.mean(x_points)) / np.max(np.abs(x_points - np.mean(x_points)))
y_points = (y_points - np.mean(y_points)) / np.max(np.abs(y_points - np.mean(y_points)))

# Create complex representation of the points
points = x_points + 1j * y_points

# Number of samples along the contour path
N = len(points)

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

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis("off")

# Initialize plot elements
(line,) = ax.plot([], [], "b-")  # Line representing the epicycles
(path_line,) = ax.plot([], [], "r-")  # Line representing the drawn path
circles = []  # List to keep track of drawn circles

# Number of Fourier components to include
num_components = 180


# Initialize the animation
def init():
    line.set_data([], [])
    path_line.set_data([], [])
    return line, path_line


# Lists to store the path coordinates
xdata, ydata = [], []


# Update function for animation
def update(frame):
    t_current = frame * (2 * np.pi / N)
    x = 0
    y = 0
    xs = []
    ys = []
    global circles
    # Remove previous circles
    for c in circles:
        c.remove()
    circles = []
    x_prev = 0
    y_prev = 0
    # Loop over Fourier components
    for n in range(num_components):
        freq = omega[n]
        coef = coefficients[n]
        x_prev = x
        y_prev = y
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
    # Update the lines
    line.set_data(xs, ys)
    xdata.append(x)
    ydata.append(y)
    path_line.set_data(xdata, ydata)

    return line, path_line, *circles


# Create the animation
ani = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=20)

# # Calculate fps for 10-second animation
# fps = N / 10

# # Save the animation as a video file (MP4 format)
# ani.save("fourier_epicycles.mp4", writer="ffmpeg", fps=fps)

plt.show()
