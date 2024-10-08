import numpy as np
import bezier


def calculate_control_points(point, curve_strength):
    x, y = point
    normal_x, normal_y = y, -x
    control1 = [-curve_strength * normal_x + x / 2, -curve_strength * normal_y + y / 2]
    control2 = [curve_strength * normal_x + x / 2, curve_strength * normal_y + y / 2]
    return control1, control2


def generate_monotile(a, b, curve_strength):
    cos_angle = np.cos(np.pi / 3)
    sin_angle = np.sin(np.pi / 3)

    direction_vectors = [
        [cos_angle * b, sin_angle * b],
        [b, 0],
        [0, a],
        [sin_angle * a, cos_angle * a],
        [cos_angle * b, -sin_angle * b],
        [-cos_angle * b, -sin_angle * b],
        [sin_angle * a, -cos_angle * a],
        [0, -a],
        [0, -a],
        [-sin_angle * a, -cos_angle * a],
        [-cos_angle * b, sin_angle * b],
        [-b, 0],
        [0, a],
        [-sin_angle * a, cos_angle * a],
    ]

    x_vals, y_vals = [0], [0]
    for dx, dy in direction_vectors:
        control1, control2 = calculate_control_points([dx, dy], curve_strength)
        nodes = np.asfortranarray(
            [
                [
                    x_vals[-1],
                    control1[0] + x_vals[-1],
                    control2[0] + x_vals[-1],
                    dx + x_vals[-1],
                ],
                [
                    y_vals[-1],
                    control1[1] + y_vals[-1],
                    control2[1] + y_vals[-1],
                    dy + y_vals[-1],
                ],
            ]
        )

        curve = bezier.Curve(nodes, degree=3)
        s_vals = np.linspace(0.0, 1.0, 50)
        x_curve, y_curve = curve.evaluate_multi(s_vals)
        x_vals.extend(x_curve.tolist())
        y_vals.extend(y_curve.tolist())

    # Close the shape by connecting the last point to the first
    x_vals.append(x_vals[0])
    y_vals.append(y_vals[0])

    return np.array(x_vals), np.array(y_vals)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Draw a Spectre Monotile with given parameters."
    )
    parser.add_argument("a", type=float, help="Value for parameter a.")
    parser.add_argument("b", type=float, help="Value for parameter b.")
    parser.add_argument("curve_strength", type=float, help="Strength of the curve.")

    args = parser.parse_args()

    x_vals, y_vals = generate_monotile(args.a, args.b, args.curve_strength)

    plt.plot(x_vals, y_vals, color="black", linewidth=1)
    plt.axis("equal")
    plt.show()
