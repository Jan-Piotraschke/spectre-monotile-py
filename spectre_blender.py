#!/usr/bin/env python3
import numpy as np
import sys, subprocess
try:
    import bpy
except:
    bpy = None

if len(sys.argv) > 1: exargs = sys.argv[1:]
else: exargs = []
if bpy is None and __name__=='__main__':
    cmd = [
        'blender', 
        '--python-exit-code', '1', 
        '--python-use-system-env', 
        '--python', __file__,
    ]
    if exargs:
        cmd += ['--'] + exargs
    print(cmd)
    subprocess.check_call(cmd)
    sys.exit()

def get_params_from_args():
    params = {'a': 10.0, 'b': 10.0, 'curve_strength': 5}
    if len(exargs) >= 3:
        params = {'a': float(exargs[-3]), 'b': float(exargs[-2]), 'curve_strength': float(exargs[-1])}
    return params    

def generate(a=10, b=10, curve_strength=5):
    bezier_points_data = generate_monotile_bezier_data(a,b, curve_strength)
    
    curve_data = bpy.data.curves.new(name='Spectre', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.03
    curve_data.bevel_resolution = 4
    curve_data.extrude = 0.25
    
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(bezier_points_data) - 1)
    
    # Loop over the generated Bezier points data and set the Blender spline's properties
    for i, p_data in enumerate(bezier_points_data):
        p = spline.bezier_points[i]
        
        # Point location (co)
        p.co.x = p_data[0]
        p.co.y = p_data[1]
        p.co.z = 0.0
        
        # Set handles using the calculated fixed geometry
        p.handle_left.x = p_data[2]
        p.handle_left.y = p_data[3]
        p.handle_left.z = 0.0
        
        p.handle_right.x = p_data[4]
        p.handle_right.y = p_data[5]
        p.handle_right.z = 0.0
        
        # Using 'FREE' to prevent Blender's smoothing interference
        p.handle_right_type = 'FREE'
        p.handle_left_type = 'FREE'

    # Close the curve
    spline.use_cyclic_u = True
    
    curve_obj = bpy.data.objects.new(curve_data.name, curve_data)
    curve_obj.data.bevel_resolution = 1
    curve_obj.show_wire = True
    bpy.context.collection.objects.link(curve_obj)
    return curve_obj


def calculate_handle_points(start_point, end_point, curve_strength):
    """
    Calculates the two fixed interior control points (handles) for a Spectre Bezier segment 
    based on the required 1/3 handle length and 1/6 perpendicular offset, scaled by curve_strength.
    """
    
    # 1. Segment vector (P3 - P0) and its length L
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    L = np.sqrt(dx**2 + dy**2)
    
    if L < 1e-6: # Avoid division by zero for zero-length segments
        return [start_point[0], start_point[1]], [end_point[0], end_point[1]]

    # Fixed factors for the Spectre s-curve
    handle_length_factor = 1.0 / 3.0 
    base_offset_factor = 1.0 / 6.0 
    
    # 2. Components along the chord
    # Chord component length = L * (1/3) -> This sets the handle length and is NOT scaled
    chord_comp_x = dx * handle_length_factor
    chord_comp_y = dy * handle_length_factor
    
    # 3. Components perpendicular to the chord (THE SCALABLE PART)
    
    # Perpendicular vector [ -dy, dx ]
    perp_x, perp_y = -dy, dx 
    
    # Normalize the perpendicular vector to get the unit normal vector n_hat
    n_hat_x = perp_x / L
    n_hat_y = perp_y / L
    
    # Perpendicular offset distance D = L * (1/6) * curve_strength
    # The base 1/6 distance is scaled by the curve_strength
    offset_distance = L * base_offset_factor * curve_strength
    
    # Offset vector = D * n_hat
    offset_x = offset_distance * n_hat_x
    offset_y = offset_distance * n_hat_y
    
    # 4. Handle Calculations
    
    # P1 (handle_right of start_point P0)
    # P1 = P0 + (Chord Component) + (Offset Vector)
    handle1_x = start_point[0] + chord_comp_x + offset_x
    handle1_y = start_point[1] + chord_comp_y + offset_y
    
    # P2 (handle_left of end_point P3)
    # P2 = P3 - (Chord Component) - (Offset Vector)
    handle2_x = end_point[0] - chord_comp_x - offset_x
    handle2_y = end_point[1] - chord_comp_y - offset_y
    
    return [handle1_x, handle1_y], [handle2_x, handle2_y]



def generate_monotile_bezier_data(a, b, curve_strength):
    """
    Generates data required for a Blender Bezier curve using fixed Spectre handle geometry.
    """
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

    current_point = np.array([0.0, 0.0])
    points = [current_point.copy()]
    
    for dx, dy in direction_vectors:
        current_point += np.array([dx, dy])
        points.append(current_point.copy())

    # --- Generate Bezier Point Data ---
    bezier_points_data = []
    num_segments = len(points) - 1
    segment_handles = []
    
    for i in range(num_segments):
        start_point = points[i]
        end_point = points[i+1]
        
        # Corrected: Pass curve_strength
        hr_i, hl_i_plus_1 = calculate_handle_points(start_point, end_point, curve_strength)
        segment_handles.append((hr_i, hl_i_plus_1))

    # Stitch the handle data together for each point
    for i in range(num_segments):
        co_x, co_y = points[i][0], points[i][1]
        
        # 1. Right handle (hr): from the current segment (i)
        hr_x, hr_y = segment_handles[i][0]
        
        # 2. Left handle (hl): from the previous segment (i-1)
        prev_segment_index = (i - 1) % num_segments
        hl_x, hl_y = segment_handles[prev_segment_index][1] 

        bezier_points_data.append((co_x, co_y, hl_x, hl_y, hr_x, hr_y))
    
    return bezier_points_data


if __name__ == "__main__":
    params = get_params_from_args()
    generate(**params)
