import sympy as sp
import numpy as np

def test_metatile_uniqueness():
    print("--- TESTING METATILE UNIQUENESS: THE 117-VERTEX LIMIT ---")
    
    # Fundamental Constants from out.tex
    lambda_F = sp.Rational(34, 13)
    delta = 1 / lambda_F
    resonance_target = sp.Integer(17)
    
    # The "Ideal" Action where Q^2 = 0
    target_action = float((resonance_target * lambda_F).evalf())
    
    # Testing different vertex counts (Pruning constants)
    test_vertices = [116, 117, 118]
    
    print(f"Target Sync Action: {target_action:.6f}\n")
    print(f"{'Vertices':<10} | {'Calculated Action':<20} | {'Q^2 (Residual)':<15} | {'Status'}")
    print("-" * 75)
    
    for v in test_vertices:
        # Calculate Action for this vertex count
        action = float((v * delta).evalf())
        
        # Calculate Q^2 (Residual)
        q_squared = abs(action - target_action)
        
        status = "SYNCHRONIZED" if q_squared < 1e-10 else "WILD / UNSTABLE"
        
        print(f"{v:<10} | {action:<20.6f} | {q_squared:<15.6f} | {status}")

if __name__ == "__main__":
    test_metatile_uniqueness()