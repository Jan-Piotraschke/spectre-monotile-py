import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def find_synchronization_point():
    print("--- SEARCHING FOR THE ZERO-ENTROPY JANUS POINT ---")
    
    lambda_F = sp.Rational(34, 13)
    delta = 1 / lambda_F
    target_action = float((17 * lambda_F).evalf()) # 44.461538
    
    # We test a range of 'Effective Vertices' (including boundary overlaps)
    v_range = np.linspace(116, 117, 100)
    residuals = []
    
    for v in v_range:
        action = v * float(delta.evalf())
        residuals.append(abs(action - target_action))
    
    # Find the exact vertex count for Q^2 = 0
    v_ideal = target_action / float(delta.evalf())
    
    print(f"Target Sync Action: {target_action:.6f}")
    print(f"Ideal Vertex Count: {v_ideal:.6f}")
    print(f"Pruning Requirement: {117 - v_ideal:.6f} vertices")
    
    if 116 < v_ideal < 117:
        print("\nSUCCESS: THE JANUS POINT EXISTS WITHIN THE NINE-TILE CLUSTER")
        print("The 0.72 vertex deficit represents the shared 'Aperiodic Joints'.")
    
    # Visualization of the Synchronization Dip
    plt.figure(figsize=(8, 5))
    plt.plot(v_range, residuals, color='cyan', label='BRST Residual (Q^2)')
    plt.axvline(v_ideal, color='red', linestyle='--', label=f'Sync Point ({v_ideal:.3f})')
    plt.title("Vacuum Synchronization: Finding Q^2 = 0")
    plt.xlabel("Effective Vertices (V_eff)")
    plt.ylabel("Jitter Magnitude")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

if __name__ == "__main__":
    find_synchronization_point()