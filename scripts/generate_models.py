import sys
import os
import numpy as np

# 1. Path Setup
# The script is in /scripts, I need to find /src
# I go up one level and then into the src folder
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.append(src_path)

from pomdp_lib.models.generator import FoggyForestGenerator

def validate_math(model_data, name):
    # I check if the probabilities sum to 1.0
    # This ensures the model is mathematically correct
    T = model_data['T']
    Z = model_data['Z']
    b0 = model_data['b0']
    
    t_ok = np.allclose(T.sum(axis=2), 1.0)
    z_ok = np.allclose(Z.sum(axis=0), 1.0)
    b_ok = np.isclose(b0.sum(), 1.0)
    
    if t_ok and z_ok and b_ok:
        print(f"  ✓ Math validation passed for {name}.")
    else:
        print(f"  ✗ Warning: Math error in {name}. Check normalization.")

def main():
    # Define where to save the generated .py files
    models_dir = os.path.join(src_path, "pomdp_lib/models")
    
    # --- Part 1: Generate 3x3 Model ---
    print("Generating 3x3 model...")
    gen_3x3 = FoggyForestGenerator(
        width=3, height=3, exit_cell=(3, 2), traps=[(1, 1)], trees=[(2, 2)],
        robot_start=(1, 3), initial_belief_type="deterministic"
    )
    data_3x3 = gen_3x3.generate()
    gen_3x3.save_to_file(os.path.join(models_dir, "model_3x3.py"))

    # --- Part 2: Generate 5x5 Model ---
    print("Generating 5x5 model...")
    gen_5x5 = FoggyForestGenerator(
        width=5, height=5, exit_cell=(5, 2), traps=[(2, 4), (3, 2)], trees=[(4, 3)],
        robot_start=(1, 1), initial_belief_type="deterministic"
    )
    data_5x5 = gen_5x5.generate()
    gen_5x5.save_to_file(os.path.join(models_dir, "model_5x5.py"))

    # --- Part 3: Verification and Validation ---
    # I need to check if the generated files work correctly
    # This part verifies the state space S and the observation space O
    print("\n" + "="*40)
    print("RUNNING FINAL VERIFICATION")
    print("="*40)

    try:
        # Check the 3x3 model first
        from pomdp_lib.models import model_3x3
        print("\nVerification success: 3x3 model loaded.")
        print(f"Number of states (S): {len(model_3x3.S)}")
        
        # Validate the math logic
        validate_math(data_3x3, "3x3 Model")
        
        # I want to see how many observation signals exist
        print(f"Number of observations (O): {len(model_3x3.O)}")
        print(f"Observation labels: {model_3x3.O}")

        print("\n" + "-"*30 + "\n")

        # Check the 5x5 model next
        from pomdp_lib.models import model_5x5
        print("Verification success: 5x5 model loaded.")
        print(f"Number of states (S): {len(model_5x5.S)}")
        
        # Validate the math logic
        validate_math(data_5x5, "5x5 Model")
        
        # The 5x5 map is more complex so it might have more signal combinations
        print(f"Number of observations (O): {len(model_5x5.O)}")
        print(f"Observation labels: {model_5x5.O}")

    except ImportError as e:
        print("Error: Could not find the model files in pomdp_lib.models.")
        print(e)

if __name__ == "__main__":
    main()
