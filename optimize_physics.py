import os
import subprocess
import shutil
import re

def get_iou_score(output_line):
    # Parse line like "Crease Map IoU: 0.8292"
    match = re.search(r"Crease Map IoU: ([0-9.]+)", output_line)
    if match:
        return float(match.group(1))
    return 0.0

def run_optimization():
    # Grid of parameters to test
    # We want sharp, paper-like creases without stretching.
    # Higher Tension = Less Stretch.
    # Higher Bending = Stiffer folds (cardboard vs tissue).
    # Turbulence = Force of crumple.
    
    settings_grid = [
        {"mass": 0.3, "tension": 80, "bending": 15, "turb": 20}, # Baseline
        {"mass": 0.3, "tension": 120, "bending": 25, "turb": 25}, # Stiffer
        {"mass": 0.3, "tension": 150, "bending": 5, "turb": 15}, # More crinkly/thin
        {"mass": 0.5, "tension": 60, "bending": 40, "turb": 30}, # Heavy/Thick
        {"mass": 0.2, "tension": 100, "bending": 2, "turb": 10}, # Very thin/tissue
        {"mass": 0.3, "tension": 120, "bending": 0.5, "turb": 25}, # Extremely crinkly
    ]
    
    best_iou = -1.0
    best_config = None
    best_image = "data/render_cloth_test/frame_0040.png"
    
    ref_image = "data/frames_ref/frame_0040.png"
    
    print("Starting Optimization Loop...")
    
    for i, config in enumerate(settings_grid):
        print(f"\n--- Iteration {i+1}/{len(settings_grid)}: {config} ---")
        
        # 1. Run Simulation
        cmd_sim = [
            "blender", "-b", "-P", "animate_cloth.py", "--",
            "--mass", str(config["mass"]),
            "--tension", str(config["tension"]),
            "--bending", str(config["bending"]),
            "--turbulence", str(config["turb"])
        ]
        
        try:
            subprocess.run(cmd_sim, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("Simulation failed.")
            continue
            
        # 2. Verify Result
        out_img = "data/render_cloth_test/frame_0040.png"
        cmd_verify = [
            "python3", "verify_structure.py",
            ref_image, out_img
        ]
        
        result = subprocess.run(cmd_verify, capture_output=True, text=True)
        output = result.stdout
        print(output.strip())
        
        iou = get_iou_score(output)
        print(f"--> IoU Score: {iou:.4f}")
        
        if iou > best_iou:
            best_iou = iou
            best_config = config
            # Save "Best" copy
            shutil.copy(out_img, "data/best_cloth_result.png")
            shutil.copy("data/structure_comparison.png", "data/best_structure_comparison.png")
            print("NEW BEST FOUND!")
            
    print("\noptimization Complete.")
    print(f"Winner: {best_config}")
    print(f"Best IoU: {best_iou:.4f}")
    print("Saved to data/best_cloth_result.png")

if __name__ == "__main__":
    run_optimization()
