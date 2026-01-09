
import cv2
import numpy as np
import os

INPUT_DIR = "/workspaces/codespaces-jupyter/data/render_crumple_v8"
OUTPUT_FILE = os.path.join(INPUT_DIR, "collage_v8.png")

def make_collage():
    # Order: Flat (44) -> Crumpled (33)
    # We want to show the progression: 44, 42, 40, 33
    frames = [44, 42, 40, 33]
    images = []
    
    images_ref = []
    images_gen = []
    
    # Path to Reference Frames
    REF_DIR = "/workspaces/codespaces-jupyter/data/frames_ref"
    
    for f in frames:
        # Load Gen
        path_gen = os.path.join(INPUT_DIR, f"val_frame_{f:04d}.png")
        # Load Ref
        path_ref = os.path.join(REF_DIR, f"frame_{f:04d}.png")
        
        img_gen = None
        img_ref = None
        
        if os.path.exists(path_gen):
            img_gen = cv2.imread(path_gen)
            cv2.putText(img_gen, f"Gen {f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3) # Green text
        else:
            print(f"Warning: Gen {path_gen} not found")
            # Create black placeholder
            img_gen = np.zeros((1104, 864, 3), dtype=np.uint8)
            
        if os.path.exists(path_ref):
            img_ref = cv2.imread(path_ref)
            cv2.putText(img_ref, f"Ref {f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3) # Red text
        else:
            print(f"Warning: Ref {path_ref} not found")
            img_ref = np.zeros((1104, 864, 3), dtype=np.uint8)
            
        images_gen.append(img_gen)
        images_ref.append(img_ref)

    if len(images_gen) != 4:
        return

    # Stack:
    # Row 1: Ref 44, Ref 42, Ref 40, Ref 33
    # Row 2: Gen 44, Gen 42, Gen 40, Gen 33
    
    row_ref = np.hstack(images_ref)
    row_gen = np.hstack(images_gen)
    
    grid = np.vstack((row_ref, row_gen))
    
    cv2.imwrite(OUTPUT_FILE, grid)
    print(f"Saved comparison collage to {OUTPUT_FILE}")

if __name__ == "__main__":
    make_collage()
