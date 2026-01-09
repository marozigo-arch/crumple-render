import cv2
import numpy as np
import os

def create_comparison(ref_path, gen_path, output_path):
    if not os.path.exists(ref_path):
        print(f"Reference frame not found: {ref_path}")
        return
    if not os.path.exists(gen_path):
        print(f"Generated frame not found: {gen_path}")
        return

    # Read images
    img_ref = cv2.imread(ref_path)
    img_gen = cv2.imread(gen_path)
    
    # Resize generated to match reference if needed (they should overlap ideally)
    if img_ref.shape != img_gen.shape:
        print(f"Resizing generated {img_gen.shape} to match reference {img_ref.shape}")
        img_gen = cv2.resize(img_gen, (img_ref.shape[1], img_ref.shape[0]))

    # Create side-by-side
    side_by_side = np.hstack((img_ref, img_gen))
    
    # Create absolute difference (to see hotspots)
    diff = cv2.absdiff(img_ref, img_gen)
    # Enhance diff visibility
    diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    
    # Stack vertically: [Ref | Gen]
    #                   [   Diff  ] (centered or stretched)
    
    # Make diff same width as side_by_side
    diff_resized = cv2.resize(diff, (side_by_side.shape[1], side_by_side.shape[0]))
    
    # Actually, let's just do Ref | Gen | Diff
    final_comp = np.hstack((img_ref, img_gen, cv2.resize(diff, (img_ref.shape[1], img_ref.shape[0]))))

    cv2.imwrite(output_path, final_comp)
    print(f"Comparison saved to {output_path}")

if __name__ == "__main__":
    # Experiment 7: Pseudo-Isometric (Scale Fix)
    ref_frame = "data/frames_ref/frame_0040.png"
    gen_frame = "data/render_uncrumple_test/frame_0040.png"
    
    create_comparison(ref_frame, gen_frame, "data/comparison_check_40_no_stretch.png")
