import cv2
import numpy as np
import os

def debug():
    ref_path = "data/frames_ref/frame_0040.png"
    gen_path = "data/best_paper_result_0040.png"
    
    if not os.path.exists(ref_path) or not os.path.exists(gen_path):
        print("Missing files")
        return

    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    gen = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)
    gen = cv2.resize(gen, (ref.shape[1], ref.shape[0]))

    # Re-run the mask logic
    from metrics_paper import detect_pause_icon_mask, crease_map, iou_binary
    
    mask = detect_pause_icon_mask(ref)
    
    cm_ref = crease_map(ref, ignore_mask=mask)
    cm_gen = crease_map(gen, ignore_mask=mask)
    
    # Calculate coverage
    gen_binary = (gen > 10).astype(np.uint8) # Approx foreground
    ref_binary = (ref > 10).astype(np.uint8)
    
    gen_pixels = gen_binary.sum()
    ref_pixels = ref_binary.sum()
    coverage = gen_pixels / (ref_pixels + 1e-9)
    
    print(f"Gen Pixels: {gen_pixels}")
    print(f"Ref Pixels: {ref_pixels}")
    print(f"Coverage: {coverage:.4f}")
    
    iou = iou_binary(cm_ref, cm_gen)
    print(f"IoU: {iou:.4f}")
    
    # Save debug images
    cv2.imwrite("data/debug_mask.png", mask)
    cv2.imwrite("data/debug_cm_ref.png", cm_ref)
    cv2.imwrite("data/debug_cm_gen.png", cm_gen)
    
    # Visualizing overlap
    overlap = np.zeros_(ref.shape, dtype=np.uint8)
    # R = Ref, G = Gen
    debug_view = np.zeros((ref.shape[0], ref.shape[1], 3), dtype=np.uint8)
    debug_view[:,:,2] = cm_ref # Red
    debug_view[:,:,1] = cm_gen # Green
    cv2.imwrite("data/debug_overlap.png", debug_view)

if __name__ == "__main__":
    debug()
