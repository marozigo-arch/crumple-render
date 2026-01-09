import cv2
import numpy as np
import sys

# Removed skimage dependency
# from skimage.metrics import structural_similarity as ssim

def get_crease_map(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return None
    
    # Invert
    inv_img = cv2.bitwise_not(img)
    
    # Adaptive Threshold for local contrast (creases)
    adaptive = cv2.adaptiveThreshold(inv_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Clean noise
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
    
    return opened

def compare_structure(ref_path, gen_path, out_vis_path=None):
    map_ref = get_crease_map(ref_path)
    map_gen = get_crease_map(gen_path)
    
    if map_ref is None or map_gen is None:
        print("Error loading images")
        return 0.0
        
    # Resize gen to match ref
    if map_ref.shape != map_gen.shape:
        map_gen = cv2.resize(map_gen, (map_ref.shape[1], map_ref.shape[0]))

    # Edge Overlap (IoU of edges)
    intersection = cv2.bitwise_and(map_ref, map_gen)
    union = cv2.bitwise_or(map_ref, map_gen)
    
    iou = np.sum(intersection > 0) / (np.sum(union > 0) + 1e-6)
    
    print(f"Crease Map IoU: {iou:.4f}")
    
    # Simple Pixel Difference as score proxy
    diff = cv2.absdiff(map_ref, map_gen)
    mse = np.mean(diff)
    print(f"Crease Map MSE: {mse:.4f}")
    
    if out_vis_path:
        h, w = map_ref.shape
        vis = np.zeros((h, w*3), dtype=np.uint8)
        vis[:, :w] = map_ref
        vis[:, w:w*2] = map_gen
        vis[:, w*2:] = diff
        cv2.imwrite(out_vis_path, vis)
        print(f"Saved visualization to {out_vis_path}")
        
    return iou #(higher is better)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_structure.py <ref> <gen> [out_vis]")
        sys.exit(1)
        
    out = sys.argv[3] if len(sys.argv) > 3 else "data/structure_comparison.png"
    compare_structure(sys.argv[1], sys.argv[2], out)
