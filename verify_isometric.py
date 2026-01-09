import cv2
import numpy as np
import sys

def calculate_stretch_score(flat_texture_path, distorted_image_path):
    # 1. Load images
    img1 = cv2.imread(flat_texture_path, 0) # QueryImage (The flat texture)
    img2 = cv2.imread(distorted_image_path, 0) # TrainImage ( The render)
    
    if img1 is None or img2 is None:
        return 999.0 # Error

    # 2. Inspect Features (ORB is fast and effective)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return 999.0

    # 3. Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort by distance
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Take top matches
    good_matches = matches[:50]
    
    if len(good_matches) < 10:
        return 999.0 # Not enough matches to judge
        
    # 4. Analyze Geometric Distortion
    # Get coordinates
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    
    # Calculate pairwise distances in Source (Flat) and Destination (Crumpled)
    # Ideally, for PAPER, distances should only *decrease* (due to folding/perspective) 
    # or stay same. They should NOT increase significantly (stretching).
    
    # Let's verify scaling factor between pairs
    ratios = []
    
    # Sample random pairs to be faster
    import random
    for _ in range(100):
        idx_a, idx_b = random.sample(range(len(src_pts)), 2)
        
        p1_src = src_pts[idx_a][0]
        p2_src = src_pts[idx_b][0]
        dist_src = np.linalg.norm(p1_src - p2_src)
        
        p1_dst = dst_pts[idx_a][0]
        p2_dst = dst_pts[idx_b][0]
        dist_dst = np.linalg.norm(p1_dst - p2_dst)
        
        if dist_src > 10.0: # Ignore tiny base distances
            ratio = dist_dst / dist_src
            ratios.append(ratio)
            
    if not ratios:
        return 999.0
        
    ratios = np.array(ratios)
    
    # Any ratio > 1.1 implies significant stretching (10% growth)
    # We want ratios to be <= 1.0 ideally (foreshortening is fine)
    
    stretch_violation = np.mean(ratios[ratios > 1.05]) if np.any(ratios > 1.05) else 0.0
    
    # If stretch_violation is 0, it means no significant stretching detected.
    # We also want to penalize if the image is too small (collapsed).
    # avg_ratio = np.mean(ratios)
    
    print(f"Stretch Violation Score: {stretch_violation:.4f}")
    return stretch_violation

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_isometric.py <flat_tex> <render>")
        sys.exit(1)
        
    score = calculate_stretch_score(sys.argv[1], sys.argv[2])
    print(f"FINAL_SCORE:{score}")
