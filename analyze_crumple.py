import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_crumple_sequence(start_frame, end_frame, frames_dir):
    roughness_scores = []
    frames = []
    
    print(f"Analyzing roughness from frame {start_frame} to {end_frame}...")
    
    for i in range(start_frame, end_frame + 1):
        fname = f"frame_{i:04d}.png"
        path = os.path.join(frames_dir, fname)
        if not os.path.exists(path):
            continue
            
        img = cv2.imread(path, 0)
        
        # Calculate Laplacian variance (standard measure for focus/texture/roughness)
        # Represents high-frequency components (edges, wrinkles)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # Calculate Entropy
        # Entropy is higher for chaotic/crumpled textures
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / hist.sum() # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        roughness_scores.append(laplacian_var)
        frames.append(i)
        
        print(f"Frame {i}: Laplacian Var (Roughness) = {laplacian_var:.2f}, Entropy = {entropy:.4f}")

    # Plot
    # We expect Frame 33 (Crumpled) to have high roughness, Frame 45 (Flat) to have low(er) roughness?
    # Actually, a flat poster with text has high edges too. 
    # But "Crumples" add *irregular* shading/edges.
    
    return frames, roughness_scores

if __name__ == "__main__":
    analyze_crumple_sequence(33, 44, "data/frames_ref")
