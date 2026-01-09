
import cv2
import numpy as np
import os

def main():
    # Frame 45 is the flattest state, ideal for UV/Crease map.
    # Note: Frame 45 might still have some perspective distortion, 
    # but it's the best 2D approximation we have.
    input_path = "data/frames_ref/frame_0045.png"
    output_path = "data/crease_map_from_45.png"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Masking (Simple intensity or color threshold to isolate paper from background)
    # Assuming dark background.
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # 2. Enhance edges
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 3. Edge Detection (Canny)
    # Adjust thresholds to pick up creases but not noise
    edges = cv2.Canny(enhanced, 50, 150)
    
    # 4. Mask the edges
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # 5. Dilation to make creases thicker (easier for simulation to hit)
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # 6. Invert ? (White lines on Black background is good for "Activity Map")
    # Simulation logic: If map(u,v) > threshold -> This edge is a Crease.
    
    cv2.imwrite(output_path, edges_dilated)
    print(f"Saved crease map to {output_path}")
    
    # Also save a debug visualization overlay
    overlay = img.copy()
    overlay[edges_dilated > 0] = [0, 0, 255] # Red creases
    cv2.imwrite("data/crease_extraction_debug.png", overlay)

if __name__ == "__main__":
    main()
