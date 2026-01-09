import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_deformation(frame_start, frame_mid, frame_end):
    # Load frames
    img_start = cv2.imread(frame_start, 0)
    img_mid = cv2.imread(frame_mid, 0)
    img_end = cv2.imread(frame_end, 0)
    
    if img_start is None or img_mid is None or img_end is None:
        print("Frames not found")
        return

    # 1. Optical Flow (Farneback) to see the vector field
    flow = cv2.calcOpticalFlowFarneback(img_mid, img_end, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Analyze flow vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Check linearity of flow along axes
    # If it's a linear unfold (scale), flow magnitude should change linearly across the image.
    # If it's a curl, it will be non-linear.
    
    # Average flow magnitude per column (assuming horizontal action based on previous check)
    # and per row.
    mag_x_profile = np.mean(mag, axis=0) # Average mag for each X coordinate
    mag_y_profile = np.mean(mag, axis=1) # Average mag for each Y coordinate
    
    # Polynomial fit to check curvature
    x = np.arange(len(mag_x_profile))
    z_x = np.polyfit(x, mag_x_profile, 2) # Quadratic fit
    p_x = np.poly1d(z_x)
    
    y = np.arange(len(mag_y_profile))
    z_y = np.polyfit(y, mag_y_profile, 2)
    p_y = np.poly1d(z_y)
    
    # Curvature score (coefficient of x^2)
    curve_score_x = abs(z_x[0]) * 1000
    curve_score_y = abs(z_y[0]) * 1000
    
    print(f"Curvature Score X-profile: {curve_score_x:.4f}")
    print(f"Curvature Score Y-profile: {curve_score_y:.4f}")
    
    # Visualizing flow direction consistency
    # If std dev of angle is low, movement is unidirectional (Planar Scale/Slide)
    # If high, it might be rotation or complex warp.
    ang_deg = ang * 180 / np.pi / 2
    active_flow_mask = mag > 1.0 # Only consider moving pixels
    if np.any(active_flow_mask):
        ang_std = np.std(ang_deg[active_flow_mask])
        print(f"Flow Angle Std Dev: {ang_std:.2f}")
    else:
        print("No significant movement detected")

    # Determine Effect Type
    print("\n--- HYPOTHESIS ---")
    if curve_score_x < 0.1 and curve_score_y < 0.1:
        print("Flow profile is Linear. Effect: Planar Scaling / Sliding / Unmasking.")
    elif curve_score_x > curve_score_y:
        print("Flow profile is Non-Linear along X. Effect: Cylindrical Bend / Page Curl (Horizontal Axis).")
    else:
        print("Flow profile is Non-Linear along Y. Effect: Cylindrical Bend / Page Curl (Vertical Axis).")
        
    # Additional Check: Edge Distortion
    # Detect edges in Start and End
    edges_start = cv2.Canny(img_start, 50, 150)
    edges_end = cv2.Canny(img_end, 50, 150)
    
    # Find lines
    lines_start = cv2.HoughLinesP(edges_start, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    lines_end = cv2.HoughLinesP(edges_end, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    print(f"Detected straight lines: Start={len(lines_start) if lines_start is not None else 0}, End={len(lines_end) if lines_end is not None else 0}")
    # If straight lines are preserved, it's likely affine/projective. If not (and detected count drops or segments fragment), it's warp.

if __name__ == "__main__":
    # Analyzing the "Unfolding" sequence 33 -> 38 -> 45
    print("Analyzing Ref Frames 38 -> 45 (Unfolding phase)...")
    analyze_deformation(
        "data/frames_ref/frame_0033.png",
        "data/frames_ref/frame_0038.png", 
        "data/frames_ref/frame_0045.png"
    )
