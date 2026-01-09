import cv2
import numpy as np

def detect_creases(image_path, output_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"Failed to load {image_path}")
        return

    # Ridge detection using Frangi or simple Hessian-based approach?
    # Simpler: Meijering or just inverted edges.
    # Paper creases light up as high intensity or shadow lines.
    
    # Shadows (dark lines) are the valleys. Highlights are peaks.
    # Let's verify valleys (dark creases).
    
    # Invert so valleys are bright
    inv_img = cv2.bitwise_not(img)
    
    # Adaptive threshold to isolate local dark lines
    adaptive = cv2.adaptiveThreshold(inv_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Skeletonize to see the "Map"
    # This roughly gives the "Voronoi" structure of the folds.
    
    # Also calculate orientation of these lines using Gradient structure tensor
    # Sobel
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    mag, ang = cv2.cartToPolar(gx, gy)
    
    # Filter only strong edges
    mask = mag > np.mean(mag) + np.std(mag)
    
    # Average angle of strong creases
    mean_angle = np.mean(ang[mask]) * 180 / np.pi
    
    print(f"Dominant Crease Angle: {mean_angle:.2f} degrees")
    # 0 deg = Horizontal vertical gradient -> Vertical Edge? 
    # OpenCV angles: 0 is horizontal x-axis?
    
    # Save visualization
    cv2.imwrite(output_path, adaptive)
    print(f"Crease map saved to {output_path}")

if __name__ == "__main__":
    detect_creases("data/frames_ref/frame_0040.png", "data/analysis_crease_map_40.png")
