import cv2
import numpy as np

def analyze_axis(frame_start, frame_end):
    img1 = cv2.imread(frame_start, 0) # grayscale
    img2 = cv2.imread(frame_end, 0)
    
    if img1 is None or img2 is None:
        print("Frames not found")
        return

    # Abs difference
    diff = cv2.absdiff(img1, img2)
    
    # Threshold to remove subtle lighting noise
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Count non-zero pixels per row (Y projection) and per column (X projection)
    y_proj = np.sum(thresh, axis=1) # Sum along row (how much changed in that Y line)
    x_proj = np.sum(thresh, axis=0) # Sum along col (how much changed in that X line)
    
    # Standard deviation tells us how "localized" the change is
    # If bending is along Y axis (like a scroll rolling down), changes might be uniform across X? 
    # Actually simpler:
    # If it bends along X axis (vertical crease), the "structure" of change varies along X?
    
    # Let's count "active" lines
    active_rows = np.count_nonzero(y_proj)
    active_cols = np.count_nonzero(x_proj)
    
    print(f"Active Rows (Y): {active_rows}")
    print(f"Active Cols (X): {active_cols}")
    
    # Variance of the projection might be better
    # If the sheet unrolls vertically (top to bottom), the "action" moves along Y.
    # So we'd expect significant variance in the X-projection? No.
    
    # Let's look at the gradient of the difference
    sobelx = cv2.Sobel(thresh, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(thresh, cv2.CV_64F, 0, 1, ksize=5)
    
    mag_x = np.sum(np.abs(sobelx))
    mag_y = np.sum(np.abs(sobely))
    
    print(f"Gradients X magnitude: {mag_x:.2f}")
    print(f"Gradients Y magnitude: {mag_y:.2f}")
    
    if mag_x > mag_y:
        print("Prediction: Vertical edges dominant -> Likely horizontal movement/bending (Axis X?)")
    else:
        print("Prediction: Horizontal edges dominant -> Likely vertical movement/bending (Axis Y?)")

if __name__ == "__main__":
    analyze_axis("data/frames_ref/frame_0033.png", "data/frames_ref/frame_0045.png")
