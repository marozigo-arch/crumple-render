import cv2
import numpy as np

def extract_poster(source_path, output_path):
    img = cv2.imread(source_path)
    if img is None:
        print(f"Failed to load {source_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edged = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest rectangle-ish contour
    largest_cnt = None
    max_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000: # Filter small noise
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # If it has 4 points, it's our best candidate for a poster
            if len(approx) == 4 and area > max_area:
                max_area = area
                largest_cnt = approx

    if largest_cnt is not None:
        print("Found poster contour.")
        x, y, w, h = cv2.boundingRect(largest_cnt)
        
        # Crop
        crop = img[y:y+h, x:x+w]
        
        # Save
        cv2.imwrite(output_path, crop)
        print(f"Extracted texture saved to {output_path}")
    else:
        print("Could not detect clear rectangular poster. Saving center crop as fallback.")
        h, w = img.shape[:2]
        # Crop central 80%
        y1 = int(h * 0.1)
        y2 = int(h * 0.9)
        x1 = int(w * 0.1)
        x2 = int(w * 0.9)
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(output_path, crop)
        print(f"Fallback texture saved to {output_path}")

if __name__ == "__main__":
    extract_poster("data/frames_ref/frame_0045.png", "data/extracted_texture_from_45.png")
