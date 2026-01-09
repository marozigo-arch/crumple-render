import cv2
import os
import numpy as np

def analyze_video(frames_dir):
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frames:
        print("No frames found.")
        return

    print(f"Total frames: {len(frames)}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    height, width, _ = first_frame.shape
    print(f"Resolution: {width}x{height}")

    # Analyze motion/color
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    diffs = []
    
    for i, frame_name in enumerate(frames[1:]):
        frame_path = os.path.join(frames_dir, frame_name)
        img = cv2.imread(frame_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(prev_gray, gray)
        score = np.mean(diff)
        diffs.append(score)
        
        prev_gray = gray

    print(f"Average frame difference score: {np.mean(diffs):.2f}")
    print(f"Max frame difference: {np.max(diffs):.2f} at frame {np.argmax(diffs)}")
    
    # Simple scene detection
    threshold = np.mean(diffs) * 3
    scene_changes = [i for i, score in enumerate(diffs) if score > threshold]
    print(f"Potential scene changes at frames: {scene_changes}")

if __name__ == "__main__":
    analyze_video("data/frames_ref")
