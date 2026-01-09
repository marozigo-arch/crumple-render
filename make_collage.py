import cv2
import numpy as np
import os

def main():
    base_dir = "data/sequence_exp13"
    frames = [38, 39, 40, 41, 42]
    images = []

    for f in frames:
        path = os.path.join(base_dir, f"frame_{f:04d}.png")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        
        img = cv2.imread(path)
        # Add frame number text
        cv2.putText(img, f"Frame {f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        images.append(img)

    if not images:
        print("No images found.")
        return

    # Stack horizontally
    collage = np.hstack(images)
    
    # Resize if too wide (optional)
    # 864 * 5 = 4320 px wide. might be okay.
    
    out_path = os.path.join(base_dir, "dynamics_collage.png")
    cv2.imwrite(out_path, collage)
    print(f"Saved collage to {out_path}")

if __name__ == "__main__":
    main()
