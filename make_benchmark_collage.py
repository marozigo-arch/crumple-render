
import cv2
import numpy as np
import os

def main():
    base_dir = "data/benchmarks"
    files = ["method_A_cloth.png", "method_D_pickplace.png", "method_B_doc3d.png", "method_F_voronoi.png", "method_E_creases.png", "method_G_mathmap.png"]
    labels = ["A: Cloth", "D: Cloth (P&D)", "B: Doc3D (GT)", "F: Voronoi", "E: Map (Static)", "G: Map + Math"]
    images = []

    for f, label in zip(files, labels):
        path = os.path.join(base_dir, f)
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        
        img = cv2.imread(path)
        # Add label
        cv2.putText(img, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        images.append(img)

    if not images:
        return

    # Resize to same height just in case
    h = min(img.shape[0] for img in images)
    images = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) for img in images]
    
    collage = np.hstack(images)
    out_path = os.path.join(base_dir, "comparison_frame42.png")
    cv2.imwrite(out_path, collage)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
