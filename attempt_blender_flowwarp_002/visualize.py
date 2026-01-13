import os
from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def make_compare_grid(ref_dir: str, gen_dir: str, frame_start: int, frame_end: int, out_path: str, scale: float = 0.35) -> None:
    ref_dir = str(ref_dir)
    gen_dir = str(gen_dir)
    frames = list(range(frame_start, frame_end + 1))

    ref_imgs = [read_bgr(os.path.join(ref_dir, f"frame_{f:04d}.png")) for f in frames]
    gen_imgs = [read_bgr(os.path.join(gen_dir, f"frame_{f:04d}.png")) for f in frames]

    def resize(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ref_row = np.concatenate([resize(i) for i in ref_imgs], axis=1)
    gen_row = np.concatenate([resize(i) for i in gen_imgs], axis=1)
    grid = np.concatenate([ref_row, gen_row], axis=0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, grid)

