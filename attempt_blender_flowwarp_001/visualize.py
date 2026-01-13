import os

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def make_compare_grid(
    ref_dir: str,
    gen_dir: str,
    frame_start: int,
    frame_end: int,
    out_path: str,
    scale: float = 0.35,
) -> None:
    refs = []
    gens = []
    for idx in range(frame_start, frame_end + 1):
        r = read_bgr(os.path.join(ref_dir, f"frame_{idx:04d}.png"))
        g = read_bgr(os.path.join(gen_dir, f"frame_{idx:04d}.png"))
        if r.shape != g.shape:
            g = cv2.resize(g, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
        cv2.putText(r, f"{idx}", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(g, f"{idx}", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        refs.append(r)
        gens.append(g)
    top = np.hstack(refs)
    bot = np.hstack(gens)
    grid = np.vstack([top, bot])
    if scale != 1.0:
        grid = cv2.resize(grid, (int(grid.shape[1] * scale), int(grid.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, grid)

