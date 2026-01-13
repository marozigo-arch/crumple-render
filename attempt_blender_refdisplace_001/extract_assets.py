import os
from dataclasses import dataclass

import cv2
import numpy as np

from metrics import detect_pause_icon_mask, paper_mask_from_ref


@dataclass(frozen=True)
class DisplaceParams:
    sigma_small: float = 2.5
    sigma_large: float = 24.0
    gamma: float = 1.0
    clip: float = 2.5  # clip std devs


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def write_u16_png(path: str, img01: np.ndarray) -> None:
    img01 = np.clip(img01, 0.0, 1.0)
    u16 = (img01 * 65535.0 + 0.5).astype(np.uint16)
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, u16)


def extract_poster_from_ref(ref_bgr: np.ndarray) -> np.ndarray:
    """
    Approximate "poster texture" from a reference frame by masking the paper and cropping its bbox.
    This is only for debug similarity checks.
    """
    mask = paper_mask_from_ref(ref_bgr)
    er = cv2.erode(mask, np.ones((25, 25), np.uint8), iterations=1)
    ys, xs = np.where(er > 0)
    if len(xs) < 10:
        ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return ref_bgr

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = ref_bgr[y0 : y1 + 1, x0 : x1 + 1].copy()

    m_crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    # Outside paper -> white
    crop[m_crop == 0] = (255, 255, 255)
    return crop


def compute_displacement_01(ref_bgr: np.ndarray, params: DisplaceParams) -> np.ndarray:
    """
    Build a per-pixel displacement texture in [0..1], where 0.5 is neutral.
    Uses a ratio of blurred luminance to emphasize folds while suppressing printed content.
    """
    gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    pause = detect_pause_icon_mask(gray)
    mask = paper_mask_from_ref(ref_bgr)

    g = gray.astype(np.float32) / 255.0
    if pause.any():
        med = float(np.median(g[mask > 0])) if (mask > 0).any() else float(np.median(g))
        g[pause > 0] = med

    small = cv2.GaussianBlur(g, (0, 0), float(params.sigma_small))
    large = cv2.GaussianBlur(g, (0, 0), float(params.sigma_large))
    ratio = (small + 1e-6) / (large + 1e-6)

    # Standardize within paper region, then map to 0..1 around 0.5.
    region = ratio[mask > 0]
    mu = float(np.mean(region)) if region.size else float(np.mean(ratio))
    sd = float(np.std(region)) if region.size else float(np.std(ratio))
    sd = max(sd, 1e-6)
    z = (ratio - mu) / sd
    z = np.clip(z, -params.clip, params.clip) / params.clip  # [-1..1]
    if params.gamma != 1.0:
        # preserve sign, apply gamma to magnitude
        z = np.sign(z) * (np.abs(z) ** float(params.gamma))

    disp = 0.5 + 0.5 * z
    # Outside paper -> neutral
    disp = np.where(mask > 0, disp, 0.5)
    return disp.astype(np.float32)


def extract_displacement_sequence(
    ref_dir: str,
    out_dir: str,
    frame_start: int,
    frame_end: int,
    params: DisplaceParams,
) -> None:
    ensure_dir(out_dir)
    for idx in range(frame_start, frame_end + 1):
        ref = read_bgr(os.path.join(ref_dir, f"frame_{idx:04d}.png"))
        disp = compute_displacement_01(ref, params=params)
        write_u16_png(os.path.join(out_dir, f"disp_{idx:04d}.png"), disp)
