import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MetricsRow:
    frame: int
    percent_diff: float
    ssim: float


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def detect_pause_icon_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    _, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cx0, cy0 = w * 0.5, h * 0.5
    keep = np.zeros_like(gray, dtype=np.uint8)

    for idx in range(1, num):
        x, y, ww, hh, area = stats[idx]
        cx, cy = centroids[idx]
        if area < 20:
            continue
        if area > (w * h) * 0.01:
            continue
        if abs(cx - cx0) > w * 0.18 or abs(cy - cy0) > h * 0.18:
            continue
        keep[labels == idx] = 255

    if keep.max() == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    keep = cv2.dilate(keep, np.ones((9, 9), np.uint8), iterations=1)
    return keep


def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    inv = 255 - m
    h, w = inv.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    inv2 = inv.copy()
    cv2.floodFill(inv2, ff_mask, (0, 0), 0)
    holes = (inv2 > 0).astype(np.uint8) * 255
    return cv2.bitwise_or(m, holes)


def paper_mask_from_ref(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    ignore = detect_pause_icon_mask(l)
    l2 = l.copy()
    if ignore.any():
        l2[ignore > 0] = int(np.median(l2))

    h, w = l2.shape[:2]
    b = max(8, int(round(min(h, w) * 0.012)))
    border = np.concatenate([l2[:b, :].ravel(), l2[-b:, :].ravel(), l2[:, :b].ravel(), l2[:, -b:].ravel()])
    bg_med = float(np.median(border))
    thr = int(min(255.0, bg_med + 22.0))
    bw = ((l2 > thr).astype(np.uint8) * 255)

    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2)
    bw = fill_holes(bw)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return bw
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == best).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8), iterations=2)
    mask = fill_holes(mask)
    return mask


def percent_diff(ref_bgr: np.ndarray, gen_bgr: np.ndarray, valid_mask: np.ndarray) -> float:
    if ref_bgr.shape != gen_bgr.shape:
        gen_bgr = cv2.resize(gen_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    m = (valid_mask > 0).astype(np.float32)[:, :, None]
    diff = np.abs(ref_bgr.astype(np.float32) - gen_bgr.astype(np.float32)) * m
    denom = (255.0 * 3.0) * (m.sum() + 1e-9)
    return float(100.0 * diff.sum() / denom)


def ssim_gray(gray_a: np.ndarray, gray_b: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    a = gray_a.astype(np.float32) / 255.0
    b = gray_b.astype(np.float32) / 255.0
    mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu12
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    num = (2 * mu12 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / (den + 1e-12)
    if valid_mask is not None and valid_mask.any():
        m = valid_mask.astype(bool)
        return float(ssim_map[m].mean())
    return float(ssim_map.mean())


def evaluate_pair(ref_path: str, gen_path: str) -> MetricsRow:
    ref = read_bgr(ref_path)
    gen = read_bgr(gen_path)
    mask = paper_mask_from_ref(ref)
    pause = detect_pause_icon_mask(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
    valid = cv2.bitwise_and(mask, cv2.bitwise_not(pause))

    pd = percent_diff(ref, gen, valid_mask=valid)
    s = ssim_gray(
        cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY),
        valid_mask=valid,
    )
    frame = int(os.path.basename(ref_path).split("_")[1].split(".")[0])
    return MetricsRow(frame=frame, percent_diff=pd, ssim=s)


def write_csv(rows: list[MetricsRow], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "percent_diff", "ssim"])
        w.writeheader()
        for r in rows:
            w.writerow({"frame": r.frame, "percent_diff": f"{r.percent_diff:.4f}", "ssim": f"{r.ssim:.6f}"})


def summarize(rows: list[MetricsRow]) -> dict:
    p = [r.percent_diff for r in rows]
    s = [r.ssim for r in rows]
    return {
        "frames": len(rows),
        "percent_diff_avg": float(np.mean(p)) if p else 1e9,
        "percent_diff_max": float(np.max(p)) if p else 1e9,
        "ssim_avg": float(np.mean(s)) if s else 0.0,
        "ssim_min": float(np.min(s)) if s else 0.0,
    }

