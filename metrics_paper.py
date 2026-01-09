import argparse
import os

import cv2
import numpy as np


def _read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def detect_pause_icon_mask(gray):
    """
    The reference frames include a bright pause icon overlay near the center.
    We detect and mask it so geometry/crease metrics are not dominated by it.
    """
    h, w = gray.shape[:2]
    _, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Keep only components near image center and small-ish
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


def crease_map(gray, ignore_mask=None):
    img = gray.copy()
    if ignore_mask is not None and ignore_mask.any():
        img[ignore_mask > 0] = 0

    inv = cv2.bitwise_not(img)
    adaptive = cv2.adaptiveThreshold(
        inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if ignore_mask is not None and ignore_mask.any():
        opened[ignore_mask > 0] = 0
    return opened


def iou_binary(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)


def ssim(gray_a, gray_b, valid_mask=None):
    a = gray_a.astype(np.float32) / 255.0
    b = gray_b.astype(np.float32) / 255.0

    # Gaussian window ~11x11, sigma~1.5 (common SSIM defaults)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("gen")
    ap.add_argument("--vis", default=None)
    ap.add_argument("--w_iou", type=float, default=0.6)
    ap.add_argument("--w_ssim", type=float, default=0.4)
    args = ap.parse_args()

    if not os.path.exists(args.ref):
        raise FileNotFoundError(args.ref)
    if not os.path.exists(args.gen):
        raise FileNotFoundError(args.gen)

    ref = _read_gray(args.ref)
    gen = _read_gray(args.gen)
    if ref.shape != gen.shape:
        gen = cv2.resize(gen, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)

    ignore = detect_pause_icon_mask(ref)

    # Detect Paper Mask (Foreground)
    # Reference and Gen might have different lighting/noise, but paper should be bright (>60).
    _, paper_mask_ref = cv2.threshold(ref, 60, 255, cv2.THRESH_BINARY)
    _, paper_mask_gen = cv2.threshold(gen, 60, 255, cv2.THRESH_BINARY)
    
    # Fill holes
    kernel = np.ones((5, 5), np.uint8)
    paper_mask_ref = cv2.morphologyEx(paper_mask_ref, cv2.MORPH_CLOSE, kernel)
    paper_mask_gen = cv2.morphologyEx(paper_mask_gen, cv2.MORPH_CLOSE, kernel)

    # Coverage Metric
    area_ref = paper_mask_ref.sum()
    area_gen = paper_mask_gen.sum()
    coverage = area_gen / (area_ref + 1e-9)
    if area_ref < 1000: # Safety for empty reference?
        coverage = 0.0
    
    # Combined mask for Creases: Ignore Pause Icon AND Background
    # We want creases ONLY on the paper.
    # mask_ref: 0 = ignore.
    # ignore (pause icon): 255 = ignore. -> valid = (ignore==0)
    
    # Valid region for Ref: PaperRef AND NOT PauseIcon
    valid_ref = cv2.bitwise_and(paper_mask_ref, (255 - ignore))
    
    # Valid region for Gen: PaperGen (Gen has no pause icon usually)
    valid_gen = paper_mask_gen

    # But we want to compare creases in the SAME visual region?
    # No, we want to see if the generated paper has the right creases.
    # If the generated paper is in the wrong place, IoU will be low naturally?
    # Actually, IoU is pixel-wise. If shapes don't overlap, IoU is 0.
    
    # Let's pass the INVERSE of valid as 'ignore_mask' to crease_map
    # crease_map takes 'ignore_mask' where >0 means ignore.
    ignore_ref = cv2.bitwise_not(valid_ref)
    ignore_gen = cv2.bitwise_not(valid_gen)

    cm_ref = crease_map(ref, ignore_mask=ignore_ref)
    cm_gen = crease_map(gen, ignore_mask=ignore_gen)

    iou = iou_binary(cm_ref, cm_gen)
    s = ssim(ref, gen, valid_mask=valid_ref) # SSIM only on reference paper area? 
    # Or intersection? Let's use valid_ref for SSIM to see if we match the ref content.
    
    # Penalize if coverage is bad
    # If coverage < 0.5, score drops rapidly.
    coverage_factor = min(1.0, coverage / 0.8) # 0.8 coverage is enough for full score
    coverage_factor = coverage_factor ** 2 # quadratic penalty
    
    raw_score = args.w_iou * iou + args.w_ssim * s
    score = raw_score * coverage_factor

    print(f"Coverage: {coverage:.4f}")
    print(f"CreaseIoU: {iou:.4f}")
    print(f"SSIM: {s:.4f}")
    print(f"SCORE: {score:.6f}")

    if args.vis:
        ref_rgb = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        gen_rgb = cv2.cvtColor(gen, cv2.COLOR_GRAY2BGR)
        ref_rgb[ignore > 0] = (0, 0, 255)
        gen_rgb[ignore > 0] = (0, 0, 255)

        cm_ref_rgb = cv2.cvtColor(cm_ref, cv2.COLOR_GRAY2BGR)
        cm_gen_rgb = cv2.cvtColor(cm_gen, cv2.COLOR_GRAY2BGR)
        diff = cv2.absdiff(cm_ref, cm_gen)
        diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        
        # Resize diff or pad to match width
        # Top/Bot width = ref + gen width
        total_w = ref_rgb.shape[1] + gen_rgb.shape[1]
        diff_show = np.zeros((diff.shape[0], total_w, 3), dtype=np.uint8)
        # Center the diff image? Or scale it up? 
        # Let's scale it up to fill
        diff_show = cv2.resize(diff, (total_w, diff.shape[0]), interpolation=cv2.INTER_NEAREST)

        top = np.hstack([ref_rgb, gen_rgb])
        bot = np.hstack([cm_ref_rgb, cm_gen_rgb])
        out = np.vstack([top, bot, diff_show])
        cv2.imwrite(args.vis, out)


if __name__ == "__main__":
    main()
