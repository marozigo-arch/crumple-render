import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowParams:
    preset: int = 1  # 0=FAST,1=MEDIUM,2=ULTRAFAST (OpenCV constants differ), we use MEDIUM create
    finest_scale: int = 1
    patch_size: int = 8
    patch_stride: int = 4
    gd_iters: int = 25


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
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


def paper_mask_from_frame(bgr: np.ndarray) -> np.ndarray:
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

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return bw
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == best).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8), iterations=2)
    mask = fill_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
    return mask


def white_backside_mask(bgr: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.int16)
    a = lab[:, :, 1].astype(np.int16) - 128
    bb = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt(a * a + bb * bb)
    white = (l > 205) & (chroma < 16) & (paper_mask > 0)
    m = (white.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    return m


def compute_flow_target_to_source(target_gray: np.ndarray, source_gray: np.ndarray, ignore_mask: np.ndarray, params: FlowParams) -> np.ndarray:
    tg = target_gray.copy()
    sg = source_gray.copy()
    if ignore_mask is not None and ignore_mask.any():
        med_t = int(np.median(tg))
        med_s = int(np.median(sg))
        tg[ignore_mask > 0] = med_t
        sg[ignore_mask > 0] = med_s

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(int(params.finest_scale))
    dis.setPatchSize(int(params.patch_size))
    dis.setPatchStride(int(params.patch_stride))
    dis.setGradientDescentIterations(int(params.gd_iters))
    return dis.calc(tg, sg, None)  # target -> source


def flow_to_uv16(flow_target_to_source: np.ndarray) -> np.ndarray:
    """
    Convert flow to a 16-bit RGB image where:
      R = U in [0..1]
      G = V in [0..1]
      B = 0
    where UV encodes source sampling coordinates.
    """
    h, w = flow_target_to_source.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src_x = grid_x + flow_target_to_source[:, :, 0].astype(np.float32)
    src_y = grid_y + flow_target_to_source[:, :, 1].astype(np.float32)

    src_x = np.clip(src_x, 0.0, float(w - 1))
    src_y = np.clip(src_y, 0.0, float(h - 1))
    u = src_x / float(w - 1)
    v = 1.0 - (src_y / float(h - 1))  # Blender UV origin differs (V up)

    rgb = np.zeros((h, w, 3), dtype=np.uint16)
    rgb[:, :, 0] = (u * 65535.0 + 0.5).astype(np.uint16)
    rgb[:, :, 1] = (v * 65535.0 + 0.5).astype(np.uint16)
    rgb[:, :, 2] = 0
    return rgb


def extract_flow_assets(
    ref_dir: str,
    out_assets_dir: str,
    frame_start: int,
    frame_end: int,
    source_frame: int,
    params: FlowParams,
) -> None:
    ensure_dir(out_assets_dir)
    uv_dir = os.path.join(out_assets_dir, "uv")
    mask_dir = os.path.join(out_assets_dir, "mask")
    back_dir = os.path.join(out_assets_dir, "backside")
    ensure_dir(uv_dir)
    ensure_dir(mask_dir)
    ensure_dir(back_dir)

    source = read_bgr(os.path.join(ref_dir, f"frame_{source_frame:04d}.png"))
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    for idx in range(frame_start, frame_end + 1):
        target = read_bgr(os.path.join(ref_dir, f"frame_{idx:04d}.png"))
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        ignore = detect_pause_icon_mask(target_gray)
        flow = compute_flow_target_to_source(target_gray, source_gray, ignore_mask=ignore, params=params)

        uv16 = flow_to_uv16(flow)
        cv2.imwrite(os.path.join(uv_dir, f"uv_{idx:04d}.png"), uv16)

        paper = paper_mask_from_frame(target)
        back = white_backside_mask(target, paper)
        # Write as 3-channel PNG to avoid single-channel handling quirks in Blender 2.82.
        cv2.imwrite(os.path.join(mask_dir, f"mask_{idx:04d}.png"), cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(os.path.join(back_dir, f"back_{idx:04d}.png"), cv2.cvtColor(back, cv2.COLOR_GRAY2BGR))
