import json
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowParams:
    finest_scale: int = 1
    patch_size: int = 8
    patch_stride: int = 4
    gd_iters: int = 25


@dataclass(frozen=True)
class AssetParams:
    uv_mode: str = "w_minus_1"  # or "half_pixel"
    alpha_feather_sigma: float = 1.0
    back_blur_sigma: float = 1.0
    light_scale: float = 3.0  # encode ratio as (ratio/light_scale) in 16-bit PNG
    light_blur_sigma: float = 0.6
    light_clip_min: float = 0.25
    light_clip_max: float = 2.5
    back_l_min: int = 225
    back_chroma_max: float = 14.0
    back_warp_diff_min: float = 18.0
    back_l_delta_min: float = 12.0


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


def _srgb_u8_to_linear_f32(u8: np.ndarray) -> np.ndarray:
    x = u8.astype(np.float32) / 255.0
    a = 0.055
    out = np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4)
    return out.astype(np.float32)


def _linear_luma_from_bgr_u8(bgr_u8: np.ndarray) -> np.ndarray:
    b = _srgb_u8_to_linear_f32(bgr_u8[:, :, 0])
    g = _srgb_u8_to_linear_f32(bgr_u8[:, :, 1])
    r = _srgb_u8_to_linear_f32(bgr_u8[:, :, 2])
    return (0.0722 * b + 0.7152 * g + 0.2126 * r).astype(np.float32)


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


def _grid_from_flow(flow_target_to_source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = flow_target_to_source.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src_x = grid_x + flow_target_to_source[:, :, 0].astype(np.float32)
    src_y = grid_y + flow_target_to_source[:, :, 1].astype(np.float32)
    return src_x, src_y


def warp_source_to_target(source_bgr: np.ndarray, flow_target_to_source: np.ndarray) -> np.ndarray:
    h, w = flow_target_to_source.shape[:2]
    src_x, src_y = _grid_from_flow(flow_target_to_source)
    src_x = np.clip(src_x, 0.0, float(w - 1)).astype(np.float32)
    src_y = np.clip(src_y, 0.0, float(h - 1)).astype(np.float32)
    return cv2.remap(source_bgr, src_x, src_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def flow_to_uv16(flow_target_to_source: np.ndarray, uv_mode: str) -> np.ndarray:
    """
    Convert flow to a 16-bit RGB image where:
      R = U in [0..1]
      G = V in [0..1]
      B = 0
    and UV encodes *source* sampling coordinates.
    """
    h, w = flow_target_to_source.shape[:2]
    src_x, src_y = _grid_from_flow(flow_target_to_source)
    src_x = np.clip(src_x, 0.0, float(w - 1))
    src_y = np.clip(src_y, 0.0, float(h - 1))

    if uv_mode == "half_pixel":
        u = (src_x + 0.5) / float(w)
        v = 1.0 - ((src_y + 0.5) / float(h))
        u = np.clip(u, 0.5 / float(w), 1.0 - 0.5 / float(w))
        v = np.clip(v, 0.5 / float(h), 1.0 - 0.5 / float(h))
    elif uv_mode == "w_minus_1":
        u = src_x / float(w - 1)
        v = 1.0 - (src_y / float(h - 1))  # Blender UV origin differs (V up)
    else:
        raise ValueError(f"Unknown uv_mode={uv_mode!r}")

    rgb = np.zeros((h, w, 3), dtype=np.uint16)
    rgb[:, :, 0] = (u * 65535.0 + 0.5).astype(np.uint16)
    rgb[:, :, 1] = (v * 65535.0 + 0.5).astype(np.uint16)
    rgb[:, :, 2] = 0
    return rgb


def _gauss_u8(mask_u8: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask_u8
    blurred = cv2.GaussianBlur(mask_u8, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def compute_light_ratio_u16(
    target_bgr: np.ndarray,
    warped_source_bgr: np.ndarray,
    paper_mask: np.ndarray,
    back_mask: np.ndarray,
    params: AssetParams,
    ignore_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Produce a multiplicative ratio map (target / warped_source) in (approx) linear space, with special handling for backside.
    Encoded as 16-bit PNG in [0..1] with scale params.light_scale so values >1 are representable.
    """
    eps = 1.0
    t_luma = _linear_luma_from_bgr_u8(target_bgr)
    w_luma = _linear_luma_from_bgr_u8(warped_source_bgr)

    ratio_front = (t_luma + (eps / 255.0)) / (w_luma + (eps / 255.0))
    ratio_back = t_luma  # white(1.0) * ratio_back ≈ target (in luma)
    ratio = np.where(back_mask > 0, ratio_back, ratio_front)
    ratio = np.where(paper_mask > 0, ratio, 1.0)
    if ignore_mask is not None and ignore_mask.any():
        ratio = np.where(ignore_mask > 0, 1.0, ratio)

    ratio = np.clip(ratio, float(params.light_clip_min), float(params.light_clip_max))
    if params.light_blur_sigma > 0:
        ratio = cv2.GaussianBlur(ratio, (0, 0), float(params.light_blur_sigma), borderType=cv2.BORDER_REPLICATE)
        ratio = np.where(paper_mask > 0, ratio, 1.0)
        if ignore_mask is not None and ignore_mask.any():
            ratio = np.where(ignore_mask > 0, 1.0, ratio)
        ratio = np.clip(ratio, float(params.light_clip_min), float(params.light_clip_max))

    encoded = np.clip(ratio / float(params.light_scale), 0.0, 1.0)
    u16 = (encoded * 65535.0 + 0.5).astype(np.uint16)
    rgb = np.zeros((u16.shape[0], u16.shape[1], 3), dtype=np.uint16)
    rgb[:, :, 0] = u16
    rgb[:, :, 1] = u16
    rgb[:, :, 2] = u16
    return rgb


def backside_mask_from_warp(
    target_bgr: np.ndarray,
    warped_source_bgr: np.ndarray,
    paper_mask: np.ndarray,
    params: AssetParams,
) -> np.ndarray:
    """
    Backside in the reference is mostly 'white paper' that cannot be explained by warping the source front texture.
    Detect it by combining whiteness (L/chroma) + large difference from warped source.
    """
    lab_t = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)
    lab_w = cv2.cvtColor(warped_source_bgr, cv2.COLOR_BGR2LAB)
    l = lab_t[:, :, 0].astype(np.float32)
    w_l = lab_w[:, :, 0].astype(np.float32)
    a = lab_t[:, :, 1].astype(np.float32) - 128.0
    bb = lab_t[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + bb * bb)

    diff = cv2.absdiff(target_bgr, warped_source_bgr)
    diff_gray = (0.114 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.299 * diff[:, :, 2]).astype(np.float32)

    m = (
        (paper_mask > 0)
        & (l >= float(params.back_l_min))
        & (chroma <= float(params.back_chroma_max))
        & (diff_gray >= float(params.back_warp_diff_min))
        & ((l - w_l) >= float(params.back_l_delta_min))
    )
    out = (m.astype(np.uint8) * 255)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    return out


def extract_flow_assets(
    ref_dir: str,
    out_assets_dir: str,
    frame_start: int,
    frame_end: int,
    source_frame: int,
    flow_params: FlowParams,
    asset_params: AssetParams,
) -> None:
    ensure_dir(out_assets_dir)
    uv_dir = os.path.join(out_assets_dir, "uv")
    alpha_dir = os.path.join(out_assets_dir, "alpha")
    back_dir = os.path.join(out_assets_dir, "backside")
    light_dir = os.path.join(out_assets_dir, "light")
    ensure_dir(uv_dir)
    ensure_dir(alpha_dir)
    ensure_dir(back_dir)
    ensure_dir(light_dir)

    source = read_bgr(os.path.join(ref_dir, f"frame_{source_frame:04d}.png"))
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    source_paper = paper_mask_from_frame(source)
    paper_med = int(np.median(source_gray[source_paper > 0])) if (source_paper > 0).any() else int(np.median(source_gray))
    source_gray_masked = source_gray.copy()
    source_gray_masked[source_paper == 0] = paper_med

    for idx in range(frame_start, frame_end + 1):
        target = read_bgr(os.path.join(ref_dir, f"frame_{idx:04d}.png"))
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        paper = paper_mask_from_frame(target)
        t_med = int(np.median(target_gray[paper > 0])) if (paper > 0).any() else int(np.median(target_gray))
        target_gray_masked = target_gray.copy()
        target_gray_masked[paper == 0] = t_med
        ignore = detect_pause_icon_mask(target_gray)
        flow = compute_flow_target_to_source(target_gray_masked, source_gray_masked, ignore_mask=ignore, params=flow_params)

        uv16 = flow_to_uv16(flow, uv_mode=asset_params.uv_mode)
        cv2.imwrite(os.path.join(uv_dir, f"uv_{idx:04d}.png"), uv16)

        warped_src = warp_source_to_target(source, flow)

        # Backside: prefer warp-inconsistency-based detection; fall back to "white" heuristic when needed.
        back = backside_mask_from_warp(target, warped_src, paper_mask=paper, params=asset_params)
        if int(back.max()) == 0:
            back = white_backside_mask(target, paper)
        if ignore.any():
            back[ignore > 0] = 0
        back = _gauss_u8(back, sigma=float(asset_params.back_blur_sigma))
        alpha = _gauss_u8(paper, sigma=float(asset_params.alpha_feather_sigma))

        light_u16 = compute_light_ratio_u16(
            target,
            warped_src,
            paper_mask=paper,
            back_mask=back,
            params=asset_params,
            ignore_mask=ignore,
        )

        # Write as 3-channel PNGs to avoid single-channel handling quirks in Blender 2.82.
        cv2.imwrite(os.path.join(alpha_dir, f"alpha_{idx:04d}.png"), cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(os.path.join(back_dir, f"back_{idx:04d}.png"), cv2.cvtColor(back, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(os.path.join(light_dir, f"light_{idx:04d}.png"), light_u16)

    meta = {
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "source_frame": int(source_frame),
        "flow_params": asdict(flow_params),
        "asset_params": asdict(asset_params),
    }
    with open(os.path.join(out_assets_dir, "assets_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
