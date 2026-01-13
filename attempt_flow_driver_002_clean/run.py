import argparse
import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int

    def indices(self):
        return list(range(self.start, self.end + 1))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def list_ref_frames(ref_dir: str) -> list[int]:
    frames = []
    for name in os.listdir(ref_dir):
        if not name.startswith("frame_") or not name.endswith(".png"):
            continue
        try:
            frames.append(int(name[len("frame_") : len("frame_") + 4]))
        except ValueError:
            continue
    frames.sort()
    return frames


def load_ref_frame(ref_dir: str, idx: int) -> np.ndarray:
    return read_bgr(os.path.join(ref_dir, f"frame_{idx:04d}.png"))


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
    filled = cv2.bitwise_or(m, holes)
    return filled


def paper_mask_from_frame(bgr: np.ndarray) -> np.ndarray:
    """
    Robust mask for the paper silhouette.

    Otsu tends to punch holes into dark creases (they get classified as background).
    We instead estimate the background brightness from the image border and threshold above it.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    ignore = detect_pause_icon_mask(l)
    l2 = l.copy()
    if ignore.any():
        l2[ignore > 0] = int(np.median(l2))

    h, w = l2.shape[:2]
    b = max(8, int(round(min(h, w) * 0.012)))
    border = np.concatenate(
        [
            l2[:b, :].ravel(),
            l2[-b:, :].ravel(),
            l2[:, :b].ravel(),
            l2[:, -b:].ravel(),
        ]
    )
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
    m = cv2.GaussianBlur(m, (21, 21), 0)
    return m


def fit_cover_to_rect(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img_bgr.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x0 = max(0, (nw - w) // 2)
    y0 = max(0, (nh - h) // 2)
    return resized[y0 : y0 + h, x0 : x0 + w].copy()


def prepare_source_canvas(poster_bgr: np.ndarray, source_ref_bgr: np.ndarray, source_paper_mask: np.ndarray) -> np.ndarray:
    h, w = source_ref_bgr.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    er = cv2.erode(source_paper_mask, np.ones((25, 25), np.uint8), iterations=1)
    ys, xs = np.where(er > 0)
    if len(xs) < 10:
        ys, xs = np.where(source_paper_mask > 0)
    if len(xs) < 10:
        return cv2.resize(poster_bgr, (w, h), interpolation=cv2.INTER_AREA)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    rect_w, rect_h = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
    fitted = fit_cover_to_rect(poster_bgr, rect_w, rect_h)
    canvas[y0 : y0 + rect_h, x0 : x0 + rect_w] = fitted
    return canvas


def compute_flow_target_to_source(target_gray: np.ndarray, source_gray: np.ndarray, ignore_mask: np.ndarray) -> np.ndarray:
    tg = target_gray.copy()
    sg = source_gray.copy()
    if ignore_mask is not None and ignore_mask.any():
        med_t = int(np.median(tg))
        med_s = int(np.median(sg))
        tg[ignore_mask > 0] = med_t
        sg[ignore_mask > 0] = med_s

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    dis.setPatchSize(8)
    dis.setPatchStride(4)
    dis.setGradientDescentIterations(25)
    return dis.calc(tg, sg, None)  # target -> source


def warp_with_flow(source_bgr: np.ndarray, flow_target_to_source: np.ndarray) -> np.ndarray:
    h, w = source_bgr.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow_target_to_source[:, :, 0].astype(np.float32)
    map_y = grid_y + flow_target_to_source[:, :, 1].astype(np.float32)
    return cv2.remap(source_bgr, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def extract_shading_map(
    target_ref_bgr: np.ndarray,
    paper_mask: np.ndarray,
    sigma_mid: float,
    sigma_low: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """
    Returns a multiplicative shading map in [0.65..1.5] with most printed/albedo texture suppressed.
    """
    lab = cv2.cvtColor(target_ref_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32) / 255.0

    ignore = detect_pause_icon_mask((lab[:, :, 0]).astype(np.uint8))
    l2 = l.copy()
    if ignore.any():
        med = float(np.median(l2[paper_mask > 0])) if (paper_mask > 0).any() else float(np.median(l2))
        l2[ignore > 0] = med

    # Use multi-scale smoothing to suppress print while keeping broad folds.
    mid = cv2.GaussianBlur(l2, (0, 0), float(sigma_mid))
    low = cv2.GaussianBlur(l2, (0, 0), float(sigma_low))
    shade = (mid + 1e-6) / (low + 1e-6)

    # Reduce influence outside paper.
    shade = np.where(paper_mask > 0, shade, 1.0)
    shade = np.clip(shade, float(clip_min), float(clip_max))
    return shade.astype(np.float32)


def apply_lighting(
    warped_bgr: np.ndarray,
    target_ref_bgr: np.ndarray,
    paper_mask: np.ndarray,
    mode: str,
    shade_sigma_mid: float,
    shade_sigma_low: float,
    shade_clip_min: float,
    shade_clip_max: float,
) -> np.ndarray:
    if mode == "none":
        return warped_bgr

    if mode == "target":
        poster_lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_ref_bgr, cv2.COLOR_BGR2LAB)
        out = poster_lab.copy()
        out[:, :, 0] = target_lab[:, :, 0]
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    if mode == "shading":
        shade = extract_shading_map(
            target_ref_bgr,
            paper_mask,
            sigma_mid=shade_sigma_mid,
            sigma_low=shade_sigma_low,
            clip_min=shade_clip_min,
            clip_max=shade_clip_max,
        )
        poster_lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
        pl = poster_lab[:, :, 0].astype(np.float32) / 255.0
        out = poster_lab.copy()
        out_l = np.clip(pl * shade, 0.0, 1.0) * 255.0
        out[:, :, 0] = out_l.astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    raise ValueError(f"Unknown l_mode: {mode}")


def render_backside(
    target_ref_bgr: np.ndarray,
    paper_mask: np.ndarray,
    shade_sigma_mid: float,
    shade_sigma_low: float,
    shade_clip_min: float,
    shade_clip_max: float,
) -> np.ndarray:
    shade = extract_shading_map(
        target_ref_bgr,
        paper_mask,
        sigma_mid=shade_sigma_mid,
        sigma_low=shade_sigma_low,
        clip_min=shade_clip_min,
        clip_max=shade_clip_max,
    )
    # Softer shading on white backside.
    shade2 = np.clip(shade**0.75, 0.78, 1.18)
    white = np.full((*paper_mask.shape[:2], 3), 255, dtype=np.uint8)
    lab = cv2.cvtColor(white, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = (255.0 * shade2).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def feather_alpha(mask_u8: np.ndarray, radius: int = 21) -> np.ndarray:
    m = (mask_u8.astype(np.float32) / 255.0)
    k = max(3, radius | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def create_background(target_ref_bgr: np.ndarray, paper_mask: np.ndarray, mode: str, solid_bgr: tuple[int, int, int]) -> np.ndarray:
    if mode == "solid":
        bg = np.full_like(target_ref_bgr, solid_bgr, dtype=np.uint8)
        return bg
    if mode == "ref":
        return target_ref_bgr.copy()
    if mode == "ref_blur":
        bg = cv2.GaussianBlur(target_ref_bgr, (0, 0), 9.0)
        return bg
    raise ValueError(f"Unknown bg_mode: {mode}")


def apply_drop_shadow(bg_bgr: np.ndarray, paper_mask: np.ndarray, strength: float = 0.35) -> np.ndarray:
    m = (paper_mask > 0).astype(np.uint8) * 255
    dil = cv2.dilate(m, np.ones((25, 25), np.uint8), iterations=1)
    shadow = cv2.GaussianBlur(dil, (0, 0), 18.0).astype(np.float32) / 255.0
    shadow = shadow[:, :, None]
    out = bg_bgr.astype(np.float32) * (1.0 - shadow * strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_paper_over_background(bg_bgr: np.ndarray, paper_bgr: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    alpha = feather_alpha(paper_mask, radius=31)[:, :, None]
    out = bg_bgr.astype(np.float32) * (1.0 - alpha) + paper_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def to_rgba(paper_bgr: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    alpha = feather_alpha(paper_mask, radius=31)
    a = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([paper_bgr, a])


def blend_front_back(front_bgr: np.ndarray, back_bgr: np.ndarray, backside_alpha_u8: np.ndarray) -> np.ndarray:
    a = (backside_alpha_u8.astype(np.float32) / 255.0)[:, :, None]
    out = front_bgr.astype(np.float32) * (1.0 - a) + back_bgr.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def percent_diff(ref_bgr: np.ndarray, gen_bgr: np.ndarray, valid_mask: np.ndarray) -> float:
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


def build_grid_image(top_frames: list[np.ndarray], bottom_frames: list[np.ndarray], scale: float) -> np.ndarray:
    top = np.hstack(top_frames)
    bot = np.hstack(bottom_frames)
    grid = np.vstack([top, bot])
    if scale != 1.0:
        grid = cv2.resize(grid, (int(grid.shape[1] * scale), int(grid.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return grid


def list_posters(afisha_dir: str) -> list[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = [f for f in os.listdir(afisha_dir) if f.lower().endswith(exts)]
    files.sort()
    return [os.path.join(afisha_dir, f) for f in files]


def parse_bgr(arg: str) -> tuple[int, int, int]:
    # Accept "r,g,b" or "#RRGGBB"
    s = arg.strip()
    if s.startswith("#") and len(s) == 7:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        return (b, g, r)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 3:
        r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
        return (b, g, r)
    raise ValueError("bg_color must be 'r,g,b' or '#RRGGBB'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", default="data/frames_ref")
    ap.add_argument("--afisha_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_start", type=int, default=33)
    ap.add_argument("--frame_end", type=int, default=44)
    ap.add_argument("--full_cycle", action="store_true", help="Use all frames found in ref_dir.")
    ap.add_argument("--source_frame", type=int, default=44)
    ap.add_argument("--l_mode", choices=["none", "shading", "target"], default="shading")
    ap.add_argument("--shade_sigma_mid", type=float, default=4.0)
    ap.add_argument("--shade_sigma_low", type=float, default=28.0)
    ap.add_argument("--shade_clip_min", type=float, default=0.65)
    ap.add_argument("--shade_clip_max", type=float, default=1.5)
    ap.add_argument("--bg_mode", choices=["solid", "ref", "ref_blur"], default="solid")
    ap.add_argument("--bg_color", default="#0b0b0b")
    ap.add_argument("--shadow", action="store_true", help="Add a soft drop shadow (only meaningful for solid bg).")
    ap.add_argument("--transparent", action="store_true", help="Write RGBA PNGs with alpha=paper mask (ignores bg_mode).")
    ap.add_argument("--poster", default=None)
    ap.add_argument("--poster_from_ref_frame", type=int, default=None)
    ap.add_argument("--grid_scale", type=float, default=0.5)
    ap.add_argument("--grid_range_start", type=int, default=33)
    ap.add_argument("--grid_range_end", type=int, default=44)
    ap.add_argument("--no_grid", action="store_true")
    ap.add_argument("--max_posters", type=int, default=0, help="0 = all posters.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    out_generated_dir = os.path.join(args.out_dir, "generated_frames")
    ensure_dir(out_generated_dir)
    out_debug_dir = os.path.join(args.out_dir, "debug")
    ensure_dir(out_debug_dir)

    if args.full_cycle:
        frame_ids = list_ref_frames(args.ref_dir)
        if not frame_ids:
            raise RuntimeError(f"No frames found in {args.ref_dir}")
        frame_range = FrameRange(frame_ids[0], frame_ids[-1])
    else:
        frame_range = FrameRange(args.frame_start, args.frame_end)

    source_ref = load_ref_frame(args.ref_dir, args.source_frame)
    source_gray = cv2.cvtColor(source_ref, cv2.COLOR_BGR2GRAY)
    source_paper = paper_mask_from_frame(source_ref)
    cv2.imwrite(os.path.join(out_debug_dir, "source_paper_mask.png"), source_paper)

    # Poster for the primary (debug) output.
    if args.poster_from_ref_frame is not None:
        poster_bgr = load_ref_frame(args.ref_dir, args.poster_from_ref_frame)
        cv2.imwrite(os.path.join(args.out_dir, f"poster_from_ref_frame_{args.poster_from_ref_frame:04d}.png"), poster_bgr)
    elif args.poster is not None:
        poster_bgr = read_bgr(args.poster)
    else:
        poster_bgr = load_ref_frame(args.ref_dir, args.source_frame)

    source_canvas = prepare_source_canvas(poster_bgr, source_ref, source_paper)
    cv2.imwrite(os.path.join(out_debug_dir, "source_canvas.png"), source_canvas)

    # Posters batch (optional)
    afisha_items = []
    sequence_out = None
    if args.afisha_dir:
        posters = list_posters(args.afisha_dir)
        if args.max_posters and args.max_posters > 0:
            posters = posters[: args.max_posters]
        posters_out = os.path.join(args.out_dir, "posters")
        ensure_dir(posters_out)
        for poster_path in posters:
            stem = os.path.splitext(os.path.basename(poster_path))[0]
            per_out = os.path.join(posters_out, stem, "generated_frames")
            ensure_dir(per_out)
            poster_img = read_bgr(poster_path)
            src_canvas = prepare_source_canvas(poster_img, source_ref, source_paper)
            afisha_items.append({"stem": stem, "out_dir": per_out, "canvas": src_canvas})

        sequence_out = os.path.join(args.out_dir, "sequence_all", "generated_frames")
        ensure_dir(sequence_out)

    solid_bgr = parse_bgr(args.bg_color)

    report_rows = []
    grid_top = []
    grid_bot = []
    seq_idx = 1

    for idx in frame_range.indices():
        target_ref = load_ref_frame(args.ref_dir, idx)
        target_gray = cv2.cvtColor(target_ref, cv2.COLOR_BGR2GRAY)
        ignore = detect_pause_icon_mask(target_gray)

        flow = compute_flow_target_to_source(target_gray, source_gray, ignore_mask=ignore)
        paper = paper_mask_from_frame(target_ref)
        backside = white_backside_mask(target_ref, paper)

        if not args.transparent:
            bg = create_background(target_ref, paper, mode=args.bg_mode, solid_bgr=solid_bgr)
            if args.shadow and args.bg_mode == "solid":
                bg = apply_drop_shadow(bg, paper, strength=0.35)

        # Primary (debug/match) output
        warped = warp_with_flow(source_canvas, flow)
        lit = apply_lighting(
            warped,
            target_ref,
            paper_mask=paper,
            mode=args.l_mode,
            shade_sigma_mid=args.shade_sigma_mid,
            shade_sigma_low=args.shade_sigma_low,
            shade_clip_min=args.shade_clip_min,
            shade_clip_max=args.shade_clip_max,
        )
        back = render_backside(
            target_ref,
            paper_mask=paper,
            shade_sigma_mid=args.shade_sigma_mid,
            shade_sigma_low=args.shade_sigma_low,
            shade_clip_min=args.shade_clip_min,
            shade_clip_max=args.shade_clip_max,
        )
        paper_img = blend_front_back(lit, back, backside_alpha_u8=backside)

        if args.transparent:
            final = to_rgba(paper_img, paper_mask=paper)
        else:
            final = blend_paper_over_background(bg, paper_img, paper_mask=paper)
        cv2.imwrite(os.path.join(out_generated_dir, f"frame_{idx:04d}.png"), final)

        final_bgr_for_metrics = final[:, :, :3] if args.transparent else final
        pd = percent_diff(target_ref, final_bgr_for_metrics, valid_mask=paper)
        s = ssim_gray(
            cv2.cvtColor(target_ref, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(final_bgr_for_metrics, cv2.COLOR_BGR2GRAY),
            valid_mask=cv2.bitwise_and(paper, cv2.bitwise_not(ignore)),
        )
        report_rows.append({"frame": idx, "percent_diff": f"{pd:.4f}", "ssim": f"{s:.6f}"})

        if args.grid_range_start <= idx <= args.grid_range_end:
            grid_top.append(target_ref)
            grid_bot.append(final)

        # Batch posters
        if afisha_items:
            for item in afisha_items:
                warped_p = warp_with_flow(item["canvas"], flow)
                lit_p = apply_lighting(
                    warped_p,
                    target_ref,
                    paper_mask=paper,
                    mode=args.l_mode,
                    shade_sigma_mid=args.shade_sigma_mid,
                    shade_sigma_low=args.shade_sigma_low,
                    shade_clip_min=args.shade_clip_min,
                    shade_clip_max=args.shade_clip_max,
                )
                paper_p = blend_front_back(lit_p, back, backside_alpha_u8=backside)
                if args.transparent:
                    out_p = to_rgba(paper_p, paper_mask=paper)
                else:
                    out_p = blend_paper_over_background(bg, paper_p, paper_mask=paper)
                cv2.imwrite(os.path.join(item["out_dir"], f"frame_{idx:04d}.png"), out_p)
                cv2.imwrite(os.path.join(sequence_out, f"frame_{seq_idx:06d}.png"), out_p)
                seq_idx += 1

        if idx == args.grid_range_end:
            cv2.imwrite(os.path.join(out_debug_dir, f"paper_mask_{idx:04d}.png"), paper)
            cv2.imwrite(os.path.join(out_debug_dir, f"backside_mask_{idx:04d}.png"), backside)

    if not args.no_grid and grid_top and grid_bot:
        grid = build_grid_image(grid_top, grid_bot, scale=args.grid_scale)
        cv2.imwrite(os.path.join(args.out_dir, f"compare_grid_{args.grid_range_start}_{args.grid_range_end}.png"), grid)

    with open(os.path.join(args.out_dir, "diff_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "percent_diff", "ssim"])
        w.writeheader()
        w.writerows(report_rows)


if __name__ == "__main__":
    main()
