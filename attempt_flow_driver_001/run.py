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


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def paper_mask_from_frame(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    ignore = detect_pause_icon_mask(l)
    l2 = l.copy()
    if ignore.any():
        l2[ignore > 0] = int(np.median(l2))

    _, bw = cv2.threshold(l2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return bw

    # Largest non-background component
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == best).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    return mask


def white_backside_mask(bgr: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.int16)
    a = lab[:, :, 1].astype(np.int16) - 128
    bb = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt(a * a + bb * bb)
    white = (l > 200) & (chroma < 14) & (paper_mask > 0)
    return (white.astype(np.uint8) * 255)


def fit_cover_to_rect(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img_bgr.shape[:2]
    if iw == 0 or ih == 0:
        raise ValueError("empty image")

    scale = max(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x0 = max(0, (nw - w) // 2)
    y0 = max(0, (nh - h) // 2)
    return resized[y0 : y0 + h, x0 : x0 + w].copy()


def prepare_source_canvas(
    poster_bgr: np.ndarray, source_ref_bgr: np.ndarray, source_paper_mask: np.ndarray
) -> np.ndarray:
    h, w = source_ref_bgr.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # Use a slightly eroded paper bbox as the "front" placement area.
    er = cv2.erode(source_paper_mask, np.ones((21, 21), np.uint8), iterations=1)
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
        # Neutralize the pause icon area to avoid polluting flow.
        med_t = int(np.median(tg))
        med_s = int(np.median(sg))
        tg[ignore_mask > 0] = med_t
        sg[ignore_mask > 0] = med_s

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    dis.setPatchSize(8)
    dis.setPatchStride(4)
    dis.setGradientDescentIterations(25)
    flow = dis.calc(tg, sg, None)  # from target -> source
    return flow


def warp_with_flow(source_bgr: np.ndarray, flow_target_to_source: np.ndarray) -> np.ndarray:
    h, w = source_bgr.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow_target_to_source[:, :, 0].astype(np.float32)
    map_y = grid_y + flow_target_to_source[:, :, 1].astype(np.float32)
    warped = cv2.remap(source_bgr, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return warped


def apply_lighting(warped_poster_bgr: np.ndarray, target_ref_bgr: np.ndarray, mode: str) -> np.ndarray:
    """
    mode:
      - 'target': take L from target reference (max similarity to reference frames)
      - 'shading': take fold/shading from target, but preserve poster brightness/contrast
    """
    poster_lab = cv2.cvtColor(warped_poster_bgr, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_ref_bgr, cv2.COLOR_BGR2LAB)

    if mode == "target":
        out = poster_lab.copy()
        out[:, :, 0] = target_lab[:, :, 0]
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    if mode == "shading":
        tl = target_lab[:, :, 0].astype(np.float32) / 255.0
        pl = poster_lab[:, :, 0].astype(np.float32) / 255.0
        base = cv2.GaussianBlur(tl, (0, 0), 11.0)
        shade = tl / (base + 1e-6)
        shade = np.clip(shade, 0.55, 1.6)
        out = poster_lab.copy()
        out_l = np.clip(pl * shade, 0.0, 1.0) * 255.0
        out[:, :, 0] = out_l.astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    raise ValueError(f"Unknown lighting mode: {mode}")


def feather_alpha(mask_u8: np.ndarray, radius: int = 11) -> np.ndarray:
    m = (mask_u8.astype(np.float32) / 255.0)
    k = max(3, radius | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def composite_over_target(
    target_ref_bgr: np.ndarray, generated_bgr: np.ndarray, front_mask: np.ndarray
) -> np.ndarray:
    alpha = feather_alpha(front_mask, radius=21)[:, :, None]
    out = target_ref_bgr.astype(np.float32) * (1.0 - alpha) + generated_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


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


def load_ref_frame(ref_dir: str, idx: int) -> np.ndarray:
    return read_bgr(os.path.join(ref_dir, f"frame_{idx:04d}.png"))


def build_grid_image(top_frames: list[np.ndarray], bottom_frames: list[np.ndarray], scale: float = 0.5) -> np.ndarray:
    assert len(top_frames) == len(bottom_frames)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", default="data/frames_ref")
    ap.add_argument("--afisha_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_start", type=int, default=33)
    ap.add_argument("--frame_end", type=int, default=44)
    ap.add_argument("--source_frame", type=int, default=44)
    ap.add_argument("--l_mode", choices=["target", "shading"], default="target")
    ap.add_argument("--poster", default=None, help="Path to a single poster image.")
    ap.add_argument("--poster_from_ref_frame", type=int, default=None, help="Use a reference frame as poster input.")
    ap.add_argument("--grid_scale", type=float, default=0.5)
    ap.add_argument("--no_grid", action="store_true", help="Skip writing the 2-row comparison grid image.")
    args = ap.parse_args()

    frame_range = FrameRange(args.frame_start, args.frame_end)
    ensure_dir(args.out_dir)
    out_generated_dir = os.path.join(args.out_dir, "generated_frames")
    ensure_dir(out_generated_dir)
    out_debug_dir = os.path.join(args.out_dir, "debug")
    ensure_dir(out_debug_dir)

    source_ref = load_ref_frame(args.ref_dir, args.source_frame)
    source_gray = cv2.cvtColor(source_ref, cv2.COLOR_BGR2GRAY)
    source_paper = paper_mask_from_frame(source_ref)

    # If afisha mode is enabled, prepare all poster canvases once.
    afisha_items = []
    if args.afisha_dir:
        posters = list_posters(args.afisha_dir)
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

    # Poster selection
    if args.poster_from_ref_frame is not None:
        poster_bgr = load_ref_frame(args.ref_dir, args.poster_from_ref_frame)
        cv2.imwrite(os.path.join(args.out_dir, f"poster_from_ref_frame_{args.poster_from_ref_frame:04d}.png"), poster_bgr)
    elif args.poster is not None:
        poster_bgr = read_bgr(args.poster)
    else:
        poster_bgr = load_ref_frame(args.ref_dir, args.source_frame)

    source_canvas = prepare_source_canvas(poster_bgr, source_ref, source_paper)
    cv2.imwrite(os.path.join(out_debug_dir, "source_canvas.png"), source_canvas)
    cv2.imwrite(os.path.join(out_debug_dir, "source_paper_mask.png"), source_paper)

    report_rows = []
    top_refs = []
    bot_gens = []
    seq_idx = 1

    for idx in frame_range.indices():
        target_ref = load_ref_frame(args.ref_dir, idx)
        target_gray = cv2.cvtColor(target_ref, cv2.COLOR_BGR2GRAY)
        ignore = detect_pause_icon_mask(target_gray)
        flow = compute_flow_target_to_source(target_gray, source_gray, ignore_mask=ignore)

        paper = paper_mask_from_frame(target_ref)
        white = white_backside_mask(target_ref, paper)
        front = cv2.bitwise_and(paper, cv2.bitwise_not(white))

        # Primary (debug/match) output
        warped_primary = warp_with_flow(source_canvas, flow)
        recolored_primary = apply_lighting(warped_primary, target_ref, mode=args.l_mode)
        final = composite_over_target(target_ref, recolored_primary, front_mask=front)
        out_path = os.path.join(out_generated_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(out_path, final)

        valid = paper
        pd = percent_diff(target_ref, final, valid_mask=valid)
        s = ssim_gray(
            cv2.cvtColor(target_ref, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(final, cv2.COLOR_BGR2GRAY),
            valid_mask=cv2.bitwise_and(valid, cv2.bitwise_not(ignore)),
        )
        report_rows.append({"frame": idx, "percent_diff": f"{pd:.4f}", "ssim": f"{s:.6f}"})

        top_refs.append(target_ref)
        bot_gens.append(final)

        # Per-frame debug
        if idx == frame_range.end:
            cv2.imwrite(os.path.join(out_debug_dir, f"mask_paper_{idx:04d}.png"), paper)
            cv2.imwrite(os.path.join(out_debug_dir, f"mask_front_{idx:04d}.png"), front)
            diff = cv2.absdiff(target_ref, final)
            heat = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(out_debug_dir, f"diff_heat_{idx:04d}.png"), heat)

        # Afisha posters: generate all posters for this frame using the same flow.
        if afisha_items:
            for item in afisha_items:
                warped = warp_with_flow(item["canvas"], flow)
                recolored = apply_lighting(warped, target_ref, mode=args.l_mode)
                out_img = composite_over_target(target_ref, recolored, front_mask=front)
                cv2.imwrite(os.path.join(item["out_dir"], f"frame_{idx:04d}.png"), out_img)
                cv2.imwrite(os.path.join(sequence_out, f"frame_{seq_idx:06d}.png"), out_img)
                seq_idx += 1

    # Comparison grid
    if not args.no_grid:
        grid = build_grid_image(top_refs, bot_gens, scale=args.grid_scale)
        grid_path = os.path.join(args.out_dir, f"compare_grid_{args.frame_start}_{args.frame_end}.png")
        cv2.imwrite(grid_path, grid)

    # Report
    report_path = os.path.join(args.out_dir, "diff_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "percent_diff", "ssim"])
        w.writeheader()
        w.writerows(report_rows)


if __name__ == "__main__":
    main()
