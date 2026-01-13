import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_posters(dir_path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    posters = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    posters.sort(key=lambda p: p.name.lower())
    return posters


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def fit_cover(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img_bgr.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x0 = max(0, (nw - w) // 2)
    y0 = max(0, (nh - h) // 2)
    return resized[y0 : y0 + h, x0 : x0 + w].copy()

def clean_pause_bars(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    x0, x1 = int(w * 0.35), int(w * 0.65)
    y0, y1 = int(h * 0.35), int(h * 0.75)
    roi = gray[y0:y1, x0:x1]
    mu, sd = float(roi.mean()), float(roi.std())
    th = mu + 0.7 * sd
    mask_roi = (roi > th).astype(np.uint8) * 255
    mask_roi = cv2.morphologyEx(
        mask_roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)), iterations=1
    )
    mask_roi = cv2.morphologyEx(
        mask_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 21)), iterations=1
    )
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        rx, ry, rw, rh = cv2.boundingRect(c)
        area = rw * rh
        if rh > 80 and rw < 80 and area > 2000:
            rects.append((area, rx, ry, rw, rh))
    rects.sort(reverse=True)
    if len(rects) < 2:
        return bgr

    mask = np.zeros((h, w), dtype=np.uint8)
    pad_x, pad_y = 12, 10
    for _, rx, ry, rw, rh in rects[:2]:
        ax0, ay0 = x0 + rx, y0 + ry
        ax1, ay1 = ax0 + rw, ay0 + rh
        ax0 = max(0, ax0 - pad_x)
        ax1 = min(w, ax1 + pad_x)
        ay0 = max(0, ay0 - pad_y)
        ay1 = min(h, ay1 + pad_y)
        mask[ay0:ay1, ax0:ax1] = 255
    out = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    # Second pass helps to suppress remaining bright edges.
    out = cv2.inpaint(out, mask, 9, cv2.INPAINT_NS)
    # Feather the region to avoid obvious seams in grids.
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=2.0, sigmaY=2.0)
    m = (mask.astype(np.float32) / 255.0)[:, :, None]
    out = (blur * m + out * (1.0 - m)).astype(np.uint8)
    return out

def mask_from_reference_frame(ref_bgr: np.ndarray) -> np.ndarray:
    # Rough segmentation: paper is bright-ish vs dark background.
    gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu on blurred image for stability.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Keep largest connected component.
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(m)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=1)
    return mask

def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return float(inter / union) if union > 0 else 0.0


def make_poster_vs_frame44(poster_path: Path, frame44_path: Path, out_path: Path) -> None:
    poster = read_bgr(poster_path)
    rgba = cv2.imread(str(frame44_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        frame = read_bgr(frame44_path)
        h, w = frame.shape[:2]
        poster_fit = fit_cover(poster, w=w, h=h)
        grid = np.concatenate([poster_fit, frame], axis=1)
        ensure_dir(out_path.parent)
        cv2.imwrite(str(out_path), grid)
        return

    bgr = rgba[:, :, :3]
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 10)
    if len(xs) < 10:
        h, w = bgr.shape[:2]
        poster_fit = fit_cover(poster, w=w, h=h)
        grid = np.concatenate([poster_fit, bgr], axis=1)
        ensure_dir(out_path.parent)
        cv2.imwrite(str(out_path), grid)
        return

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = bgr[y0 : y1 + 1, x0 : x1 + 1]
    # Compare poster to the extracted paper region (approx).
    h, w = crop.shape[:2]
    poster_fit = fit_cover(poster, w=w, h=h)
    grid = np.concatenate([poster_fit, crop], axis=1)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), grid)

def poster_diff_vs_frame_rgba(poster_path: Path, frame_path: Path) -> float:
    poster = read_bgr(poster_path)
    rgba = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        raise FileNotFoundError(frame_path)
    bgr = rgba[:, :, :3]
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 10)
    if len(xs) < 10:
        return 1.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = bgr[y0 : y1 + 1, x0 : x1 + 1]
    h, w = crop.shape[:2]
    poster_fit = fit_cover(poster, w=w, h=h)
    # Compare only well-inside the paper to avoid antialiased boundary differences.
    ac = a[y0 : y1 + 1, x0 : x1 + 1]
    inner = ac > 220
    if int(inner.sum()) < 50:
        inner = ac > 100
    diff = np.mean(np.abs(poster_fit[inner].astype(np.float32) - crop[inner].astype(np.float32))) / 255.0
    return float(diff)


def make_background(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.52
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = r / (max(h, w) * 0.9)
    r = np.clip(r, 0.0, 1.0)
    # center slightly brighter, edges darker
    base = 14.0 + (1.0 - r) * 18.0
    bg = np.clip(base, 0, 255).astype(np.uint8)
    return np.dstack([bg, bg, bg])


def composite_on_bg(rgba_path: Path) -> None:
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        return
    bgr = rgba[:, :, :3].astype(np.float32)
    a = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    h, w = bgr.shape[:2]
    bg = make_background(h, w).astype(np.float32)
    out = bgr * a + bg * (1.0 - a)
    cv2.imwrite(str(rgba_path), np.clip(out, 0, 255).astype(np.uint8))

def make_compare_grid(ref_dir: Path, gen_dir: Path, frame_start: int, frame_end: int, out_path: Path, scale: float = 0.35) -> None:
    frames = list(range(frame_start, frame_end + 1))
    ref_imgs = [clean_pause_bars(read_bgr(ref_dir / f"frame_{f:04d}.png")) for f in frames]
    gen_imgs = [read_bgr(gen_dir / f"frame_{f:04d}.png") for f in frames]

    def resize(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ref_row = np.concatenate([resize(i) for i in ref_imgs], axis=1)
    gen_row = np.concatenate([resize(i) for i in gen_imgs], axis=1)
    grid = np.concatenate([ref_row, gen_row], axis=0)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), grid)

def run_blender(poster_path: Path, out_frames_dir: Path, frame_start: int, frame_end: int, seed: int) -> None:
    script = Path(__file__).with_name("blender_xpbd_paper.py")
    cmd = [
        "blender",
        "-b",
        "-noaudio",
        "-P",
        str(script),
        "--",
        "--poster",
        str(poster_path),
        "--out_dir",
        str(out_frames_dir),
        "--frame_start",
        str(frame_start),
        "--frame_end",
        str(frame_end),
        "--seed",
        str(seed),
        "--samples",
        "48",
    ]
    subprocess.check_call(cmd)

def copy_cycle(frames_dir: Path, out_cycle_dir: Path, frame_start: int, frame_end: int, sample_n: Optional[int] = None) -> None:
    ensure_dir(out_cycle_dir)
    frames = list(range(frame_start, frame_end + 1))
    if sample_n and sample_n > 0 and sample_n < len(frames):
        idxs = np.linspace(0, len(frames) - 1, sample_n)
        frames = [frames[int(round(i))] for i in idxs]
        frames = sorted(set(frames))

    if sample_n and sample_n > 0:
        # Exactly N frames unfold + N frames fold (20 frames when N=10).
        seq = frames + list(reversed(frames))
    else:
        seq = frames + list(reversed(frames[:-1]))  # avoid duplicate end

    for out_i, f in enumerate(seq, start=1):
        src = frames_dir / f"frame_{f:04d}.png"
        dst = out_cycle_dir / f"frame_{out_i:04d}.png"
        dst.write_bytes(src.read_bytes())

def append_sequence(src_cycle_dir: Path, dst_sequence_dir: Path, start_index: int) -> int:
    ensure_dir(dst_sequence_dir)
    frames = sorted([p for p in src_cycle_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    idx = start_index
    for p in frames:
        dst = dst_sequence_dir / f"frame_{idx:06d}.png"
        dst.write_bytes(p.read_bytes())
        idx += 1
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afisha_dir", default="afisha")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_start", type=int, default=33)
    ap.add_argument("--frame_end", type=int, default=44)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ref_dir", default="data/frames_ref")
    ap.add_argument("--max_posters", type=int, default=0)
    ap.add_argument("--make_cycle", action="store_true")
    ap.add_argument("--cycle_sample_n", type=int, default=0)
    ap.add_argument("--make_sequence", action="store_true")
    args = ap.parse_args()

    afisha_dir = Path(args.afisha_dir)
    if not afisha_dir.exists():
        fallback = Path("data/afisha-selected")
        if fallback.exists():
            afisha_dir = fallback
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    posters = list_posters(afisha_dir)
    if not posters:
        raise SystemExit(f"No posters found in {afisha_dir}")
    if int(args.max_posters) > 0:
        posters = posters[: int(args.max_posters)]

    posters_root = out_dir / "posters"
    ensure_dir(posters_root)
    ref_dir = Path(args.ref_dir)
    sequence_manifest = []
    seq_dir = out_dir / "sequence" / "frames"
    seq_index = 1

    for idx, poster in enumerate(posters):
        poster_out = posters_root / poster.stem
        frames_rgba_out = poster_out / "frames_rgba"
        frames_out = poster_out / "frames"
        ensure_dir(frames_rgba_out)
        ensure_dir(frames_out)
        run_blender(
            poster_path=poster,
            out_frames_dir=frames_rgba_out,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            seed=int(args.seed) + idx * 17,
        )
        # Composite RGBA renders onto a consistent dark background for convenient viewing,
        # while keeping the original RGBA frames in frames_rgba/.
        for f in range(args.frame_start, args.frame_end + 1):
            src = frames_rgba_out / f"frame_{f:04d}.png"
            dst = frames_out / f"frame_{f:04d}.png"
            rgba = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
                # If Blender produced RGB, just copy.
                cv2.imwrite(str(dst), cv2.imread(str(src), cv2.IMREAD_COLOR))
                continue
            bgr = rgba[:, :, :3].astype(np.float32)
            a = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
            h, w = bgr.shape[:2]
            bg = make_background(h, w).astype(np.float32)
            out_bgr = bgr * a + bg * (1.0 - a)
            cv2.imwrite(str(dst), np.clip(out_bgr, 0, 255).astype(np.uint8))

        frame44 = frames_rgba_out / f"frame_{args.frame_end:04d}.png"
        make_poster_vs_frame44(poster, frame44, poster_out / "poster_vs_frame44.png")
        metrics = {"poster_vs_frame44_mean_abs_diff": poster_diff_vs_frame_rgba(poster, frame44)}
        # Visual comparison vs reference sequence (top ref / bottom generated).
        if ref_dir.exists():
            make_compare_grid(
                ref_dir=ref_dir,
                gen_dir=frames_out,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                out_path=poster_out / f"compare_grid_{args.frame_start}_{args.frame_end}.png",
                scale=0.35,
            )
            # Compare silhouette similarity (IoU) as a texture-invariant metric.
            ious = {}
            for f in range(args.frame_start, args.frame_end + 1):
                ref = clean_pause_bars(read_bgr(ref_dir / f"frame_{f:04d}.png"))
                ref_mask = mask_from_reference_frame(ref)
                gen_rgba = cv2.imread(str(frames_rgba_out / f"frame_{f:04d}.png"), cv2.IMREAD_UNCHANGED)
                gen_mask = (gen_rgba[:, :, 3] > 10).astype(np.uint8) * 255
                ious[str(f)] = iou(ref_mask, gen_mask)
            metrics["ref_mask_iou_per_frame"] = ious
            metrics["ref_mask_iou_avg"] = float(np.mean([ious[str(f)] for f in range(args.frame_start, args.frame_end + 1)]))
        (poster_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if args.make_cycle:
            sample_n = int(args.cycle_sample_n) if int(args.cycle_sample_n) > 0 else None
            copy_cycle(frames_out, poster_out / "cycle_frames", args.frame_start, args.frame_end, sample_n=sample_n)
        if args.make_sequence:
            if not args.make_cycle:
                raise SystemExit("--make_sequence requires --make_cycle (sequence is built from cycle_frames)")
            start = seq_index
            seq_index = append_sequence(poster_out / "cycle_frames", seq_dir, start_index=seq_index)
            sequence_manifest.append(
                {
                    "poster": str(poster),
                    "poster_stem": poster.stem,
                    "sequence_start": start,
                    "sequence_end": seq_index - 1,
                }
            )

    if args.make_sequence:
        ensure_dir((out_dir / "sequence"))
        (out_dir / "sequence" / "manifest.json").write_text(json.dumps(sequence_manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
