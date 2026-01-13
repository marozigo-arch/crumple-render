import argparse
import json
import subprocess
from pathlib import Path

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


def make_background(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.52
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = r / (max(h, w) * 0.9)
    r = np.clip(r, 0.0, 1.0)
    base = 14.0 + (1.0 - r) * 18.0
    bg = np.clip(base, 0, 255).astype(np.uint8)
    return np.dstack([bg, bg, bg])


def composite_rgba_on_bg(rgba: np.ndarray) -> np.ndarray:
    bgr = rgba[:, :, :3].astype(np.float32)
    a = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    h, w = bgr.shape[:2]
    bg = make_background(h, w).astype(np.float32)
    out_bgr = bgr * a + bg * (1.0 - a)
    return np.clip(out_bgr, 0, 255).astype(np.uint8)


def clean_pause_bars(bgr: np.ndarray) -> np.ndarray:
    # Remove the two white "pause" bars from screen-capture reference frames.
    # We detect two tall thin bright rectangles near the center and inpaint them.
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

    # Inpaint (best-effort) to remove the overlay; if it fails visually, metrics can still ignore this region.
    out = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    return out


def make_compare_grid(
    ref_dir: Path, gen_dir: Path, frame_start: int, frame_end: int, out_path: Path, scale: float = 0.35
) -> None:
    frames = list(range(frame_start, frame_end + 1))
    ref_imgs = [clean_pause_bars(read_bgr(ref_dir / f"frame_{f:04d}.png")) for f in frames]
    gen_imgs = [read_bgr(gen_dir / f"frame_{f:04d}.png") for f in frames]

    def resize(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ref_row = np.concatenate([resize(i) for i in ref_imgs], axis=1)
    gen_row = np.concatenate([resize(i) for i in gen_imgs], axis=1)
    grid = np.concatenate([ref_row, gen_row], axis=0)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), grid)


def make_poster_vs_frame(poster_path: Path, frame_rgba_path: Path, out_path: Path) -> float:
    poster = read_bgr(poster_path)
    rgba = cv2.imread(str(frame_rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        raise FileNotFoundError(frame_rgba_path)
    bgr = rgba[:, :, :3]
    a = rgba[:, :, 3]

    ys, xs = np.where(a > 10)
    if len(xs) < 10:
        raise RuntimeError("No alpha content in rendered frame")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = bgr[y0 : y1 + 1, x0 : x1 + 1]

    h, w = crop.shape[:2]
    poster_fit = fit_cover(poster, w=w, h=h)
    grid = np.concatenate([poster_fit, crop], axis=1)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), grid)

    diff = np.mean(np.abs(poster_fit.astype(np.float32) - crop.astype(np.float32))) / 255.0
    return float(diff)

def mean_abs_diff_on_alpha(ref_bgr: np.ndarray, gen_rgba: np.ndarray) -> float:
    if gen_rgba is None or gen_rgba.ndim != 3 or gen_rgba.shape[2] < 4:
        raise ValueError("gen_rgba must be RGBA")
    if ref_bgr.shape[:2] != gen_rgba.shape[:2]:
        raise ValueError("ref/gen must have same resolution for diff")
    a = gen_rgba[:, :, 3].astype(np.float32) / 255.0
    mask = a > 0.04
    if int(mask.sum()) < 50:
        return 1.0
    ref = ref_bgr.astype(np.float32)
    gen = gen_rgba[:, :, :3].astype(np.float32)
    diff = np.mean(np.abs(ref[mask] - gen[mask])) / 255.0
    return float(diff)


def run_blender(script: Path, poster_path: Path, out_frames_dir: Path, frame_start: int, frame_end: int, seed: int) -> None:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afisha_dir", default="afisha")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_start", type=int, default=33)
    ap.add_argument("--frame_end", type=int, default=44)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ref_dir", default="data/frames_ref")
    ap.add_argument("--max_posters", type=int, default=0)
    args = ap.parse_args()

    afisha_dir = Path(args.afisha_dir)
    if not afisha_dir.exists():
        fallback = Path("data/afisha-selected")
        if fallback.exists():
            afisha_dir = fallback
    if not afisha_dir.exists():
        raise SystemExit(f"Poster directory not found: {args.afisha_dir}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    posters = list_posters(afisha_dir)
    if not posters:
        raise SystemExit(f"No posters found in {afisha_dir}")
    if int(args.max_posters) > 0:
        posters = posters[: int(args.max_posters)]

    ref_dir = Path(args.ref_dir)
    script = Path(__file__).with_name("blender_cloth_crumple.py")

    posters_root = out_dir / "posters"
    ensure_dir(posters_root)

    for idx, poster in enumerate(posters):
        poster_out = posters_root / poster.stem
        frames_rgba_out = poster_out / "frames_rgba"
        frames_out = poster_out / "frames"
        ensure_dir(frames_rgba_out)
        ensure_dir(frames_out)

        run_blender(
            script=script,
            poster_path=poster,
            out_frames_dir=frames_rgba_out,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            seed=int(args.seed) + idx * 17,
        )

        # Composite for viewing.
        for f in range(args.frame_start, args.frame_end + 1):
            src = frames_rgba_out / f"frame_{f:04d}.png"
            dst = frames_out / f"frame_{f:04d}.png"
            rgba = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
                cv2.imwrite(str(dst), cv2.imread(str(src), cv2.IMREAD_COLOR))
                continue
            cv2.imwrite(str(dst), composite_rgba_on_bg(rgba))

        metrics = {}
        frame_end_rgba = frames_rgba_out / f"frame_{args.frame_end:04d}.png"
        poster_diff = make_poster_vs_frame(poster, frame_end_rgba, poster_out / "poster_vs_frame_end.png")
        metrics["poster_vs_frame_end_mean_abs_diff"] = poster_diff

        # Compare against reference (pause bars cleaned) on the generated paper alpha-mask.
        if ref_dir.exists():
            per_frame = {}
            for f in range(args.frame_start, args.frame_end + 1):
                ref_bgr = clean_pause_bars(read_bgr(ref_dir / f"frame_{f:04d}.png"))
                gen_rgba = cv2.imread(str(frames_rgba_out / f"frame_{f:04d}.png"), cv2.IMREAD_UNCHANGED)
                per_frame[str(f)] = mean_abs_diff_on_alpha(ref_bgr, gen_rgba)
            metrics["ref_mean_abs_diff_on_alpha_per_frame"] = per_frame
            metrics["ref_mean_abs_diff_on_alpha_avg"] = float(np.mean([per_frame[str(f)] for f in range(args.frame_start, args.frame_end + 1)]))

        if ref_dir.exists():
            make_compare_grid(
                ref_dir=ref_dir,
                gen_dir=frames_out,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                out_path=poster_out / f"compare_grid_{args.frame_start}_{args.frame_end}.png",
                scale=0.35,
            )

        ensure_dir(poster_out)
        (poster_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
