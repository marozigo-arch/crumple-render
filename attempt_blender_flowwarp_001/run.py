import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from flow_assets import FlowParams, extract_flow_assets, paper_mask_from_frame
from metrics import evaluate_pair, write_csv
from visualize import make_compare_grid


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def list_posters(afisha_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    posters = [p for p in afisha_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    posters.sort(key=lambda p: p.name.lower())
    return posters


def prepare_poster(poster_path: Path, out_path: Path, max_size: int = 2048) -> None:
    img = read_bgr(poster_path)
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale != 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), img)


def fit_cover_to_rect(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img_bgr.shape[:2]
    scale = max(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x0 = max(0, (nw - w) // 2)
    y0 = max(0, (nh - h) // 2)
    return resized[y0 : y0 + h, x0 : x0 + w].copy()


def prepare_source_canvas(poster_bgr: np.ndarray, source_ref_bgr: np.ndarray, source_paper_mask: np.ndarray) -> np.ndarray:
    """
    Build a full-frame source texture where the poster occupies only the paper area (rest = white).
    This prevents sampling background pixels into the paper when flow maps slightly outside.
    """
    h, w = source_ref_bgr.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

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

def render_blender(poster: Path, assets_dir: Path, out_frames_dir: Path, frame_start: int, frame_end: int, samples: int) -> None:
    blender_script = Path(__file__).with_name("blender_render_flowwarp.py")
    cmd = [
        "blender",
        "-b",
        "-noaudio",
        "-P",
        str(blender_script),
        "--",
        "--poster",
        str(poster),
        "--assets_dir",
        str(assets_dir),
        "--out_dir",
        str(out_frames_dir),
        "--frame_start",
        str(frame_start),
        "--frame_end",
        str(frame_end),
        "--samples",
        str(samples),
    ]
    subprocess.check_call(cmd)


def score(ref_dir: Path, gen_dir: Path, frame_start: int, frame_end: int) -> list:
    rows = []
    for f in range(frame_start, frame_end + 1):
        r = ref_dir / f"frame_{f:04d}.png"
        g = gen_dir / f"frame_{f:04d}.png"
        rows.append(evaluate_pair(str(r), str(g)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", default="data/frames_ref")
    ap.add_argument("--afisha_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_start", type=int, default=33)
    ap.add_argument("--frame_end", type=int, default=44)
    ap.add_argument("--source_frame", type=int, default=44)
    ap.add_argument("--poster", default=None)
    ap.add_argument("--poster_from_ref_frame", type=int, default=None)
    ap.add_argument("--grid_scale", type=float, default=0.35)
    ap.add_argument("--samples", type=int, default=24)
    args = ap.parse_args()

    ref_dir = Path(args.ref_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    prepared = out_dir / "_prepared"
    ensure_dir(prepared)

    # Poster selection
    if args.poster_from_ref_frame is not None:
        poster_path = prepared / f"poster_from_ref_{args.poster_from_ref_frame:04d}.png"
        src_ref = read_bgr(ref_dir / f"frame_{args.source_frame:04d}.png")
        src_mask = paper_mask_from_frame(src_ref)
        img = read_bgr(ref_dir / f"frame_{args.poster_from_ref_frame:04d}.png")
        canvas = img.copy()
        # Only the paper pixels should be in the source texture (outside = white).
        canvas[src_mask == 0] = (255, 255, 255)
        cv2.imwrite(str(poster_path), canvas)
    elif args.poster is not None:
        poster_path = prepared / Path(args.poster).name
        src_ref = read_bgr(ref_dir / f"frame_{args.source_frame:04d}.png")
        src_mask = paper_mask_from_frame(src_ref)
        poster_img = read_bgr(Path(args.poster))
        canvas = prepare_source_canvas(poster_img, src_ref, src_mask)
        cv2.imwrite(str(poster_path), canvas)
    else:
        if not args.afisha_dir:
            raise SystemExit("Provide --poster or --poster_from_ref_frame or --afisha_dir.")
        posters = list_posters(Path(args.afisha_dir))
        if not posters:
            raise SystemExit("No posters found.")
        poster_path = prepared / f"{posters[0].stem}.png"
        src_ref = read_bgr(ref_dir / f"frame_{args.source_frame:04d}.png")
        src_mask = paper_mask_from_frame(src_ref)
        poster_img = read_bgr(posters[0])
        canvas = prepare_source_canvas(poster_img, src_ref, src_mask)
        cv2.imwrite(str(poster_path), canvas)

    # Assets (flow+mask+backside)
    assets_dir = out_dir / "assets"
    params = FlowParams()
    extract_flow_assets(
        ref_dir=str(ref_dir),
        out_assets_dir=str(assets_dir),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        source_frame=args.source_frame,
        params=params,
    )
    (out_dir / "flow_params.json").write_text(json.dumps(params.__dict__, indent=2), encoding="utf-8")

    # Render primary (debug) sequence
    gen_dir = out_dir / "generated_frames"
    ensure_dir(gen_dir)
    render_blender(
        poster=poster_path,
        assets_dir=assets_dir,
        out_frames_dir=gen_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        samples=args.samples,
    )

    # Metrics + grid
    rows = score(ref_dir, gen_dir, args.frame_start, args.frame_end)
    write_csv(rows, str(out_dir / "diff_report.csv"))
    make_compare_grid(
        ref_dir=str(ref_dir),
        gen_dir=str(gen_dir),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        out_path=str(out_dir / f"compare_grid_{args.frame_start}_{args.frame_end}.png"),
        scale=float(args.grid_scale),
    )

    # Batch posters
    if args.afisha_dir:
        posters = list_posters(Path(args.afisha_dir))
        posters_root = out_dir / "posters"
        ensure_dir(posters_root)
        src_ref = read_bgr(ref_dir / f"frame_{args.source_frame:04d}.png")
        src_mask = paper_mask_from_frame(src_ref)
        for p in posters:
            prep = prepared / f"{p.stem}.png"
            poster_img = read_bgr(p)
            canvas = prepare_source_canvas(poster_img, src_ref, src_mask)
            cv2.imwrite(str(prep), canvas)
            per_gen = posters_root / p.stem / "generated_frames"
            ensure_dir(per_gen)
            render_blender(
                poster=prep,
                assets_dir=assets_dir,
                out_frames_dir=per_gen,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                samples=args.samples,
            )


if __name__ == "__main__":
    main()
