import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


PAPER_ASPECT_W_OVER_H = 864 / 1104  # match reference framing


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_posters(afisha_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    posters = [p for p in afisha_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    posters.sort(key=lambda p: p.name.lower())
    return posters


def fit_cover_to_aspect(img_bgr: np.ndarray, aspect_w_over_h: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    src_aspect = w / h
    if abs(src_aspect - aspect_w_over_h) < 1e-6:
        return img_bgr

    if src_aspect > aspect_w_over_h:
        # too wide => crop width
        new_w = int(round(h * aspect_w_over_h))
        x0 = max(0, (w - new_w) // 2)
        return img_bgr[:, x0 : x0 + new_w].copy()
    # too tall => crop height
    new_h = int(round(w / aspect_w_over_h))
    y0 = max(0, (h - new_h) // 2)
    return img_bgr[y0 : y0 + new_h, :].copy()


def prepare_poster(poster_path: Path, out_path: Path, max_size: int = 2048) -> None:
    img = cv2.imread(str(poster_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(poster_path)

    img = fit_cover_to_aspect(img, PAPER_ASPECT_W_OVER_H)

    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale != 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), img)


def make_contact_sheet(frames_dir: Path, out_path: Path, cols: int = 10) -> None:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return

    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in frames]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return

    # downscale for sheet
    target_h = 240
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        scale = target_h / h
        resized.append(cv2.resize(im, (int(round(w * scale)), target_h), interpolation=cv2.INTER_AREA))

    cols = min(cols, len(resized))
    rows = int(np.ceil(len(resized) / cols))
    cell_h = target_h
    cell_w = max(im.shape[1] for im in resized)
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, im in enumerate(resized):
        r = i // cols
        c = i % cols
        y0 = r * cell_h
        x0 = c * cell_w
        sheet[y0 : y0 + im.shape[0], x0 : x0 + im.shape[1]] = im

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), sheet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--afisha_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_posters", type=int, default=0, help="0 = all")
    ap.add_argument("--frames_out", type=int, default=10)
    ap.add_argument("--sim_frames", type=int, default=120)
    ap.add_argument("--engine", choices=["EEVEE", "CYCLES"], default="CYCLES")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--grid_x", type=int, default=100)
    ap.add_argument("--grid_y", type=int, default=130)
    ap.add_argument("--cloth_quality", type=int, default=8)
    ap.add_argument("--vary_sim", action="store_true", help="Re-bake sim per poster (slow).")
    args = ap.parse_args()

    afisha_dir = Path(args.afisha_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    posters = list_posters(afisha_dir)
    if args.max_posters and args.max_posters > 0:
        posters = posters[: args.max_posters]
    if not posters:
        raise SystemExit(f"No posters found in {afisha_dir}")

    prepared_dir = out_dir / "_prepared_posters"
    ensure_dir(prepared_dir)

    items = []
    for i, poster in enumerate(posters):
        stem = poster.stem
        prepared = prepared_dir / f"{stem}.png"
        prepare_poster(poster, prepared)
        items.append({"stem": stem, "path": str(prepared), "seed": 1000 + i})

    posters_json = out_dir / "posters.json"
    posters_json.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    blender_script = Path(__file__).with_name("blender_batch.py")
    cmd = [
        "blender",
        "-b",
        "-noaudio",
        "-P",
        str(blender_script),
        "--",
        "--posters_json",
        str(posters_json),
        "--out_dir",
        str(out_dir),
        "--frames_out",
        str(args.frames_out),
        "--sim_frames",
        str(args.sim_frames),
        "--engine",
        args.engine,
        "--samples",
        str(args.samples),
        "--grid_x",
        str(args.grid_x),
        "--grid_y",
        str(args.grid_y),
        "--cloth_quality",
        str(args.cloth_quality),
    ]
    if args.vary_sim:
        cmd.append("--vary_sim")

    subprocess.check_call(cmd)

    # contact sheet for first poster
    first_frames = out_dir / "posters" / items[0]["stem"] / "frames"
    make_contact_sheet(first_frames, out_dir / "contact_sheet.png", cols=args.frames_out)


if __name__ == "__main__":
    main()
