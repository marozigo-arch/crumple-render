import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from flow_assets import AssetParams, FlowParams, extract_flow_assets, paper_mask_from_frame
from metrics import evaluate_pair, write_csv
from render_opencv import RenderParams, render_sequence
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


def render_blender(
    poster: Path,
    assets_dir: Path,
    out_frames_dir: Path,
    frame_start: int,
    frame_end: int,
    samples: int,
    poster_interp: str,
    view: str,
    light_scale: float,
) -> None:
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
        "--poster_interp",
        poster_interp,
        "--view",
        view,
        "--light_scale",
        str(light_scale),
    ]
    subprocess.check_call(cmd)


def render_opencv(
    poster: Path,
    assets_dir: Path,
    out_frames_dir: Path,
    frame_start: int,
    frame_end: int,
    poster_interp: str,
    light_scale: float,
) -> None:
    render_sequence(
        poster_path=poster,
        assets_dir=assets_dir,
        out_dir=out_frames_dir,
        frame_start=frame_start,
        frame_end=frame_end,
        params=RenderParams(poster_interp=poster_interp, light_scale=float(light_scale)),
    )


def score(ref_dir: Path, gen_dir: Path, frame_start: int, frame_end: int) -> list:
    rows = []
    for f in range(frame_start, frame_end + 1):
        r = ref_dir / f"frame_{f:04d}.png"
        g = gen_dir / f"frame_{f:04d}.png"
        rows.append(evaluate_pair(str(r), str(g)))
    return rows


def summarize(rows) -> dict:
    diffs = [r.percent_diff for r in rows]
    ssims = [r.ssim for r in rows]
    return {
        "avg_percent_diff": float(np.mean(diffs)) if diffs else 0.0,
        "max_percent_diff": float(np.max(diffs)) if diffs else 0.0,
        "avg_ssim": float(np.mean(ssims)) if ssims else 0.0,
        "min_ssim": float(np.min(ssims)) if ssims else 0.0,
    }


def copy_best(src: Path, dst: Path, grid_name: str) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ensure_dir(dst)
    for name in ["generated_frames", "assets", "diff_report.csv", grid_name]:
        p = src / name
        if p.is_dir():
            shutil.copytree(p, dst / name)
        elif p.is_file():
            shutil.copy2(p, dst / name)


def candidate_params() -> list[tuple[FlowParams, AssetParams, str, str]]:
    """
    Returns a compact set of candidate settings:
    - FlowParams
    - AssetParams
    - poster_interp
    - view
    """
    flow_candidates = [
        FlowParams(finest_scale=1, patch_size=8, patch_stride=4, gd_iters=25),
        FlowParams(finest_scale=1, patch_size=8, patch_stride=4, gd_iters=40),
        FlowParams(finest_scale=1, patch_size=12, patch_stride=6, gd_iters=25),
        FlowParams(finest_scale=2, patch_size=8, patch_stride=4, gd_iters=25),
    ]
    asset_candidates = [
        AssetParams(
            uv_mode="w_minus_1",
            alpha_feather_sigma=1.0,
            back_blur_sigma=1.0,
            light_blur_sigma=0.6,
            back_l_min=225,
            back_chroma_max=14.0,
            back_warp_diff_min=18.0,
            back_l_delta_min=12.0,
        ),
        AssetParams(
            uv_mode="half_pixel",
            alpha_feather_sigma=1.0,
            back_blur_sigma=1.0,
            light_blur_sigma=0.6,
            back_l_min=225,
            back_chroma_max=14.0,
            back_warp_diff_min=18.0,
            back_l_delta_min=12.0,
        ),
        AssetParams(
            uv_mode="half_pixel",
            alpha_feather_sigma=0.6,
            back_blur_sigma=0.8,
            light_blur_sigma=0.2,
            back_l_min=232,
            back_chroma_max=12.0,
            back_warp_diff_min=22.0,
            back_l_delta_min=14.0,
        ),
        AssetParams(
            uv_mode="half_pixel",
            alpha_feather_sigma=1.6,
            back_blur_sigma=1.2,
            light_blur_sigma=1.0,
            back_l_min=225,
            back_chroma_max=14.0,
            back_warp_diff_min=18.0,
            back_l_delta_min=12.0,
        ),
    ]
    poster_interps = ["Linear"]
    views = ["Filmic", "Standard"]

    out = []
    for fp in flow_candidates:
        for ap in asset_candidates:
            for pi in poster_interps:
                for v in views:
                    out.append((fp, ap, pi, v))
    return out


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
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--renderer", choices=["opencv", "blender"], default="opencv")
    args = ap.parse_args()

    ref_dir = Path(args.ref_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    prepared = out_dir / "_prepared"
    ensure_dir(prepared)

    # Poster selection
    posters_for_batch: list[Path] | None = None
    if args.poster_from_ref_frame is not None:
        poster_path = prepared / f"poster_from_ref_{args.poster_from_ref_frame:04d}.png"
        src_mask = paper_mask_from_frame(read_bgr(ref_dir / f"frame_{args.source_frame:04d}.png"))
        img = read_bgr(ref_dir / f"frame_{args.poster_from_ref_frame:04d}.png")
        canvas = img.copy()
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
        posters_for_batch = list_posters(Path(args.afisha_dir))
        if not posters_for_batch:
            raise SystemExit("No posters found.")
        # For reference verification we need a comparable texture; default to using the source ref frame as the poster.
        poster_path = prepared / f"poster_from_ref_{args.source_frame:04d}.png"
        src_mask = paper_mask_from_frame(read_bgr(ref_dir / f"frame_{args.source_frame:04d}.png"))
        img = read_bgr(ref_dir / f"frame_{args.source_frame:04d}.png")
        canvas = img.copy()
        canvas[src_mask == 0] = (255, 255, 255)
        cv2.imwrite(str(poster_path), canvas)

    # Main path: optionally autotune against reference
    best_dir = out_dir / "best"
    ensure_dir(best_dir)

    best_summary = None
    best_payload = None

    candidates_root = out_dir / "_candidates"
    grid_name = f"compare_grid_{args.frame_start}_{args.frame_end}.png"
    if args.autotune:
        ensure_dir(candidates_root)
        for idx, (flow_p, asset_p, poster_interp, view) in enumerate(candidate_params()):
            cand_dir = candidates_root / f"cand_{idx:03d}"
            ensure_dir(cand_dir)

            assets_dir = cand_dir / "assets"
            extract_flow_assets(
                ref_dir=str(ref_dir),
                out_assets_dir=str(assets_dir),
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                source_frame=args.source_frame,
                flow_params=flow_p,
                asset_params=asset_p,
            )

            gen_dir = cand_dir / "generated_frames"
            ensure_dir(gen_dir)
            if args.renderer == "blender":
                render_blender(
                    poster=poster_path,
                    assets_dir=assets_dir,
                    out_frames_dir=gen_dir,
                    frame_start=args.frame_start,
                    frame_end=args.frame_end,
                    samples=1,
                    poster_interp=poster_interp,
                    view=view,
                    light_scale=float(asset_p.light_scale),
                )
            else:
                render_opencv(
                    poster=poster_path,
                    assets_dir=assets_dir,
                    out_frames_dir=gen_dir,
                    frame_start=args.frame_start,
                    frame_end=args.frame_end,
                    poster_interp=poster_interp,
                    light_scale=float(asset_p.light_scale),
                )

            rows = score(ref_dir, gen_dir, args.frame_start, args.frame_end)
            write_csv(rows, str(cand_dir / "diff_report.csv"))
            make_compare_grid(
                ref_dir=str(ref_dir),
                gen_dir=str(gen_dir),
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                out_path=str(cand_dir / grid_name),
                scale=float(args.grid_scale),
            )
            cand_summary = summarize(rows)
            payload = {
                "candidate": idx,
                "flow_params": asdict(flow_p),
                "asset_params": asdict(asset_p),
                "poster_interp": poster_interp,
                "view": view,
                "summary": cand_summary,
            }
            (cand_dir / "candidate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            if best_summary is None:
                best_summary = cand_summary
                best_payload = payload
                best_payload["_cand_dir"] = str(cand_dir)
                continue

            def key(s):
                return (s["avg_percent_diff"], s["max_percent_diff"], -s["avg_ssim"])

            if key(cand_summary) < key(best_summary):
                best_summary = cand_summary
                best_payload = payload
                best_payload["_cand_dir"] = str(cand_dir)

            if cand_summary["avg_percent_diff"] <= 10.0 and cand_summary["max_percent_diff"] <= 10.5:
                break

        if best_payload is None:
            raise SystemExit("No candidates produced.")

        best_cand_dir = Path(best_payload["_cand_dir"])
        copy_best(best_cand_dir, best_dir, grid_name=grid_name)
        (out_dir / "best_params.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    else:
        # Single run with defaults
        flow_p = FlowParams()
        asset_p = AssetParams()
        assets_dir = out_dir / "assets"
        extract_flow_assets(
            ref_dir=str(ref_dir),
            out_assets_dir=str(assets_dir),
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            source_frame=args.source_frame,
            flow_params=flow_p,
            asset_params=asset_p,
        )
        gen_dir = out_dir / "generated_frames"
        ensure_dir(gen_dir)
        if args.renderer == "blender":
            render_blender(
                poster=poster_path,
                assets_dir=assets_dir,
                out_frames_dir=gen_dir,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                samples=args.samples,
                poster_interp="Linear",
                view="Standard",
                light_scale=float(asset_p.light_scale),
            )
        else:
            render_opencv(
                poster=poster_path,
                assets_dir=assets_dir,
                out_frames_dir=gen_dir,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                poster_interp="Linear",
                light_scale=float(asset_p.light_scale),
            )
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

    # Batch posters (uses best assets if autotune ran; otherwise uses default assets)
    if args.afisha_dir:
        posters = posters_for_batch if posters_for_batch is not None else list_posters(Path(args.afisha_dir))
        posters_root = out_dir / "posters"
        ensure_dir(posters_root)
        seq_root = out_dir / "sequence_all" / "frames"
        ensure_dir(seq_root)
        manifest = []
        global_idx = 1
        src_ref = read_bgr(ref_dir / f"frame_{args.source_frame:04d}.png")
        src_mask = paper_mask_from_frame(src_ref)

        if args.autotune:
            best_assets = best_dir / "assets"
            best_params = json.loads((out_dir / "best_params.json").read_text(encoding="utf-8"))
            poster_interp = best_params["poster_interp"]
            view = best_params["view"]
            light_scale = float(best_params["asset_params"]["light_scale"])
        else:
            best_assets = out_dir / "assets"
            poster_interp = "Linear"
            view = "Standard"
            light_scale = float(AssetParams().light_scale)

        for p in posters:
            prep = prepared / f"{p.stem}.png"
            poster_img = read_bgr(p)
            canvas = prepare_source_canvas(poster_img, src_ref, src_mask)
            cv2.imwrite(str(prep), canvas)
            per_gen = posters_root / p.stem / "generated_frames"
            ensure_dir(per_gen)
            start_idx = global_idx
            if args.renderer == "blender":
                render_blender(
                    poster=prep,
                    assets_dir=best_assets,
                    out_frames_dir=per_gen,
                    frame_start=args.frame_start,
                    frame_end=args.frame_end,
                    samples=args.samples,
                    poster_interp=poster_interp,
                    view=view,
                    light_scale=light_scale,
                )
            else:
                render_opencv(
                    poster=prep,
                    assets_dir=best_assets,
                    out_frames_dir=per_gen,
                    frame_start=args.frame_start,
                    frame_end=args.frame_end,
                    poster_interp=poster_interp,
                    light_scale=light_scale,
                )

            # Concatenate into a single sequence for easy video assembly.
            for f in range(args.frame_start, args.frame_end + 1):
                src = per_gen / f"frame_{f:04d}.png"
                dst = seq_root / f"frame_{global_idx:06d}.png"
                shutil.copy2(src, dst)
                global_idx += 1
            manifest.append(
                {
                    "poster": p.name,
                    "poster_stem": p.stem,
                    "frames_ref_start": int(args.frame_start),
                    "frames_ref_end": int(args.frame_end),
                    "sequence_start": int(start_idx),
                    "sequence_end": int(global_idx - 1),
                }
            )

        (out_dir / "sequence_all" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
