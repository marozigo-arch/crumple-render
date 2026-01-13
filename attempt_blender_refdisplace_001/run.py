import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import cv2

from extract_assets import DisplaceParams, extract_displacement_sequence, extract_poster_from_ref
from metrics import evaluate_pair, summarize, write_csv
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


def prepare_poster(path_in: Path, path_out: Path, max_size: int = 2048) -> None:
    img = read_bgr(path_in)
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale != 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ensure_dir(path_out.parent)
    cv2.imwrite(str(path_out), img)


def render_blender(
    poster_path: Path,
    disp_dir: Path,
    out_frames_dir: Path,
    frame_start: int,
    frame_end: int,
    disp_strength: float,
    samples: int,
    subdiv: int,
) -> None:
    blender_script = Path(__file__).with_name("blender_render_displace.py")
    cmd = [
        "blender",
        "-b",
        "-noaudio",
        "-P",
        str(blender_script),
        "--",
        "--poster",
        str(poster_path),
        "--disp_dir",
        str(disp_dir),
        "--out_dir",
        str(out_frames_dir),
        "--frame_start",
        str(frame_start),
        "--frame_end",
        str(frame_end),
        "--disp_strength",
        str(disp_strength),
        "--samples",
        str(samples),
        "--subdiv",
        str(subdiv),
    ]
    subprocess.check_call(cmd)


def score_sequence(ref_dir: Path, gen_dir: Path, frame_start: int, frame_end: int) -> dict:
    rows = []
    for f in range(frame_start, frame_end + 1):
        r = ref_dir / f"frame_{f:04d}.png"
        g = gen_dir / f"frame_{f:04d}.png"
        rows.append(evaluate_pair(str(r), str(g)))
    return {"rows": rows, "summary": summarize(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", default="data/frames_ref")
    ap.add_argument("--afisha_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_start", type=int, default=33)
    ap.add_argument("--frame_end", type=int, default=44)
    ap.add_argument("--poster", default=None)
    ap.add_argument("--poster_from_ref", type=int, default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--grid_scale", type=float, default=0.35)
    args = ap.parse_args()

    ref_dir = Path(args.ref_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Prepare poster
    prepared_dir = out_dir / "_prepared"
    ensure_dir(prepared_dir)

    if args.poster_from_ref is not None:
        ref_img = read_bgr(ref_dir / f"frame_{args.poster_from_ref:04d}.png")
        poster_img = extract_poster_from_ref(ref_img)
        poster_path = prepared_dir / f"poster_from_ref_{args.poster_from_ref:04d}.png"
        cv2.imwrite(str(poster_path), poster_img)
    elif args.poster is not None:
        poster_path = prepared_dir / Path(args.poster).name
        prepare_poster(Path(args.poster), poster_path)
    else:
        # fallback: first afisha if provided
        if args.afisha_dir:
            posters = list_posters(Path(args.afisha_dir))
            if not posters:
                raise SystemExit("No posters found.")
            poster_path = prepared_dir / f"{posters[0].stem}.png"
            prepare_poster(posters[0], poster_path)
        else:
            raise SystemExit("Provide --poster or --poster_from_ref or --afisha_dir.")

    # Tuning: try a small grid of params and keep the best by max percent_diff.
    candidates = []
    if args.tune:
        for sigma_small in [2.0, 2.5, 3.0]:
            for sigma_large in [18.0, 24.0, 30.0]:
                for gamma in [0.8, 1.0, 1.2]:
                    candidates.append(
                        {
                            "disp_params": DisplaceParams(sigma_small=sigma_small, sigma_large=sigma_large, gamma=gamma, clip=2.5),
                            "disp_strength": 0.18,
                        }
                    )
        # few strength variants
        for c in list(candidates):
            for strength in [0.14, 0.18, 0.22, 0.26]:
                cc = dict(c)
                cc["disp_strength"] = strength
                candidates.append(cc)
        # keep unique-ish by json
        seen = set()
        uniq = []
        for c in candidates:
            k = json.dumps({"dp": asdict(c["disp_params"]), "s": c["disp_strength"]}, sort_keys=True)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(c)
        candidates = uniq[:18]  # cap for runtime
    else:
        candidates = [{"disp_params": DisplaceParams(), "disp_strength": 0.22}]

    best = None
    best_dir = None
    best_summary = None

    for i, cand in enumerate(candidates):
        trial_dir = out_dir / ("trial_%02d" % i)
        ensure_dir(trial_dir)

        disp_dir = trial_dir / "displacement"
        extract_displacement_sequence(
            ref_dir=str(ref_dir),
            out_dir=str(disp_dir),
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            params=cand["disp_params"],
        )

        gen_dir = trial_dir / "generated_frames"
        ensure_dir(gen_dir)
        # low-cost render for tuning
        render_blender(
            poster_path=poster_path,
            disp_dir=disp_dir,
            out_frames_dir=gen_dir,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            disp_strength=float(cand["disp_strength"]),
            samples=8 if args.tune else 16,
            subdiv=4,
        )

        scored = score_sequence(ref_dir, gen_dir, args.frame_start, args.frame_end)
        summary = scored["summary"]

        # Optimize for max percent_diff (primary), then avg
        key = (summary["percent_diff_max"], summary["percent_diff_avg"])
        if best is None or key < best:
            best = key
            best_dir = trial_dir
            best_summary = summary
            (trial_dir / "params.json").write_text(
                json.dumps(
                    {
                        "disp_params": asdict(cand["disp_params"]),
                        "disp_strength": cand["disp_strength"],
                        "summary": summary,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    # Promote best to stable outputs
    assert best_dir is not None
    final_disp = out_dir / "displacement"
    final_gen = out_dir / "generated_frames"
    ensure_dir(final_disp)
    ensure_dir(final_gen)

    params_data = json.loads((best_dir / "params.json").read_text(encoding="utf-8"))
    dp = DisplaceParams(**params_data["disp_params"])
    ds = float(params_data["disp_strength"])

    extract_displacement_sequence(
        ref_dir=str(ref_dir),
        out_dir=str(final_disp),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        params=dp,
    )
    render_blender(
        poster_path=poster_path,
        disp_dir=final_disp,
        out_frames_dir=final_gen,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        disp_strength=ds,
        samples=24,
        subdiv=5,
    )

    scored = score_sequence(ref_dir, final_gen, args.frame_start, args.frame_end)
    write_csv(scored["rows"], str(out_dir / "diff_report.csv"))
    (out_dir / "best_params.json").write_text(json.dumps(params_data, ensure_ascii=False, indent=2), encoding="utf-8")
    make_compare_grid(
        ref_dir=str(ref_dir),
        gen_dir=str(final_gen),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        out_path=str(out_dir / f"compare_grid_{args.frame_start}_{args.frame_end}.png"),
        scale=float(args.grid_scale),
    )

    # Batch posters (optional) using the best displacement maps/strength.
    if args.afisha_dir:
        posters = list_posters(Path(args.afisha_dir))
        posters_root = out_dir / "posters"
        ensure_dir(posters_root)
        for p in posters:
            poster_prep = prepared_dir / f"{p.stem}.png"
            prepare_poster(p, poster_prep)
            gen_dir_p = posters_root / p.stem / "generated_frames"
            ensure_dir(gen_dir_p)
            render_blender(
                poster_path=poster_prep,
                disp_dir=final_disp,
                out_frames_dir=gen_dir_p,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                disp_strength=ds,
                samples=24,
                subdiv=5,
            )


if __name__ == "__main__":
    main()
