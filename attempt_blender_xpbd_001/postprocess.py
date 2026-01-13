import argparse
import json
from pathlib import Path

# When executed as `python attempt_blender_xpbd_001/postprocess.py`, this folder is on sys.path,
# so `import run` resolves to `attempt_blender_xpbd_001/run.py`.
from run import append_sequence, copy_cycle, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Existing run output directory (contains posters/<stem>/frames)")
    ap.add_argument("--frame_start", type=int, default=33)
    ap.add_argument("--frame_end", type=int, default=44)
    ap.add_argument("--cycle_sample_n", type=int, default=10)
    ap.add_argument("--make_sequence", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    posters_root = out_dir / "posters"
    if not posters_root.exists():
        raise SystemExit(f"Missing posters dir: {posters_root}")

    posters = sorted([p for p in posters_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if not posters:
        raise SystemExit(f"No posters in: {posters_root}")

    seq_dir = out_dir / "sequence" / "frames"
    seq_index = 1
    manifest = []

    for poster_dir in posters:
        frames_dir = poster_dir / "frames"
        if not frames_dir.exists():
            raise SystemExit(f"Missing frames dir: {frames_dir}")
        copy_cycle(
            frames_dir=frames_dir,
            out_cycle_dir=poster_dir / "cycle_frames",
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            sample_n=int(args.cycle_sample_n) if int(args.cycle_sample_n) > 0 else None,
        )
        if args.make_sequence:
            start = seq_index
            seq_index = append_sequence(poster_dir / "cycle_frames", seq_dir, start_index=seq_index)
            manifest.append(
                {
                    "poster_stem": poster_dir.name,
                    "sequence_start": start,
                    "sequence_end": seq_index - 1,
                }
            )

    if args.make_sequence:
        ensure_dir(out_dir / "sequence")
        (out_dir / "sequence" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        lengths = [m["sequence_end"] - m["sequence_start"] + 1 for m in manifest]
        print("sequence frames:", sum(lengths))


if __name__ == "__main__":
    main()
