import argparse
import subprocess
from pathlib import Path

import cv2


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_frame(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def write_frames(frames, out_dir: Path, stem: str):
    ensure_dir(out_dir)
    for i, img in enumerate(frames, start=1):
        cv2.imwrite(str(out_dir / f"{stem}_{i:06d}.png"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Directory with frame_0033.png..frame_0044.png (BGR)")
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--pause_frames", type=int, default=45)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    out_mp4 = Path(args.out_mp4)
    ensure_dir(out_mp4.parent)

    unfold = [read_frame(frames_dir / f"frame_{f:04d}.png") for f in range(33, 45)]
    pause = [unfold[-1]] * int(args.pause_frames)
    fold = list(reversed(unfold))
    seq = unfold + pause + fold

    tmp = out_mp4.parent / "_tmp_video_frames"
    if tmp.exists():
        for p in tmp.glob("*.png"):
            p.unlink()
    ensure_dir(tmp)
    write_frames(seq, tmp, stem="vid")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(args.fps)),
        "-i",
        str(tmp / "vid_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(out_mp4),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

