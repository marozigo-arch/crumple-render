import os
import random
import re
import shutil
import subprocess


def _parse_score(text):
    m = re.search(r"SCORE:\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def run():
    ref = "data/frames_ref/frame_0040.png"
    out_dir = "data/render_paper_xpbd"
    os.makedirs(out_dir, exist_ok=True)

    # Small search space by default; increase --trials for better results.
    trials = int(os.environ.get("PAPER_TRIALS", "12"))
    base_seed = int(os.environ.get("PAPER_SEED", "7"))
    samples = int(os.environ.get("PAPER_SAMPLES", "32"))

    rng = random.Random(base_seed)

    candidates = []
    for t in range(trials):
        cfg = {
            "seed": rng.randint(1, 10_000_000),
            "nx": rng.choice([32, 35, 40]), # Higher res for better folds
            "ny": rng.choice([42, 45, 50]),
            "n_creases": rng.choice([250, 300, 350]), # More creases
            "crease_angle_deg": rng.choice([140.0, 160.0, 175.0]), # SHARP folds
            "stretch_stiffness": rng.choice([0.95, 0.98, 1.0]), # No stretch
            "hinge_stiffness": rng.choice([0.85, 0.9, 0.95]), # Very stiff hinge (Plasticity)
            "unfold_max": rng.choice([0.6, 0.8, 0.9]), # Ensure opening
            "unfold_power": rng.choice([1.0, 1.3]),
            "attract0": rng.choice([8.0, 10.0]),
            "attract_tau": rng.choice([0.02, 0.04]),
            "crease_tau": rng.choice([0.3, 0.5]),
            "pre_roll": rng.choice([40, 60]), 
            "substeps": rng.choice([4, 5]), # High stability for stiff constraints
            "iters": rng.choice([10, 15, 20]), # More solver iterations for stiffness
            "damping": rng.choice([0.15, 0.2]),
            "gravity": rng.choice([-0.15, -0.25]),
        }
        candidates.append(cfg)

    best = {"score": -1e9, "cfg": None}

    for idx, cfg in enumerate(candidates, start=1):
        print(f"\n--- Trial {idx}/{len(candidates)} ---")
        print(cfg)

        cmd = [
            "blender",
            "-b",
            "-noaudio",
            "-P",
            "simulate_paper_xpbd.py",
            "--",
            "--frame",
            "40",
            "--start_frame",
            "33",
            "--end_frame",
            "45",
            "--out_dir",
            out_dir,
            "--samples",
            str(samples),
        ]
        for k, v in cfg.items():
            cmd += [f"--{k}", str(v)]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("Simulation failed.")
            continue

        gen = os.path.join(out_dir, "frame_0040.png")
        vis = os.path.join(out_dir, "compare_0040.png")
        metric_cmd = ["python3", "metrics_paper.py", ref, gen, "--vis", vis]
        m = subprocess.run(metric_cmd, capture_output=True, text=True)
        print(m.stdout.strip())
        score = _parse_score(m.stdout)
        if score is None:
            print("Metric parse failed.")
            continue

        if score > best["score"]:
            best = {"score": score, "cfg": cfg}
            try:
                shutil.copy(gen, "data/best_paper_result_0040.png")
                shutil.copy(vis, "data/best_paper_compare_0040.png")
                print("NEW BEST")
            except Exception as e:
                print(f"Warning: Failed to save best image: {e}")

    print("\nDone.")
    print("Best score:", best["score"])
    print("Best cfg:", best["cfg"])
    print("Saved: data/best_paper_result_0040.png and data/best_paper_compare_0040.png")


if __name__ == "__main__":
    run()
