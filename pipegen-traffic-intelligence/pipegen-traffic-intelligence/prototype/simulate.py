"""SIMULATED traffic -> real analytics code -> metrics/alerts for the dashboard.
Use ONLY to test the dashboard on a laptop. Output is labelled SIMULATED and is NOT inference.
Run from the project root:  python prototype/simulate.py"""
import argparse, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import store
from core.config import load_config
from core.pipeline import TrafficAnalytics
from core.types import Track

W, H = 1280, 720


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/analytics_config.json")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out)
    store.reset(out)
    an = TrafficAnalytics(cfg, "simulated", {"note": "SIMULATED DATA - no model, no GPU"}, "simulator")
    random.seed(1)
    cars, nid, t, dt = {}, 1, 0.0, 1.0 / a.fps
    while t < a.seconds:
        # arrival rate ramps up after 25 s to create congestion
        rate = 0.15 if t < 25 else 0.7 if t < 60 else 0.1
        if random.random() < rate:
            cars[nid] = [random.uniform(0.2, 0.8) * W, 0.3 * H, random.choice(["car"] * 6 + ["bus", "truck", "motorcycle"])]
            nid += 1
        tracks = []
        for i in list(cars):
            x, y, c = cars[i]
            cars[i][1] = y + (4 if 25 <= t < 60 else 10) * (0.6 + 0.4 * (i % 3))  # slower when busy
            if cars[i][1] > 0.98 * H:
                del cars[i]; continue
            tracks.append(Track(i, c, 0.9, (x - 25, cars[i][1] - 40, x + 25, cars[i][1])))
        for al in an.process_frame(tracks, t, W, H):
            store.insert_alert(out / "events.db", al, "simulated")
        if an.frame_idx % 5 == 0:
            store.write_metrics(out / "metrics.json", an.metrics())
        t += dt
        time.sleep(dt / a.speed)
    store.write_metrics(out / "metrics.json", an.metrics(finished=True))


if __name__ == "__main__":
    main()
