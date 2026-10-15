"""LOCAL PROTOTYPE (Windows/Linux, CPU or any GPU PyTorch can use).
YOLOv8 + ByteTrack via ultralytics -> shared analytics -> dashboard files.

THIS IS NOT TensorRT / DeepStream. Metrics are labelled 'prototype-python' so the
dashboard never presents them as NVIDIA-accelerated results.

Run from the project root:
    python prototype/detect_track.py --source videos/traffic.mp4
"""
import argparse, os, platform, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from core import store
from core.config import ConfigError, VideoSourceError, load_config, validate_video_source
from core.pipeline import TrafficAnalytics
from core.types import Track

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {"car": (0, 200, 0), "motorcycle": (255, 160, 0), "bus": (0, 160, 255), "truck": (0, 0, 255)}


def draw_overlay(frame, an, tracks):
    h, w = frame.shape[:2]
    for lid, (a, b) in an.lines_px.items():
        cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (0, 0, 255), 3)
    for poly in an.regions_px.values():
        pts = [[int(x), int(y)] for x, y in poly]
        cv2.polylines(frame, [np.array(pts)], True, (255, 255, 0), 2)
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        col = COLORS.get(t.cls, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"#{t.track_id} {t.cls} {t.conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, camera index, or rtsp:// URL")
    ap.add_argument("--config", default="config/analytics_config.json")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--conf", type=float, default=None, help="override detection threshold")
    ap.add_argument("--show", action="store_true", help="also open an OpenCV window")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    try:
        src = validate_video_source(a.source)
        cfg = load_config(a.config)
    except (VideoSourceError, ConfigError) as e:
        print(f"ERROR: {e}")
        return 2
    if a.conf:
        cfg["detection"]["min_confidence"] = a.conf

    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: run `pip install -r requirements-prototype.txt` first.")
        return 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = {"device": device, "python": platform.python_version(), "os": platform.platform(),
           "model": a.model, "note": "PyTorch prototype - NOT TensorRT/DeepStream"}
    out = Path(a.out)
    store.reset(out)
    an = TrafficAnalytics(cfg, f"prototype-python ({device})", env, str(a.source))

    cap = cv2.VideoCapture(src)
    vfps = (cap.get(cv2.CAP_PROP_FPS) or 0) if isinstance(src, str) and os.path.isfile(src) else 0
    cap.release()
    t0 = time.monotonic()
    try:
        results = YOLO(a.model).track(source=src, stream=True, persist=True, tracker="bytetrack.yaml",
                                      classes=list(COCO_VEHICLES), conf=cfg["detection"]["min_confidence"],
                                      verbose=False)
        for idx, r in enumerate(results):
            frame = r.orig_img
            h, w = frame.shape[:2]
            tracks = []
            if r.boxes is not None:
                for i, b in enumerate(r.boxes):
                    tid = int(r.boxes.id[i]) if r.boxes.id is not None else -1
                    tracks.append(Track(tid, COCO_VEHICLES[int(b.cls)], float(b.conf),
                                        tuple(float(v) for v in b.xyxy[0])))
            ts = idx / vfps if vfps else time.monotonic() - t0
            for al in an.process_frame(tracks, ts, w, h):
                store.insert_alert(out / "events.db", al, an.mode)
                print(f"ALERT t={ts:.1f}s {al['region_id']} {al['previous']}->{al['severity']}: {al['rule']}")
            frame = draw_overlay(frame, an, tracks)
            small = frame if w <= 1280 else cv2.resize(frame, (1280, int(h * 1280 / w)))
            if idx % 3 == 0:
                cv2.imwrite(str(out / "latest.tmp.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                store.safe_replace(out / "latest.tmp.jpg", out / "latest.jpg")
                store.write_metrics(out / "metrics.json", an.metrics())
            if a.show:
                cv2.imshow("PipeGen prototype (q to quit)", small)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if a.max_frames and idx + 1 >= a.max_frames:
                break
        store.write_metrics(out / "metrics.json", an.metrics(finished=True))
        print(f"Done. {an.frame_idx} frames, avg {an.frame_idx / (time.monotonic() - t0):.1f} FPS on {device}.")
        return 0
    except Exception as e:  # keep the dashboard alive and show the failure
        store.write_metrics(out / "metrics.json", an.metrics(finished=True, error=f"{type(e).__name__}: {e}"))
        print(f"ERROR during inference: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
