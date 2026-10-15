#!/usr/bin/env python3
"""DeepStream pipeline: file/RTSP -> nvstreammux -> nvinfer (TensorRT) -> nvtracker -> probe -> annotated MP4.
The probe feeds tracked vehicles into the SAME core analytics used by the prototype and
writes outputs/metrics.json + outputs/events.db for the dashboard.

Run on the NVIDIA machine, from the project root:
    python3 deepstream/run_pipeline.py --source videos/traffic.mp4

Requires DeepStream + pyds (Python bindings) matching your DeepStream version.
STATUS: written against the DeepStream Python API but NOT executed on NVIDIA hardware by the author.
"""
import argparse, signal, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import store
from core.config import ConfigError, VideoSourceError, load_config, validate_video_source
from core.pipeline import TrafficAnalytics
from core.types import Track

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
try:
    import pyds
except ImportError:
    sys.exit("ERROR: pyds (DeepStream Python bindings) not found. See docs/deployment.md step 4.")

DS = "/opt/nvidia/deepstream/deepstream"
MUX_W, MUX_H = 1280, 720   # nvstreammux size: bounding boxes in metadata are in THIS resolution


def gather_env():
    env = {"deepstream_path": DS}
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    smi = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if smi:
        env["gpu"], env["driver"] = [x.strip() for x in smi.splitlines()[0].split(",", 1)]
    else:
        env["gpu"] = "nvidia-smi not available (Jetson? check `tegrastats`)"
    env["deepstream_version"] = run(["deepstream-app", "--version-all"]).replace("\n", " | ")[:400]
    return env


class State:
    analytics = None
    out = None
    cfg = None
    last_write = 0.0


def seg_list(an):
    segs = [(a, b) for a, b in an.lines_px.values()]
    return segs, [(poly[i], poly[(i + 1) % len(poly)]) for poly in an.regions_px.values() for i in range(len(poly))]


def add_lines(batch_meta, frame_meta, segs, rgba):
    for i in range(0, len(segs), 16):          # display meta holds at most 16 lines
        dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        chunk = segs[i:i + 16]
        dm.num_lines = len(chunk)
        for k, (a, b) in enumerate(chunk):
            ln = dm.line_params[k]
            ln.x1, ln.y1, ln.x2, ln.y2 = int(a[0]), int(a[1]), int(b[0]), int(b[1])
            ln.line_width = 3
            ln.line_color.set(*rgba)
        pyds.nvds_add_display_meta_to_frame(frame_meta, dm)


def probe(pad, info, _):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    l_frame = batch.frame_meta_list
    while l_frame is not None:
        fm = pyds.NvDsFrameMeta.cast(l_frame.data)
        tracks, l_obj = [], fm.obj_meta_list
        while l_obj is not None:
            om = pyds.NvDsObjectMeta.cast(l_obj.data)
            r = om.rect_params
            conf = om.confidence if om.confidence >= 0 else 1.0   # tracker-predicted boxes report -0.1
            tr = Track(int(om.object_id), om.obj_label, float(conf), (r.left, r.top, r.left + r.width, r.top + r.height))
            tracks.append(tr)
            om.text_params.display_text = f"#{tr.track_id} {tr.cls} {conf:.2f}" if tr.is_tracked else f"{tr.cls} {conf:.2f}"
            l_obj = l_obj.next
        an = State.analytics
        for al in an.process_frame(tracks, fm.buf_pts / 1e9, MUX_W, MUX_H):
            store.insert_alert(State.out / "events.db", al, an.mode)
            print(f"ALERT t={al['video_time_s']}s {al['region_id']} {al['previous']}->{al['severity']}: {al['rule']}", flush=True)
        lines, edges = seg_list(an)
        add_lines(batch, fm, lines, (1, 0, 0, 1))
        add_lines(batch, fm, edges, (1, 1, 0, 1))
        if time.time() - State.last_write > 0.5:
            State.last_write = time.time()
            store.write_metrics(State.out / "metrics.json", an.metrics())
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file or rtsp:// URL")
    ap.add_argument("--config", default=str(ROOT / "config/analytics_config.json"))
    ap.add_argument("--infer-config", default=str(ROOT / "deepstream/config_infer_primary.txt"))
    ap.add_argument("--tracker-lib", default=f"{DS}/lib/libnvds_nvmultiobjecttracker.so")
    ap.add_argument("--tracker-config", default=f"{DS}/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    a = ap.parse_args()
    try:
        src = validate_video_source(a.source)
        cfg = load_config(a.config)
    except (VideoSourceError, ConfigError) as e:
        sys.exit(f"ERROR: {e}")
    if isinstance(src, int):
        sys.exit("ERROR: camera index not supported here; use a file or an rtsp:// URL.")
    is_live = src.startswith(("rtsp://", "http"))
    uri = src if is_live else Path(src).as_uri()

    State.out = Path(a.out)
    store.reset(State.out)
    State.analytics = TrafficAnalytics(cfg, "deepstream-tensorrt", gather_env(), a.source)

    Gst.init(None)
    desc = (
        f'nvstreammux name=mux batch-size=1 width={MUX_W} height={MUX_H} live-source={int(is_live)} '
        f'batched-push-timeout=40000 ! '
        f'nvinfer name=pgie config-file-path="{a.infer_config}" ! '
        f'nvtracker tracker-width=640 tracker-height=384 ll-lib-file="{a.tracker_lib}" '
        f'll-config-file="{a.tracker_config}" ! '
        f'nvvideoconvert ! nvdsosd name=osd ! nvvideoconvert ! nvv4l2h264enc bitrate=4000000 ! h264parse ! '
        f'qtmux ! filesink location="{State.out / "annotated.mp4"}" sync=false '
        f'uridecodebin uri="{uri}" ! mux.sink_0'
    )
    try:
        pipe = Gst.parse_launch(desc)
    except GLib.Error as e:
        sys.exit(f"ERROR building pipeline (missing DeepStream plugin?): {e}\nCheck: gst-inspect-1.0 nvinfer")
    pipe.get_by_name("osd").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe, None)

    loop = GLib.MainLoop()
    bus = pipe.get_bus()
    bus.add_signal_watch()
    failed = {"msg": None}

    def on_msg(_, msg):
        if msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            failed["msg"] = f"{err.message} ({dbg})"
            print("PIPELINE ERROR:", failed["msg"], flush=True)
            loop.quit()
    bus.connect("message", on_msg)
    signal.signal(signal.SIGINT, lambda *_: pipe.send_event(Gst.Event.new_eos()))  # Ctrl+C finalises the MP4

    t0 = time.time()
    pipe.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipe.set_state(Gst.State.NULL)
    an = State.analytics
    store.write_metrics(State.out / "metrics.json", an.metrics(finished=True, error=failed["msg"]))
    print(f"Done: {an.frame_idx} frames in {time.time() - t0:.1f}s "
          f"(wall-clock avg {an.frame_idx / max(time.time() - t0, 1e-6):.1f} FPS incl. engine build/encode).")
    return 1 if failed["msg"] else 0


if __name__ == "__main__":
    sys.exit(main())
