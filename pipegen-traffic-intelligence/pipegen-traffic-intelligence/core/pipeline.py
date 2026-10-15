"""TrafficAnalytics: one object that turns per-frame tracks into metrics + alerts.
Both prototype/detect_track.py and deepstream/run_pipeline.py call process_frame()."""
import time
from collections import Counter, deque
from .congestion import CongestionEngine, region_occupancy, LEVELS
from .counting import LineCounter
from .types import filter_vehicles


class TrafficAnalytics:
    def __init__(self, cfg, mode, env=None, source=""):
        self.cfg, self.mode, self.env, self.source = cfg, mode, env or {}, source
        cg = cfg.get("congestion", {})
        self.min_conf = cfg.get("detection", {}).get("min_confidence", 0.35)
        self.engine = CongestionEngine(
            {r["id"]: {"warning": r["warning"], "critical": r["critical"]} for r in cfg.get("regions", [])},
            cg.get("persist_seconds", 3), cg.get("recover_seconds", 5))
        self.counter = None
        self.regions_px = {}
        self.lines_px = {}
        self.frame_idx = 0
        self._wall = deque(maxlen=60)
        self.last = {"frame_vehicles": 0, "class_frame": {}, "occupancy": {}}

    def _build_geometry(self, w, h):
        sc = lambda p: (p[0] * w, p[1] * h)
        self.lines_px = {l["id"]: (sc(l["p1"]), sc(l["p2"])) for l in self.cfg.get("lines", [])}
        self.regions_px = {r["id"]: [sc(p) for p in r["polygon"]] for r in self.cfg.get("regions", [])}
        self.counter = LineCounter(self.lines_px)

    def process_frame(self, tracks, ts, w, h):
        """tracks: list[Track] (pixels); ts: video time in seconds. Returns list of new alerts."""
        if self.counter is None:
            self._build_geometry(w, h)
        veh = filter_vehicles(tracks, self.min_conf)
        self.counter.update(veh)
        occ = region_occupancy(self.regions_px, veh)
        alerts = [a for rid, n in occ.items() if (a := self.engine.update(rid, n, ts))]
        self.frame_idx += 1
        self._wall.append(time.perf_counter())
        self.last = {"frame_vehicles": len(veh), "class_frame": dict(Counter(v.cls for v in veh)),
                     "occupancy": occ}
        return alerts

    def fps(self):
        if len(self._wall) < 2 or self._wall[-1] == self._wall[0]:
            return 0.0
        return (len(self._wall) - 1) / (self._wall[-1] - self._wall[0])

    def metrics(self, finished=False, error=None):
        regions = {r["id"]: {"name": r.get("name", r["id"]), "count": self.last["occupancy"].get(r["id"], 0),
                             "status": self.engine.status(r["id"]),
                             "warning": r["warning"], "critical": r["critical"]}
                   for r in self.cfg.get("regions", [])}
        worst = max([self.engine.level[r] for r in regions] or [0])
        return {"source_mode": self.mode, "source": self.source, "env": self.env,
                "frame_idx": self.frame_idx, "fps": round(self.fps(), 1),
                "frame_vehicles": self.last["frame_vehicles"], "class_counts_frame": self.last["class_frame"],
                "line_counts": {k: dict(v) for k, v in (self.counter.counts if self.counter else {}).items()},
                "class_crossings": self.counter.totals_by_class() if self.counter else {},
                "total_crossings": self.counter.total if self.counter else 0,
                "regions": regions, "overall_status": LEVELS[worst],
                "finished": finished, "error": error}
