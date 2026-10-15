import json
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".h264", ".webm"}


class ConfigError(ValueError):
    pass


class VideoSourceError(ValueError):
    pass


def validate_video_source(src):
    """Returns an int (camera index), a URL string, or an absolute file path."""
    s = str(src).strip()
    if not s:
        raise VideoSourceError("No video source given.")
    if s.isdigit():
        return int(s)
    if s.lower().startswith(("rtsp://", "http://", "https://")):
        return s
    p = Path(s)
    if not p.exists():
        raise VideoSourceError(f"Video file not found: {p}")
    if not p.is_file():
        raise VideoSourceError(f"Not a file: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        raise VideoSourceError(f"Unsupported video format '{p.suffix}'. Use one of {sorted(VIDEO_EXTS)}")
    return str(p.resolve())


def _pt(v, where):
    if (not isinstance(v, (list, tuple)) or len(v) != 2
            or not all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in v)):
        raise ConfigError(f"{where}: points must be [x, y] with values between 0 and 1 (normalised)")
    return (float(v[0]), float(v[1]))


def validate_config(cfg):
    try:
        det = cfg.get("detection", {})
        conf = det.get("min_confidence", 0.35)
        if not 0 < conf < 1:
            raise ConfigError("detection.min_confidence must be between 0 and 1")
        cg = cfg.get("congestion", {})
        if cg.get("persist_seconds", 3) < 0 or cg.get("recover_seconds", 5) < 0:
            raise ConfigError("congestion persist/recover seconds must be >= 0")
        ids = set()
        for ln in cfg.get("lines", []):
            _pt(ln["p1"], ln["id"]); _pt(ln["p2"], ln["id"])
            ids.add(ln["id"])
        if len(ids) != len(cfg.get("lines", [])):
            raise ConfigError("duplicate line ids")
        rids = set()
        for r in cfg.get("regions", []):
            if len(r["polygon"]) < 3:
                raise ConfigError(f"region {r['id']}: polygon needs >= 3 points")
            for p in r["polygon"]:
                _pt(p, r["id"])
            if not 0 < r["warning"] < r["critical"]:
                raise ConfigError(f"region {r['id']}: need 0 < warning < critical")
            rids.add(r["id"])
        if len(rids) != len(cfg.get("regions", [])):
            raise ConfigError("duplicate region ids")
    except KeyError as e:
        raise ConfigError(f"missing config key: {e}") from e
    return cfg


def load_config(path):
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {p}: {e}") from e
    return validate_config(cfg)
