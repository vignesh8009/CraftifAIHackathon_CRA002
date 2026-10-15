"""Tiny persistence layer: metrics.json (latest state) + SQLite (alert history).
The dashboard ONLY reads what the pipeline wrote here. Windows-safe."""
import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def safe_replace(src, dst, tries=40):
    """os.replace can fail on Windows while another process has the file open; retry briefly."""
    for _ in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(src, dst)


def write_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    metrics = dict(metrics, updated_epoch=time.time())
    tmp.write_text(json.dumps(metrics), encoding="utf-8")
    safe_replace(tmp, path)


def read_metrics(path):
    """Returns dict, or None if missing/unreadable (never raises)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _open(db):
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db, timeout=5)
    c.execute("""CREATE TABLE IF NOT EXISTS alerts(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, video_time_s REAL,
        region_id TEXT, count INTEGER, severity TEXT, previous TEXT, rule TEXT, source_mode TEXT)""")
    return c


def insert_alert(db, alert, source_mode):
    with closing(_open(db)) as c:
        c.execute("INSERT INTO alerts(ts_utc,video_time_s,region_id,count,severity,previous,rule,source_mode)"
                  " VALUES(?,?,?,?,?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(timespec="seconds"), alert.get("video_time_s"),
                   alert["region_id"], alert["count"], alert["severity"], alert["previous"],
                   alert["rule"], source_mode))
        c.commit()


def recent_alerts(db, limit=20):
    if not Path(db).is_file():
        return []
    try:
        with closing(_open(db)) as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))]
    except sqlite3.Error:
        return []


def reset(out_dir):
    """Start a fresh session. Never crashes if the dashboard has a file open."""
    out_dir = Path(out_dir)
    for name in ("metrics.json", "latest.jpg", "annotated.mp4"):
        try:
            (out_dir / name).unlink()
        except (FileNotFoundError, PermissionError):
            pass
    try:
        with closing(_open(out_dir / "events.db")) as c:
            c.execute("DELETE FROM alerts")
            c.commit()
    except sqlite3.Error:
        pass
