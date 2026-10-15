import json, os, tempfile, time, unittest
from pathlib import Path
from core import store
from core.config import (ConfigError, VideoSourceError, load_config, validate_config,
                         validate_video_source)
from core.pipeline import TrafficAnalytics
from core.types import Track

ROOT = Path(__file__).resolve().parents[1]


class VideoSourceTests(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(VideoSourceError):
            validate_video_source("does_not_exist.mp4")

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with self.assertRaises(VideoSourceError):
                validate_video_source(f.name)

    def test_valid_file_url_and_camera(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            self.assertTrue(validate_video_source(f.name).endswith(".mp4"))
        self.assertEqual(validate_video_source("0"), 0)
        self.assertTrue(validate_video_source("rtsp://cam/stream").startswith("rtsp://"))

    def test_empty(self):
        with self.assertRaises(VideoSourceError):
            validate_video_source("  ")


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        load_config(ROOT / "config" / "analytics_config.json")

    def test_bad_thresholds(self):
        with self.assertRaises(ConfigError):
            validate_config({"regions": [{"id": "a", "polygon": [[0, 0], [1, 0], [1, 1]], "warning": 10, "critical": 5}]})

    def test_bad_point(self):
        with self.assertRaises(ConfigError):
            validate_config({"lines": [{"id": "l", "p1": [0, 0], "p2": [5, 1]}]})

    def test_missing_file_and_bad_json(self):
        with self.assertRaises(ConfigError):
            load_config("nope.json")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{bad")
        with self.assertRaises(ConfigError):
            load_config(f.name)
        os.unlink(f.name)


class StoreTests(unittest.TestCase):
    def test_missing_pipeline_data_returns_none_not_crash(self):
        d = tempfile.mkdtemp()
        self.assertIsNone(store.read_metrics(Path(d) / "metrics.json"))
        self.assertEqual(store.recent_alerts(Path(d) / "events.db"), [])

    def test_corrupt_metrics_returns_none(self):
        d = Path(tempfile.mkdtemp())
        (d / "metrics.json").write_text("{half")
        self.assertIsNone(store.read_metrics(d / "metrics.json"))

    def test_roundtrip_and_alerts(self):
        d = Path(tempfile.mkdtemp())
        store.write_metrics(d / "metrics.json", {"a": 1})
        self.assertEqual(store.read_metrics(d / "metrics.json")["a"], 1)
        store.insert_alert(d / "events.db", {"region_id": "z", "count": 9, "severity": "Warning",
                                             "previous": "Normal", "rule": "r", "video_time_s": 1.0}, "test")
        self.assertEqual(store.recent_alerts(d / "events.db")[0]["severity"], "Warning")


class PipelineIntegrationTests(unittest.TestCase):
    def test_end_to_end_metrics(self):
        cfg = load_config(ROOT / "config" / "analytics_config.json")
        a = TrafficAnalytics(cfg, "test")
        w, h = 1000, 1000
        # 10 stationary cars in the region for 5 s (video time) -> Warning
        cars = [Track(i, "car", 0.9, (150 + i * 60, 700, 190 + i * 60, 800)) for i in range(10)]
        alerts = []
        for k in range(10):
            alerts += a.process_frame(cars, k * 0.5, w, h)
        m = a.metrics()
        self.assertEqual(m["frame_vehicles"], 10)
        self.assertEqual(m["overall_status"], "Warning")
        self.assertEqual(len(alerts), 1)
        self.assertEqual(m["regions"]["zone_1"]["count"], 10)


if __name__ == "__main__":
    unittest.main()
