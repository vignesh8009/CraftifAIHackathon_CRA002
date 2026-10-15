import unittest
from core.congestion import CongestionEngine

TH = {"z": {"warning": 5, "critical": 10}}


def run(engine, series):
    """series: [(ts, count)] -> list of alerts"""
    out = []
    for ts, n in series:
        a = engine.update("z", n, ts)
        if a:
            out.append(a)
    return out


class CongestionTests(unittest.TestCase):
    def test_short_spike_does_not_alert(self):
        e = CongestionEngine(TH, persist_seconds=3, recover_seconds=5)
        self.assertEqual(run(e, [(0, 6), (1, 6), (2, 1), (3, 1)]), [])
        self.assertEqual(e.status("z"), "Normal")

    def test_warning_after_persistence_only_once(self):
        e = CongestionEngine(TH, 3, 5)
        alerts = run(e, [(t, 6) for t in range(0, 12)])
        self.assertEqual(len(alerts), 1)           # not one alert per frame
        self.assertEqual(alerts[0]["severity"], "Warning")
        self.assertIn("warning threshold 5", alerts[0]["rule"])

    def test_escalate_to_critical(self):
        e = CongestionEngine(TH, 2, 5)
        alerts = run(e, [(0, 6), (2, 6), (3, 12), (5, 12)])
        self.assertEqual([a["severity"] for a in alerts], ["Warning", "Critical"])
        self.assertIn("critical threshold 10", alerts[1]["rule"])

    def test_recovery_to_normal(self):
        e = CongestionEngine(TH, 2, 4)
        alerts = run(e, [(0, 6), (2, 6), (3, 0), (6, 0), (7, 0)])
        self.assertEqual([a["severity"] for a in alerts], ["Warning", "Normal"])
        self.assertEqual(e.status("z"), "Normal")

    def test_recovery_needs_to_persist(self):
        e = CongestionEngine(TH, 2, 4)
        alerts = run(e, [(0, 6), (2, 6), (3, 0), (4, 6), (5, 0), (6, 0)])
        self.assertEqual([a["severity"] for a in alerts], ["Warning"])

    def test_critical_drops_to_warning(self):
        e = CongestionEngine(TH, 1, 2)
        alerts = run(e, [(0, 12), (1, 12), (2, 6), (4, 6)])
        self.assertEqual([a["severity"] for a in alerts], ["Critical", "Warning"])


if __name__ == "__main__":
    unittest.main()
