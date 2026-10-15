import unittest
from core.counting import LineCounter
from core.geometry import point_in_polygon
from core.congestion import region_occupancy
from core.types import Track, filter_vehicles

LINE = {"L": ((0, 100), (200, 100))}


def car(tid, y, x=100, cls="car", conf=0.9):
    return Track(tid, cls, conf, (x - 10, y - 20, x + 10, y))  # anchor = (x, y)


class CountingTests(unittest.TestCase):
    def test_crossing_counted_once(self):
        lc = LineCounter(LINE)
        for y in (80, 90, 110, 120):
            lc.update([car(1, y)])
        self.assertEqual(lc.total, 1)

    def test_no_duplicate_when_jittering_back_and_forth(self):
        lc = LineCounter(LINE)
        for y in (90, 110, 90, 110, 90, 110):
            lc.update([car(1, y)])
        self.assertEqual(lc.total, 1)

    def test_two_tracks_two_counts_by_class(self):
        lc = LineCounter(LINE)
        for y in (90, 110):
            lc.update([car(1, y), car(2, y, x=50, cls="bus")])
        self.assertEqual(lc.totals_by_class(), {"car": 1, "bus": 1})

    def test_untracked_objects_ignored(self):
        lc = LineCounter(LINE)
        for y in (90, 110):
            lc.update([car(-1, y), car(2 ** 64 - 1, y)])  # -1 and DeepStream UNTRACKED id
        self.assertEqual(lc.total, 0)

    def test_new_id_for_same_vehicle_is_a_new_track(self):
        # Documents a known limitation: ID switches can cause double counts.
        lc = LineCounter(LINE)
        lc.update([car(1, 90)]); lc.update([car(1, 110)])
        lc.update([car(5, 90)]); lc.update([car(5, 110)])
        self.assertEqual(lc.total, 2)

    def test_point_on_line_still_counts(self):
        lc = LineCounter(LINE)
        lc.update([car(1, 90)]); lc.update([car(1, 100)]); lc.update([car(1, 110)])
        self.assertEqual(lc.total, 1)

    def test_vehicle_not_crossing_not_counted(self):
        lc = LineCounter(LINE)
        for y in (10, 20, 30):
            lc.update([car(1, y)])
        self.assertEqual(lc.total, 0)


class FilterAndRegionTests(unittest.TestCase):
    def test_class_and_confidence_filter(self):
        ts = [car(1, 5, cls="car"), car(2, 5, cls="person"), car(3, 5, cls="truck", conf=0.2),
              car(4, 5, cls="motorcycle"), car(5, 5, cls="bus")]
        got = {t.track_id for t in filter_vehicles(ts, 0.35)}
        self.assertEqual(got, {1, 4, 5})

    def test_point_in_polygon(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(point_in_polygon((5, 5), sq))
        self.assertFalse(point_in_polygon((15, 5), sq))

    def test_region_occupancy(self):
        regions = {"A": [(0, 0), (100, 0), (100, 100), (0, 100)], "B": [(200, 0), (300, 0), (300, 100), (200, 100)]}
        veh = [car(1, 50, x=50), car(2, 60, x=60), car(3, 50, x=250), car(4, 500, x=500)]
        self.assertEqual(region_occupancy(regions, veh), {"A": 2, "B": 1})


if __name__ == "__main__":
    unittest.main()
