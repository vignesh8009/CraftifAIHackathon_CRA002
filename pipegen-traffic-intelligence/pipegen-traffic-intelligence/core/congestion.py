"""Congestion rule engine: Normal / Warning / Critical per region.

Escalation needs the higher level to hold for `persist_seconds`;
recovery needs the lower level to hold for `recover_seconds`.
This is an ESTIMATE from vehicle count in a region - not measured road speed."""
from .geometry import point_in_polygon

LEVELS = ("Normal", "Warning", "Critical")


def region_occupancy(regions, vehicles):
    """regions: {rid: [(x, y), ...]} pixels. Counts vehicle anchors inside each polygon."""
    occ = {rid: 0 for rid in regions}
    for v in vehicles:
        for rid, poly in regions.items():
            if point_in_polygon(v.anchor, poly):
                occ[rid] += 1
    return occ


class CongestionEngine:
    def __init__(self, thresholds, persist_seconds=3.0, recover_seconds=5.0):
        """thresholds: {rid: {"warning": int, "critical": int}}"""
        self.th = thresholds
        self.persist, self.recover = persist_seconds, recover_seconds
        self.level = {rid: 0 for rid in thresholds}
        self._pending = {rid: None for rid in thresholds}   # (target_level, since_ts)

    def status(self, rid):
        return LEVELS[self.level[rid]]

    def update(self, rid, count, ts):
        """Returns an alert dict ONLY when the status changes, else None."""
        th = self.th[rid]
        target = 2 if count >= th["critical"] else 1 if count >= th["warning"] else 0
        cur = self.level[rid]
        if target == cur:
            self._pending[rid] = None
            return None
        pend = self._pending[rid]
        if pend is None or pend[0] != target:
            self._pending[rid] = pend = (target, ts)
        hold = self.persist if target > cur else self.recover
        if ts - pend[1] < hold:
            return None
        self.level[rid], self._pending[rid] = target, None
        if target > cur:
            key = "critical" if target == 2 else "warning"
            rule = f"count {count} >= {key} threshold {th[key]} for {hold:g}s"
        else:
            rule = f"count {count} below {'warning' if target == 0 else 'critical'} " \
                   f"threshold {th['warning'] if target == 0 else th['critical']} for {hold:g}s"
        return {"region_id": rid, "count": count, "severity": LEVELS[target],
                "previous": LEVELS[cur], "rule": rule, "video_time_s": round(ts, 2)}
