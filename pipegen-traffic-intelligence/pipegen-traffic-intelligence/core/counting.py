from collections import Counter, defaultdict
from .geometry import segments_cross


class LineCounter:
    """Counts each tracked vehicle at most ONCE per line.
    lines: {line_id: ((x1, y1), (x2, y2))} in pixels."""

    def __init__(self, lines, prune_after=600):
        self.lines = lines
        self.counts = {lid: Counter() for lid in lines}
        self._prev, self._seen, self._counted = {}, {}, set()
        self._frame, self._prune_after = 0, prune_after

    def update(self, tracks):
        self._frame += 1
        events = []
        for t in tracks:
            if not t.is_tracked:          # no persistent ID -> cannot count safely
                continue
            cur, prev = t.anchor, self._prev.get(t.track_id)
            if prev is not None:
                for lid, (a, b) in self.lines.items():
                    key = (lid, t.track_id)
                    if key not in self._counted and segments_cross(prev, cur, a, b):
                        self._counted.add(key)
                        self.counts[lid][t.cls] += 1
                        events.append({"line_id": lid, "track_id": t.track_id, "cls": t.cls})
            self._prev[t.track_id] = cur
            self._seen[t.track_id] = self._frame
        if self._frame % self._prune_after == 0:
            self._prune()
        return events

    def _prune(self):
        old = {i for i, f in self._seen.items() if self._frame - f > self._prune_after}
        for i in old:
            self._prev.pop(i, None)
            self._seen.pop(i, None)
        self._counted = {k for k in self._counted if k[1] not in old}

    def totals_by_class(self):
        out = defaultdict(int)
        for c in self.counts.values():
            for k, v in c.items():
                out[k] += v
        return dict(out)

    @property
    def total(self):
        return sum(sum(c.values()) for c in self.counts.values())
