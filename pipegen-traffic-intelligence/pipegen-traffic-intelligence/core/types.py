"""Shared data types. Used by BOTH the CPU prototype and the DeepStream app."""
from dataclasses import dataclass

VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")
# DeepStream marks objects the tracker has not (yet) assigned an ID to with 0xFFFFFFFFFFFFFFFF
_UNTRACKED_MIN = 2 ** 63


@dataclass(frozen=True)
class Track:
    track_id: int
    cls: str
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) in pixels

    @property
    def is_tracked(self) -> bool:
        return self.track_id is not None and 0 <= self.track_id < _UNTRACKED_MIN

    @property
    def anchor(self):
        """Bottom-centre of the box ~ where the vehicle touches the road."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


def filter_vehicles(tracks, min_conf=0.0):
    return [t for t in tracks if t.cls in VEHICLE_CLASSES and t.conf >= min_conf]
