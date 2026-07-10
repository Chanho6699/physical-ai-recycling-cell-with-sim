from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: list[float]

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
