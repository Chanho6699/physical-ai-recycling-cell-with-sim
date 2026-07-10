"""YOLO-based SafetyMonitor (v0).

Runs a pretrained Ultralytics YOLO model as-is (no training, no ONNX
export, no TensorRT conversion) and flags configured hazard labels
(default: "person") in a frame. Implements the same SafetyMonitor
interface as MockSafetyMonitor, so it is a drop-in replacement in the
safety gate.
"""

from typing import Optional

import numpy as np
from ultralytics import YOLO

from safety.safety_monitor import SafetyMonitor
from safety.safety_types import SafetyDecision

DEFAULT_HAZARD_LABELS = {"person"}


class YOLOSafetyMonitor(SafetyMonitor):
    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        hazard_labels: Optional[set] = None,
        confidence_threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.hazard_labels = hazard_labels if hazard_labels is not None else set(DEFAULT_HAZARD_LABELS)
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)

    def check(self, frame: np.ndarray) -> SafetyDecision:
        results = self.model.predict(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            label = results.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )

        hazards = [
            d
            for d in detections
            if d["label"] in self.hazard_labels and d["confidence"] >= self.confidence_threshold
        ]

        if hazards:
            top_hazard = max(hazards, key=lambda d: d["confidence"])
            return SafetyDecision(
                emergency_stop=True,
                reason=f"hazard_detected:{top_hazard['label']}",
                detections=detections,
            )

        return SafetyDecision(emergency_stop=False, reason="safe", detections=detections)
