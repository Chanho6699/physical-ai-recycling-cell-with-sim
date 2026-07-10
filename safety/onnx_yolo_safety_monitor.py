"""ONNX Runtime-based YOLO SafetyMonitor (v0).

Runs weights/yolo26n.onnx directly through onnxruntime.InferenceSession
(no Ultralytics YOLO wrapper, no PyTorch at inference time) and flags
configured hazard labels (default: "person"). Implements the same
SafetyMonitor interface as MockSafetyMonitor/YOLOSafetyMonitor, so it is
a drop-in replacement in the safety gate.

Assumes the export used here (Ultralytics end-to-end ONNX export) already
applies NMS internally: output shape [1, 300, 6], each row
[x1, y1, x2, y2, confidence, class_id] in the resized (input_size x
input_size) image space -- confirmed via benchmark/run_yolo_onnx_smoke_test.py.
"""

from typing import List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from safety.safety_monitor import SafetyMonitor
from safety.safety_types import SafetyDecision

DEFAULT_HAZARD_LABELS = {"person"}

# Ultralytics COCO 80-class names, indexed by class_id (person == 0).
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class ONNXRuntimeYOLOSafetyMonitor(SafetyMonitor):
    def __init__(
        self,
        model_path: str = "weights/yolo26n.onnx",
        hazard_labels: Optional[set] = None,
        confidence_threshold: float = 0.5,
        input_size: int = 640,
        providers: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.hazard_labels = hazard_labels if hazard_labels is not None else set(DEFAULT_HAZARD_LABELS)
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.providers = providers if providers is not None else ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = np.array(Image.fromarray(frame).resize((self.input_size, self.input_size)))
        normalized = resized.astype(np.float32) / 255.0
        chw = normalized.transpose(2, 0, 1)
        return chw[np.newaxis, ...]

    def check(self, frame: np.ndarray) -> SafetyDecision:
        original_height, original_width = frame.shape[:2]
        scale_x = original_width / self.input_size
        scale_y = original_height / self.input_size

        input_tensor = self._preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        rows = outputs[0]  # drop batch dimension -> [300, 6]

        detections = []
        for x1, y1, x2, y2, confidence, class_id in rows:
            if confidence < self.confidence_threshold:
                continue

            class_index = int(class_id)
            label = COCO_CLASS_NAMES[class_index] if class_index < len(COCO_CLASS_NAMES) else str(class_index)

            detections.append(
                {
                    "label": label,
                    "confidence": float(confidence),
                    # Rescaled from the resized (input_size x input_size)
                    # image back to the original frame size -- the model
                    # itself outputs coordinates in the resized space.
                    "bbox_xyxy": [
                        float(x1) * scale_x,
                        float(y1) * scale_y,
                        float(x2) * scale_x,
                        float(y2) * scale_y,
                    ],
                }
            )

        hazards = [d for d in detections if d["label"] in self.hazard_labels]

        if hazards:
            top_hazard = max(hazards, key=lambda d: d["confidence"])
            return SafetyDecision(
                emergency_stop=True,
                reason=f"hazard_detected:{top_hazard['label']}",
                detections=detections,
            )

        return SafetyDecision(emergency_stop=False, reason="safe", detections=detections)
