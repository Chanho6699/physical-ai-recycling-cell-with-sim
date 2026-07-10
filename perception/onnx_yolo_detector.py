"""Generic ONNX Runtime YOLO detector (no safety judgment).

Runs weights/yolo26n.onnx through onnxruntime.InferenceSession and
returns a plain list of Detection objects. Preprocessing/postprocessing
mirrors safety/onnx_yolo_safety_monitor.py (same [1, 300, 6] NMS'd output
assumption), but this class makes no hazard/safety decisions -- it's a
building block for Real2Sim mapping and other perception use cases.
"""

from typing import List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from perception.coco_classes import COCO_CLASS_NAMES
from perception.detection_types import Detection


class ONNXYOLODetector:
    def __init__(
        self,
        model_path: str = "weights/yolo26n.onnx",
        confidence_threshold: float = 0.25,
        input_size: int = 640,
        providers: Optional[List[str]] = None,
    ):
        self.model_path = model_path
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

    def detect(self, frame: np.ndarray) -> List[Detection]:
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
                Detection(
                    label=label,
                    confidence=float(confidence),
                    # Rescaled from the resized (input_size x input_size)
                    # image back to the original frame size -- the model
                    # itself outputs coordinates in the resized space.
                    bbox_xyxy=[
                        float(x1) * scale_x,
                        float(y1) * scale_y,
                        float(x2) * scale_x,
                        float(y2) * scale_y,
                    ],
                )
            )

        return detections
