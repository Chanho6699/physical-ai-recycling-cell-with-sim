"""Selects a recyclable-object candidate from YOLO detections (v0).

Only COCO's "bottle" and "cup" labels are treated as recyclable objects
for now -- no custom-trained recycling classes yet.
"""

from typing import List, Optional, Tuple

from perception.detection_types import Detection

RECYCLABLE_LABEL_TO_SIM_TYPE = {
    "bottle": "plastic_bottle",
    "cup": "plastic_cup",
}


class RecyclableObjectMapper:
    def select_best_recyclable(self, detections: List[Detection]) -> Optional[Tuple[Detection, str]]:
        candidates = [d for d in detections if d.label in RECYCLABLE_LABEL_TO_SIM_TYPE]
        if not candidates:
            return None

        best = max(candidates, key=lambda d: d.confidence)
        return best, RECYCLABLE_LABEL_TO_SIM_TYPE[best.label]

    def select_recyclable_by_target(
        self,
        detections: List[Detection],
        target_object: str,
    ) -> Optional[Tuple[Detection, str]]:
        candidates = [
            d for d in detections if RECYCLABLE_LABEL_TO_SIM_TYPE.get(d.label) == target_object
        ]
        if not candidates:
            return None

        best = max(candidates, key=lambda d: d.confidence)
        return best, RECYCLABLE_LABEL_TO_SIM_TYPE[best.label]
