from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyDecision:
    emergency_stop: bool
    reason: str = "safe"
    detections: list[dict[str, Any]] = field(default_factory=list)
    severity: str = "none"
