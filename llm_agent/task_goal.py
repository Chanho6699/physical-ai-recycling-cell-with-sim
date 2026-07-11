from dataclasses import dataclass, field


@dataclass
class TaskGoal:
    action: str
    target_object: str
    target_bin: str
    instruction: str
    constraints: dict = field(default_factory=dict)
