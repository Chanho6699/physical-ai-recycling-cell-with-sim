"""NativePolicyAction -- a checkpoint's raw network output, AFTER its own
official postprocessor (unnormalization) has already run, BEFORE this
project's semantic ActionAdapter has interpreted it.

Distinguishing "raw model tensor" from "native policy action" matters
because normalization can otherwise be silently skipped or duplicated:
vla_server/model_loader.py must run the checkpoint's own official
postprocessor (e.g. LeRobot's UnnormalizerProcessorStep, loaded via
PolicyProcessorPipeline.from_pretrained(..., "policy_postprocessor.json"))
on the model's raw tensor output BEFORE building this type -- a
NativePolicyAction's `values` are in the checkpoint's own native action
space/units (e.g. LIBERO's Box(-1, 1) 6D-EE-delta+gripper), not
arbitrary network activations. policy_semantics/adapters/*.py's
ActionAdapter implementations take this as input and are the ones that
know the physical (meters/radians) meaning of `values`, using the
manifest's declared scale/frame/rotation-representation -- never the
raw model tensor directly.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class NativePolicyAction:
    values: List[float]
    source_policy: str
    postprocessor_used: bool  # True only if the checkpoint's own official
    # postprocessor actually ran -- False means `values` are the model's
    # raw un-postprocessed tensor, which ActionAdapter implementations
    # must refuse rather than silently interpret as if it were unnormalized.
    metadata: Dict[str, Any] = field(default_factory=dict)
