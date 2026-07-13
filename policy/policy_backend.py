"""PolicyBackend is a naming alias for BasePolicy (see policy/base_policy.py).

BasePolicy already is the hardware-portable policy interface this
project needs (reset() / predict_action(PolicyInput) -> PolicyOutput),
and DummyOpenVLAPolicy/FastAPIVLAPolicyClient already implement it
identically -- see run_full_recycling_cell_demo.py's --policy-backend
{local-dummy,fastapi-dummy} switch, which constructs either one behind
this same interface. This module exists only so robot_core/vision/
safety's *_backend.py interfaces (RobotBackend, CameraBackend,
SafetySupervisor) and policy's line up under one consistent naming
scheme, without duplicating BasePolicy's ABC or requiring existing
policy classes to change their base class.

A future real OpenVLA client only needs to implement BasePolicy
(equivalently, PolicyBackend) to be a drop-in --policy-backend choice.
"""

from policy.base_policy import BasePolicy as PolicyBackend

__all__ = ["PolicyBackend"]
