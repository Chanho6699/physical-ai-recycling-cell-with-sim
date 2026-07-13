"""SafetySupervisor v0 -- the Safety Pause/Resume decision boundary,
kept outside the VLA policy.

Core structure this project is built around (see docs/architecture.md):

    Policy proposes an action every step.
    SafetySupervisor decides whether that action may be applied.
    RobotBackend executes only what SafetySupervisor allows through.

SafetySupervisor does NOT do hand/hazard *detection* itself -- callers
still run their own SafetyMonitor (MockHandIntrusionMonitor,
ExternalCameraHandSafetyMonitor, ONNXRuntimeYOLOSafetyMonitor, ...)
through a SafetyGate and pass the resulting hand_detected boolean into
step(). SafetySupervisor only owns the pause/resume *state machine* --
running -> paused_by_safety -> (resuming) -> running -- that decides,
from that boolean, whether the current step counts as safe to act on.

v0 is a narrow, mechanical extraction of the state machine that used to
be inlined directly in run_full_recycling_cell_demo.py's control loop
(same states/events/counters, moved here unchanged) -- it does not yet
absorb the separate hard-block SafetyGate path (--safety-monitor
mock/onnx, which still ends the episode with task_status=blocked_by_safety
rather than pausing/resuming it) or fuse multiple hand-intrusion sensors.
See docs/hardware_portability.md for what a more complete
SafetySupervisor would eventually own.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SafetySupervisorState:
    safety_state: str = "running"
    pause_count: int = 0
    resume_count: int = 0
    paused_steps: int = 0
    hand_intrusion_events: int = 0
    stable_clear_steps: int = 0


class SafetySupervisor:
    def __init__(self, resume_stable_steps: int = 3):
        self.resume_stable_steps = resume_stable_steps
        self.state = SafetySupervisorState()

    def step(
        self,
        hand_detected: bool,
        reason: Optional[str] = None,
        still_paused_reason: Optional[str] = None,
    ) -> Optional[dict]:
        """Advances the pause/resume state machine by exactly one
        control loop step and returns the event dict to record this
        step (safety_pause / safety_still_paused / safety_resume), or
        None if nothing changed (safety_state stayed "running" and
        hand_detected is False -- the normal, unpaused case where the
        caller should go on to call the policy and apply its action).

        `reason` is used for the safety_pause event (and safety_still_paused
        while hand_detected stays True); `still_paused_reason` is used
        for safety_still_paused once hand_detected has already gone
        False but resume stability hasn't been reached yet -- these can
        differ because "why did we pause" (e.g. "hand_in_workspace") and
        "why are we still not resuming" are conceptually the same event
        type but were reported with different reason strings even in v0.
        """
        state = self.state

        if hand_detected:
            state.stable_clear_steps = 0
            if state.safety_state == "running":
                state.pause_count += 1
                state.hand_intrusion_events += 1
                event_type = "safety_pause"
            else:
                event_type = "safety_still_paused"
            state.safety_state = "paused_by_safety"
            state.paused_steps += 1
            return {
                "event_type": event_type,
                "reason": reason or "hand_intrusion",
                "robot_action_applied": False,
                "hand_detected": True,
            }

        if state.safety_state in ("paused_by_safety", "resuming"):
            state.stable_clear_steps += 1
            state.paused_steps += 1
            if state.stable_clear_steps >= self.resume_stable_steps:
                state.safety_state = "running"
                state.resume_count += 1
                event = {
                    "event_type": "safety_resume",
                    "reason": "hand_cleared",
                    "stable_clear_steps": state.stable_clear_steps,
                    "robot_action_applied": False,
                    "hand_detected": False,
                }
                state.stable_clear_steps = 0
                return event

            state.safety_state = "resuming"
            return {
                "event_type": "safety_still_paused",
                "reason": still_paused_reason or reason or "hand_intrusion",
                "stable_clear_steps": state.stable_clear_steps,
                "robot_action_applied": False,
                "hand_detected": False,
            }

        return None

    def is_running(self) -> bool:
        return self.state.safety_state == "running"

    def summary(self) -> dict:
        return {
            "safety_pause_count": self.state.pause_count,
            "safety_resume_count": self.state.resume_count,
            "paused_steps": self.state.paused_steps,
            "hand_intrusion_events": self.state.hand_intrusion_events,
            "final_safety_state": self.state.safety_state,
        }
