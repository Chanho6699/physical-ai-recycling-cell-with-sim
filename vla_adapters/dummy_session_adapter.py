"""Trivial session-aware "dummy" backend -- see this task's chat report,
"Desktop용 Expert-Replay VLA Server". Speaks the EXACT SAME
action_chunk/action_space_metadata/trajectory_finished shape as
vla_adapters/expert_replay_adapter.py (same get_step()/
check_initial_conditions() interface) but returns a fixed all-zero
action_chunk instead of a replayed trajectory -- purpose-built so a test
can assert "dummy와 expert_replay의 response schema 일치" without either
adapter needing to know about the other. NOT the same object as the
existing vla_adapters/mock_vla_adapter.py (MockVLAAdapter), which
implements the OLDER, non-session BaseVLAAdapter interface for the
Panda/LIBERO families -- this is a new, separate, minimal stand-in for
the session-aware SO-101 protocol only.
"""

from typing import Optional

from robot_sim.so101_pybullet_backend import ARM_JOINT_NAMES

DUMMY_NUM_STEPS = 68  # matches the generated expert_replay trajectory's own step count, so both backends finish at the same point in a side-by-side comparison


class DummySessionAdapter:
    backend_type = "dummy"

    def __init__(self, config: Optional[dict] = None):
        self.model_id = "dummy-zero-action-v0"
        self.action_space_metadata = {
            "joint_order": list(ARM_JOINT_NAMES) + ["gripper"],
            "arm_units": "radians_absolute_joint_target",
            "gripper_units": "normalized_0_1",
            "gripper_convention": "0.0 = closed, 1.0 = open",
            "action_dim": len(ARM_JOINT_NAMES) + 1,
            "chunk_size": 1,
        }
        self.initial_conditions = {}
        self.num_steps = DUMMY_NUM_STEPS

    def predict(self, request, session_state) -> dict:
        """Same external interface as ExpertReplayAdapter.predict() --
        see that method's own docstring. `request` genuinely unused
        here too (dummy always returns the same fixed action)."""
        if session_state.position >= self.num_steps:
            return {"status": "completed"}

        step = self.get_step(session_state.position)
        warning = self.check_initial_conditions(session_state.initial_conditions)
        return {
            "status": "ok",
            "action_chunk": step["action_chunk"],
            "trajectory_finished": step["trajectory_finished"],
            "phase": step["phase"],
            "warning": warning,
        }

    def get_step(self, position: int) -> dict:
        clamped_position = min(position, self.num_steps - 1)
        action_chunk = [[0.0] * len(ARM_JOINT_NAMES) + [1.0]]  # arm at all-zero joint targets, gripper open
        return {
            "step_index": clamped_position,
            "phase": "dummy",
            "action_chunk": action_chunk,
            "trajectory_finished": position >= self.num_steps - 1,
        }

    def check_initial_conditions(self, caller_conditions: Optional[dict]) -> Optional[str]:
        return None  # dummy has no real initial conditions to mismatch against
