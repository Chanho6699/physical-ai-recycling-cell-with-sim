"""Expert-Replay backend adapter (see this task's chat report, "Desktop용
Expert-Replay VLA Server"). Loads a trajectory JSON file generated
OFFLINE by benchmark/generate_so101_expert_replay_trajectory.py (which
itself reuses benchmark.so101_scripted_expert.run_pick_and_place_episode()
-- the scripted expert is never re-run per HTTP request, only replayed
from the file this loads once at startup) and serves one recorded step
per /predict call for a given session, in order.

Deliberately does NOT implement vla_adapters/base_vla_adapter.py's
BaseVLAAdapter interface -- that ABC (build_model_input/
normalize_model_output) is shaped around the OLDER Panda/LIBERO
mock-action/smolvla/openvla families' single-global-model,
no-session design (see vla_server/model_loader.py's own module
docstring). Forcing this session-aware, SO-101-specific replay concept
into that ABC would mean either bending BaseVLAAdapter's contract (risk
to the 3 existing families) or adding session args it was never meant
to carry. This adapter is instead used directly by
vla_server/generic_vla_server.py's own new session-aware /predict
branch -- see that file's docstring for exactly where the two code
paths (old adapter-based vs. new session-aware) split.
"""

import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_PATH = PROJECT_ROOT / "vla_server" / "expert_replay_trajectories" / "so101_pick_place_seed1_v1.json"
TRAJECTORY_PATH_ENV_VAR = "VLA_EXPERT_REPLAY_TRAJECTORY_PATH"


class ExpertReplayAdapter:
    backend_type = "expert_replay"

    def __init__(self, config: Optional[dict] = None):
        import os

        config = config or {}
        trajectory_path = Path(
            os.environ.get(TRAJECTORY_PATH_ENV_VAR) or config.get("trajectory_path") or DEFAULT_TRAJECTORY_PATH
        )
        if not trajectory_path.exists():
            raise FileNotFoundError(
                f"ExpertReplayAdapter: trajectory file not found at {trajectory_path} -- run "
                "`.venv-vla/bin/python -m benchmark.generate_so101_expert_replay_trajectory` first, "
                f"or set {TRAJECTORY_PATH_ENV_VAR} to an existing trajectory JSON file."
            )
        with open(trajectory_path, "r", encoding="utf-8") as f:
            self._trajectory = json.load(f)
        self.trajectory_path = str(trajectory_path)
        self.model_id = self._trajectory["trajectory_id"]
        self.action_space_metadata = self._trajectory["action_space_metadata"]
        self.initial_conditions = self._trajectory["initial_conditions"]
        self.num_steps = self._trajectory["num_steps"]

    def predict(self, request, session_state) -> dict:
        """The common external interface (see this task's chat report,
        "session backend 공통 인터페이스를... request 기반 추론
        인터페이스로 정리") -- vla_server/generic_vla_server.py calls
        ONLY this method, never get_step()/check_initial_conditions()
        directly, so it never needs to know this backend replays a
        pre-generated trajectory rather than running real inference.
        `request` (the raw PredictRequest) is accepted for interface
        parity with a future real-model backend but genuinely unused
        here -- a replayed trajectory doesn't condition on
        instruction/image/robot_state. `session_state` is read-only here
        (this method never mutates it; generic_vla_server.py's own
        session_store.SessionStore.record_response() owns all state
        mutation, keeping "who's allowed to write session state" in one
        place).

        Returns {"status": "ok", "action_chunk", "trajectory_finished",
        "phase", "warning"} for a real step, or {"status": "completed"}
        (this task's own "완료 응답") once session_state.position has
        already advanced past the last recorded step -- i.e. the LAST
        real step of the trajectory itself still returns status="ok"
        with trajectory_finished=True (the caller gets one final real
        action); only a call AFTER that returns status="completed" with
        no action at all, ending the repeat-the-last-action behavior
        this task's chat report explicitly asked to remove."""
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
        """Returns the step at `position` (clamped to the last step once
        the trajectory has ended, so a caller that keeps calling past
        the end gets the SAME final action_chunk repeated rather than an
        IndexError -- trajectory_finished already told them to stop
        advancing)."""
        clamped_position = min(position, self.num_steps - 1)
        step = self._trajectory["steps"][clamped_position]
        action_chunk = [step["arm_joint_targets_rad"] + [step["gripper_target_normalized"]]]
        return {
            "step_index": step["step_index"],
            "phase": step["phase"],
            "action_chunk": action_chunk,
            "trajectory_finished": position >= self.num_steps - 1,
        }

    def check_initial_conditions(self, caller_conditions: Optional[dict]) -> Optional[str]:
        """Returns a human-readable warning string if caller_conditions
        (whatever the caller declared via POST /session/reset) does not
        match the conditions this trajectory was actually generated
        under -- None if they match or the caller declared none (this
        task's own "replay가 생성된 초기 환경 조건 metadata 포함... 조건
        불일치 시 명확한 경고 또는 오류"). Never raises -- a mismatch is
        reported, not fatal, since the replay is a stand-in for a real
        model and the caller may legitimately be testing against a
        different scenario on purpose."""
        if not caller_conditions:
            return None
        mismatches = []
        for key, expected in self.initial_conditions.items():
            if key not in caller_conditions:
                continue
            actual = caller_conditions[key]
            if isinstance(expected, list) and isinstance(actual, list):
                if len(expected) != len(actual) or any(abs(e - a) > 1e-3 for e, a in zip(expected, actual)):
                    mismatches.append(f"{key}: expected={expected} got={actual}")
            elif expected != actual:
                mismatches.append(f"{key}: expected={expected} got={actual}")
        if mismatches:
            return "initial_conditions mismatch vs. trajectory generation conditions: " + "; ".join(mismatches)
        return None
