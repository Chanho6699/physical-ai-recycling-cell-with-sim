"""In-process session state for the session-aware VLA backends
(model_family in ("dummy", "expert_replay") -- see
vla_server/generic_vla_server.py's own docstring, "Desktop용
Expert-Replay VLA Server"). Deliberately separate from
vla_server/model_loader.py's own global `_state` dict, which serves an
entirely different concept (ONE model instance for the whole process,
no per-caller isolation) that the older mock-action/smolvla/openvla
families still use unchanged.

Nothing here is SO-101-specific -- this module only tracks position/
idempotency/ordering per session_id; vla_adapters/expert_replay_adapter.py
and vla_adapters/dummy_session_adapter.py own the actual trajectory
content and action-space semantics.
"""

import threading
import time
from typing import Any, Dict, Optional


class SessionState:
    def __init__(self, session_id: str, initial_conditions: Optional[dict] = None):
        self.session_id = session_id
        self.position = 0
        self.initial_conditions = initial_conditions or {}
        self.created_at = time.time()
        self.last_request_id: Optional[str] = None
        self.last_response: Optional[dict] = None
        self.trajectory_finished = False
        # True once a backend.predict() call has returned status="completed"
        # for this session (see this task's chat report, "완료 응답") --
        # once set, generic_vla_server.py's own session-aware /predict
        # branch short-circuits straight to a fresh completed response for
        # ANY new request_id, without asking the backend again or
        # validating step order. Reset (a brand-new SessionState) always
        # starts this False.
        self.completed = False


class SessionStore:
    """One instance per server process (module-level singleton in
    generic_vla_server.py, same lifetime as model_loader's own _state --
    reset when the process restarts, exactly matching this task's own
    "서버 재시작 후 정상 초기화" requirement: nothing here is persisted
    to disk)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionState] = {}

    def reset_session(self, session_id: str, initial_conditions: Optional[dict] = None) -> SessionState:
        with self._lock:
            state = SessionState(session_id, initial_conditions)
            self._sessions[session_id] = state
            return state

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id)
            return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._sessions.get(session_id)

    def num_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def record_response(
        self, session_id: str, request_id: str, response: dict, advance: bool, trajectory_finished: bool,
        completed: bool = False,
    ) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            state.last_request_id = request_id
            state.last_response = response
            state.trajectory_finished = trajectory_finished
            if completed:
                state.completed = True
            if advance:
                state.position += 1


class StepOrderError(Exception):
    """Raised when a request's step_index does not match what this
    session expects next -- caught by generic_vla_server.py's own
    session-aware /predict branch and turned into a structured
    status="error" response (never a raw 500)."""

    def __init__(self, message: str, expected_step_index: int, received_step_index: Any):
        super().__init__(message)
        self.expected_step_index = expected_step_index
        self.received_step_index = received_step_index


def resolve_step_outcome(state: SessionState, request_id: Optional[str], step_index: Optional[int]):
    """Pure decision function (no I/O, easy to unit-test): given a
    session's current state and an incoming request's request_id/
    step_index, decides whether this is (a) an idempotent replay of the
    immediately-preceding request (same request_id as last time -> True
    is returned so the caller returns the cached response unchanged,
    position NOT advanced again), (b) the expected next step (position
    == step_index, or step_index omitted -> proceed, advance), or (c) an
    out-of-order request (raises StepOrderError).

    `step_index` is optional on the wire (this task's own "step_index
    또는 chunk_index" wording) -- when the caller omits it entirely,
    ordering is not checked and the session's own internal position
    counter is trusted as the sole source of truth."""
    if request_id is not None and request_id == state.last_request_id:
        return "replay"
    if step_index is not None and step_index != state.position:
        raise StepOrderError(
            f"session {state.session_id!r} expected step_index={state.position}, got {step_index}",
            expected_step_index=state.position, received_step_index=step_index,
        )
    return "advance"
