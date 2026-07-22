"""Standalone test script (plain asserts + PASS/FAIL summary, matching
this project's existing convention -- see e.g.
benchmark/test_smolvla_libero_action_adapter.py -- not pytest) for
vla_server's new Expert-Replay VLA Server (see this task's chat report,
"Desktop용 Expert-Replay VLA Server").

Each scenario runs in its OWN subprocess (see
benchmark/_expert_replay_server_test_worker.py's own docstring for why:
model_family is resolved once at generic_vla_server.py import time, so
one process can only ever exercise one model_family).

Run:
  .venv-vla/bin/python -m benchmark.test_expert_replay_vla_server
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
WORKER = "benchmark._expert_replay_server_test_worker"

results = []


def run_scenario(scenario: str, model_family: str, extra_env: dict = None) -> dict:
    env = {"VLA_MODEL_FAMILY": model_family}
    if extra_env:
        env.update(extra_env)
    import os

    full_env = dict(os.environ)
    full_env.update(env)
    proc = subprocess.run(
        [PYTHON, "-m", WORKER, scenario], cwd=str(PROJECT_ROOT), env=full_env,
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scenario {scenario!r} (family={model_family!r}) subprocess failed:\n{proc.stderr}")
    # stdout may contain pybullet/warning noise printed by imports before
    # the worker's own final print(json.dumps(...)) -- only the LAST
    # non-empty line is the actual result.
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def check_common_interface() -> None:
    """Directly inspects both adapter CLASSES (no subprocess/HTTP needed)
    -- confirms dummy and expert_replay both expose the same
    predict(request, session_state) method signature this task's chat
    report asked for (see vla_server/generic_vla_server.py's own
    _handle_session_predict(), which calls ONLY this method)."""
    import inspect

    from vla_adapters.dummy_session_adapter import DummySessionAdapter
    from vla_adapters.expert_replay_adapter import ExpertReplayAdapter

    expert_params = list(inspect.signature(ExpertReplayAdapter.predict).parameters.keys())
    dummy_params = list(inspect.signature(DummySessionAdapter.predict).parameters.keys())
    check("common interface: both adapters define predict(self, request, session_state)",
          expert_params == dummy_params == ["self", "request", "session_state"],
          detail=f"expert={expert_params} dummy={dummy_params}")


def main() -> None:
    # 0. common predict(request, session_state) interface
    check_common_interface()

    # 1. health
    health = run_scenario("health", "expert_replay")
    check("health: status_code 200", health["status_code"] == 200)
    body = health["body"]
    check("health: model_family=expert_replay", body["model_family"] == "expert_replay")
    check("health: backend_type=expert_replay", body["backend_type"] == "expert_replay")
    check("health: action_space_metadata present", "joint_order" in body["action_space_metadata"])
    check("health: trajectory_num_steps=68", body["trajectory_num_steps"] == 68)

    # 2. reset
    reset = run_scenario("reset", "expert_replay")
    check("reset: status_code 200", reset["status_code"] == 200)
    rbody = reset["body"]
    check("reset: status=reset", rbody["status"] == "reset")
    check("reset: session_id echoed", rbody["session_id"] == "s1")
    check("reset: trajectory_num_steps=68", rbody["trajectory_num_steps"] == 68)
    check("reset: no warning (no initial_conditions declared)", rbody["warning"] is None)

    # 3. normal sequential predict
    seq = run_scenario("sequential_predict", "expert_replay")["results"]
    check("sequential: 3 responses", len(seq) == 3)
    check("sequential: step_index increments 0,1,2", [r["step_index"] for r in seq] == [0, 1, 2])
    check("sequential: all status=ok", all(r["status"] == "ok" for r in seq))
    check("sequential: action_chunk shape (1,6) each step", all(len(r["action_chunk"]) == 1 and len(r["action_chunk"][0]) == 6 for r in seq))
    check("sequential: not finished yet (68-step trajectory, only 3 steps in)", all(not r["trajectory_finished"] for r in seq))
    check("sequential: backend_type=expert_replay on every response", all(r["backend_type"] == "expert_replay" for r in seq))
    check("sequential: distinct action_chunks across steps (not a static replay)", seq[0]["action_chunk"] != seq[1]["action_chunk"])

    # 4. same request resent -> identical response
    dup = run_scenario("duplicate_request_id", "expert_replay")
    first, second = dup["first"], dup["second"]
    first_no_latency = {k: v for k, v in first.items() if k != "latency_ms"}
    second_no_latency = {k: v for k, v in second.items() if k != "latency_ms"}
    check("duplicate request_id: identical response (excl. latency_ms)", first_no_latency == second_no_latency,
          detail=f"first={first_no_latency} second={second_no_latency}")

    # 5. wrong step order
    wrong = run_scenario("wrong_step_order", "expert_replay")
    check("wrong step order: HTTP 200 (structured error, not a crash)", wrong["status_code"] == 200)
    wbody = wrong["body"]
    check("wrong step order: status=error", wbody["status"] == "error")
    check("wrong step order: failure_reason mentions step_order_mismatch", "step_order_mismatch" in (wbody["failure_reason"] or ""))
    check("wrong step order: action_chunk is None", wbody["action_chunk"] is None)

    # 6. trajectory finished -> safe "completed" response (no repeated last action)
    finished = run_scenario("trajectory_finished", "expert_replay")
    check("trajectory finished: last real step still returns status=ok with an action", finished["last"]["status"] == "ok" and finished["last"]["action_chunk"] is not None)
    check("trajectory finished: last real step trajectory_finished=True", finished["last"]["trajectory_finished"] is True)
    cf = finished["completed_first"]
    check("completed: status=completed", cf["status"] == "completed")
    check("completed: trajectory_finished=True", cf["trajectory_finished"] is True)
    check("completed: action_chunk=None (never the last action repeated)", cf["action_chunk"] is None)
    check("completed: action=None", cf["action"] is None)
    check("completed: failure_reason=None", cf["failure_reason"] is None)
    retry = finished["completed_retry_same_request_id"]
    cf_no_latency = {k: v for k, v in cf.items() if k != "latency_ms"}
    retry_no_latency = {k: v for k, v in retry.items() if k != "latency_ms"}
    check("completed: same request_id retry returns identical completed response", cf_no_latency == retry_no_latency)
    new_req = finished["completed_new_request_id"]
    check("completed: new request_id (arbitrary step_index=9999) still returns completed, not step_order_mismatch", new_req["status"] == "completed")
    check("completed: new request_id after completion still returns action_chunk=None", new_req["action_chunk"] is None)

    # 7. two sessions independence
    two = run_scenario("two_sessions_independent", "expert_replay")
    check("two sessions: sA at step_index=3 after 4 calls", two["next_a_step_index"] == 3)
    check("two sessions: sB at step_index=1 after 2 calls", two["next_b_step_index"] == 1)
    check("two sessions: sA/sB positions differ (independent)", two["next_a_step_index"] != two["next_b_step_index"])

    # 8. dummy vs expert_replay response schema parity
    dummy_probe = run_scenario("schema_probe", "dummy")
    expert_probe = run_scenario("schema_probe", "expert_replay")
    dummy_keys = set(dummy_probe["body"].keys())
    expert_keys = set(expert_probe["body"].keys())
    check("schema parity: identical top-level response keys (dummy vs expert_replay)", dummy_keys == expert_keys,
          detail=f"dummy_only={dummy_keys - expert_keys} expert_only={expert_keys - dummy_keys}")
    dummy_metadata_keys = set(dummy_probe["body"]["action_space_metadata"].keys())
    expert_metadata_keys = set(expert_probe["body"]["action_space_metadata"].keys())
    check("schema parity: identical action_space_metadata keys", dummy_metadata_keys == expert_metadata_keys)
    check("schema parity: identical action_chunk shape", (
        len(dummy_probe["body"]["action_chunk"]) == len(expert_probe["body"]["action_chunk"]) == 1
        and len(dummy_probe["body"]["action_chunk"][0]) == len(expert_probe["body"]["action_chunk"][0]) == 6
    ))

    # 9. server restart reinitializes cleanly -- two ENTIRELY SEPARATE
    # subprocesses, second one never resets session "s1" first.
    run_scenario("sequential_predict", "expert_replay")  # first "server" advances s1 to position 3, then exits
    restarted = run_scenario("restart_reinit", "expert_replay")  # brand-new process/import, fresh SessionStore
    check("restart: fresh process starts session at step_index=0 (no leaked state)", restarted["body"]["step_index"] == 0)
    check("restart: fresh process predict succeeds (clean init)", restarted["status_code"] == 200 and restarted["body"]["status"] == "ok")

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Total: {passed}/{len(results)} passed")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
