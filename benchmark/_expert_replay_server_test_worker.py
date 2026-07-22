"""Subprocess worker for benchmark/test_expert_replay_vla_server.py.

vla_server/generic_vla_server.py resolves model_family ONCE at import
time (module-level _MODEL_FAMILY/_SESSION_BACKEND/_SESSION_STORE) -- a
single process cannot exercise two different model_family values, and
"server restart reinitializes cleanly" can only be tested across two
genuinely separate process launches. Every scenario below therefore
runs in its OWN subprocess (see the parent test script), using
fastapi.testclient.TestClient against the already-built `app` instead
of actually binding a port -- this is a same-process, in-memory HTTP
call, not a real network round-trip, but exercises the exact same
FastAPI routing/Pydantic validation a real request would.

Prints one JSON object to stdout (the scenario's result) -- the parent
script asserts on it. Never prints anything else to stdout (all
diagnostic prints, if any, go to stderr) so parent-side json.loads()
never has to parse around noise.
"""

import json
import sys

INSTRUCTION = "pick up the object and place it in the bin"


def _client():
    from fastapi.testclient import TestClient

    from vla_server.generic_vla_server import app

    return TestClient(app)


def scenario_health():
    client = _client()
    resp = client.get("/health")
    return {"status_code": resp.status_code, "body": resp.json()}


def scenario_reset():
    client = _client()
    resp = client.post("/session/reset", json={"session_id": "s1"})
    return {"status_code": resp.status_code, "body": resp.json()}


def scenario_sequential_predict():
    client = _client()
    client.post("/session/reset", json={"session_id": "s1"})
    results = []
    for i in range(3):
        resp = client.post("/predict", json={
            "instruction": INSTRUCTION, "session_id": "s1", "request_id": f"req-{i}", "step_index": i,
        })
        results.append(resp.json())
    return {"results": results}


def scenario_duplicate_request_id():
    client = _client()
    client.post("/session/reset", json={"session_id": "s1"})
    first = client.post("/predict", json={
        "instruction": INSTRUCTION, "session_id": "s1", "request_id": "req-0", "step_index": 0,
    }).json()
    second = client.post("/predict", json={
        "instruction": INSTRUCTION, "session_id": "s1", "request_id": "req-0", "step_index": 0,
    }).json()
    return {"first": first, "second": second}


def scenario_wrong_step_order():
    client = _client()
    client.post("/session/reset", json={"session_id": "s1"})
    resp = client.post("/predict", json={
        "instruction": INSTRUCTION, "session_id": "s1", "request_id": "req-skip", "step_index": 5,
    })
    return {"status_code": resp.status_code, "body": resp.json()}


def scenario_trajectory_finished():
    client = _client()
    health = client.get("/health").json()
    num_steps = health["trajectory_num_steps"]
    client.post("/session/reset", json={"session_id": "s1"})
    last = None
    for i in range(num_steps):
        last = client.post("/predict", json={
            "instruction": INSTRUCTION, "session_id": "s1", "request_id": f"req-{i}", "step_index": i,
        }).json()
    # First call after the trajectory's last real step -- backend.predict()
    # itself detects position >= num_steps and returns status="completed".
    completed_first = client.post("/predict", json={
        "instruction": INSTRUCTION, "session_id": "s1", "request_id": f"req-{num_steps}", "step_index": num_steps,
    }).json()
    # Retry the SAME request_id that just got "completed" -- idempotency
    # rule applies unconditionally (this task's own "완료 이후 같은
    # request_id 재전송").
    completed_retry_same_request_id = client.post("/predict", json={
        "instruction": INSTRUCTION, "session_id": "s1", "request_id": f"req-{num_steps}", "step_index": num_steps,
    }).json()
    # A BRAND NEW request_id, arbitrary step_index -- must ALSO get
    # completed, with no step-order validation at all (this task's own
    # "완료 이후 새로운 request_id로 다음 step 요청").
    completed_new_request_id = client.post("/predict", json={
        "instruction": INSTRUCTION, "session_id": "s1", "request_id": "req-way-later", "step_index": 9999,
    }).json()
    return {
        "num_steps": num_steps, "last": last,
        "completed_first": completed_first,
        "completed_retry_same_request_id": completed_retry_same_request_id,
        "completed_new_request_id": completed_new_request_id,
    }


def scenario_two_sessions_independent():
    client = _client()
    client.post("/session/reset", json={"session_id": "sA"})
    client.post("/session/reset", json={"session_id": "sB"})
    for i in range(3):
        client.post("/predict", json={"instruction": INSTRUCTION, "session_id": "sA", "request_id": f"a-{i}", "step_index": i})
    client.post("/predict", json={"instruction": INSTRUCTION, "session_id": "sB", "request_id": "b-0", "step_index": 0})

    next_a = client.post("/predict", json={"instruction": INSTRUCTION, "session_id": "sA", "request_id": "a-3", "step_index": 3}).json()
    next_b = client.post("/predict", json={"instruction": INSTRUCTION, "session_id": "sB", "request_id": "b-1", "step_index": 1}).json()
    return {"next_a_step_index": next_a["step_index"], "next_b_step_index": next_b["step_index"]}


def scenario_schema_probe():
    client = _client()
    client.post("/session/reset", json={"session_id": "s1"})
    resp = client.post("/predict", json={
        "instruction": INSTRUCTION, "session_id": "s1", "request_id": "req-0", "step_index": 0,
    })
    return {"status_code": resp.status_code, "body": resp.json()}


def scenario_restart_reinit():
    """Only the SECOND half of the "server restart" test -- the parent
    script runs THIS scenario in a fresh subprocess without ever calling
    /session/reset for session_id="s1" first, simulating a client that
    (wrongly assumes state survived a server restart and) just resumes
    predicting -- session_store.py's SessionStore is a fresh, empty dict
    in this new process regardless of what a PRIOR process did, so this
    must come back at step_index=0."""
    client = _client()
    resp = client.post("/predict", json={
        "instruction": INSTRUCTION, "session_id": "s1", "request_id": "req-0", "step_index": 0,
    })
    return {"status_code": resp.status_code, "body": resp.json()}


SCENARIOS = {
    "health": scenario_health,
    "reset": scenario_reset,
    "sequential_predict": scenario_sequential_predict,
    "duplicate_request_id": scenario_duplicate_request_id,
    "wrong_step_order": scenario_wrong_step_order,
    "trajectory_finished": scenario_trajectory_finished,
    "two_sessions_independent": scenario_two_sessions_independent,
    "schema_probe": scenario_schema_probe,
    "restart_reinit": scenario_restart_reinit,
}


def main() -> None:
    scenario_name = sys.argv[1]
    result = SCENARIOS[scenario_name]()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
