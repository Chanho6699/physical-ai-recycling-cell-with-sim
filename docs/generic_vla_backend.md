# Generic VLA Backend (v0)

## Why SmolVLA instead of OpenVLA 7B, for now

The [Colab OpenVLA spike](colab_vla_server_spike.md) proved the
plumbing (tunnel, `/health`, `/load_model`, lazy loading, graceful
failure) works, but actually loading OpenVLA-7B kept hitting real
environment limits: multi-GB shard downloads, a Colab GPU tier that
isn't guaranteed to have enough VRAM for a 7B model, and sessions that
can disconnect mid-load. None of that is a bug to fix -- it's the
actual cost of a 7B model on free/shared compute. SmolVLA (from
LeRobot) is meant to run on much more modest hardware, so it's a better
first real (non-mock) model to get an actual `action_len == 7` from
than OpenVLA is right now. OpenVLA-7B stays on the roadmap, not
abandoned.

## Why it's still designed to swap back to OpenVLA easily

Nothing in this backend is SmolVLA-specific at the server/client
boundary. The local robot control loop only ever talks to
`RealVLAPolicyClient`, which only ever speaks one fixed `/predict`
request/response schema -- it has no idea whether the model behind
that schema is SmolVLA, OpenVLA, or a mock. What actually changes
between models lives entirely inside a `model_family`-keyed adapter +
loader pair:

```text
vla_adapters/<family>_adapter.py   data transforms only (request -> model input,
                                    raw output -> normalized action) -- never
                                    touches a model object or the network
vla_server/model_loader.py         owns model lifecycle + the actual inference
                                    call, dispatched by model_family
vla_server/model_registry.py       model_family -> adapter class lookup
```

Switching the active model later is: set `VLA_MODEL_FAMILY=openvla`
(or add a new family the same way), point `VLA_MODEL_ID_OR_PATH` at the
right checkpoint, and (if OpenVLA's action space gets verified) fill in
`OpenVLAActionAdapter._decode_openvla_action()`. The server, the
client, and the local robot control loop do not change.

## The common `/predict` schema

Every model family behind `vla_server/generic_vla_server.py` is called
through the exact same request/response shape (`RealVLAPolicyClient`
already builds this, unchanged from earlier turns):

```json
// request
{
  "instruction": "...", "robot_state": {...}, "task_goal": {...},
  "target_object_position": [...], "bin_position": [...],
  "step_index": 12, "phase": "move_to_object",
  "observation_source": "wrist",
  "visual_observation": {"object_visible": true, ...},
  "image": {"encoding": "jpg_base64", "shape": [224, 224, 3], "data": "..."}
}
// response (success)
{"action": [dx, dy, dz, droll, dpitch, dyaw, gripper], "phase": "...", "done": false,
 "info": {"model_family": "...", "adapter_used": "...", "raw_model_output_available": true}}
// response (refused -- HTTP 503, structured detail)
{"detail": {"error": "model_not_loaded" | "openvla_action_adapter_required" | "...",
            "model_family": "...", "reason": "..."}}
```

The local `robot control loop never knows the model name` -- it always
sends this request and expects either a normalized 7-DoF action or a
structured error it can fall back from.

## Adapter structure

```text
request dict -> adapter.build_model_input() -> model_loader.run_inference()
             -> raw_output -> adapter.normalize_model_output() -> {action, phase, done, info}
```

`normalize_model_output()` is the safety-relevant boundary: it must
return `action=None` (never a fabricated 7-DoF guess) whenever it
isn't confident the raw output maps cleanly onto
`[dx, dy, dz, droll, dpitch, dyaw, gripper]`. `generic_vla_server.py`
turns `action=None` into a structured HTTP 503 so
`RealVLAPolicyClient` falls back instead of executing a guess -- this
runs in addition to (not instead of) `RealVLAPolicyClient`'s own
client-side `policy/vla_action_postprocessor.py`; server-side adapters
clip/validate too, as defense in depth.

## Role of each `model_family`

- **`mock-action`**: no real model. Wraps the same `DummyOpenVLAPolicy`
  phase engine every other mock path in this repo already uses (
  `local-dummy`, `fastapi-dummy`, `real-vla-compatible-mock`,
  `colab_vla_server.py`'s `mock-action`). Marked `loaded` immediately
  at server startup -- there's nothing to download, so `POST
  /load_model` is never required for this family to work.
- **`smolvla`**: `SmolVLAActionAdapter` handles several possible raw
  output shapes (a plain 7-number action; `{"action": [...]}`;
  chunked/action-horizon `{"actions": [[...], ...]}` selected by
  `step_index`; numpy arrays/torch tensors anywhere in that structure)
  and rejects anything else with a structured, fallback-triggering
  reason. `vla_server/model_loader.py` tries LeRobot's own policy
  loader first, then a plain `transformers` `AutoModelForImageTextToText`
  load as a fallback; either is best-effort (neither library is pinned
  by this repo) -- a missing/incompatible install is an expected,
  gracefully-handled `model_status=load_failed`, not a crash.
- **`openvla`**: `OpenVLAActionAdapter` **always** returns
  `action=None`, `project_action_available=false`,
  `reason="openvla_action_adapter_required"`, regardless of whether
  the model is loaded or what its raw output looks like -- OpenVLA's
  own action space/normalization (`unnorm_key`, gripper convention,
  frame convention) has not been verified against this project's
  schema. The loader can still attempt a real load (mirrors
  `openvla_server_real/colab_vla_server.py`'s approach) so the family
  is genuinely swappable later, but a working load is explicitly not
  required for this v0 -- see `vla_adapters/openvla_adapter.py`'s
  `_decode_openvla_action()` TODO for where a real decoder would go.

## What changes when SmolVLA is swapped for OpenVLA (or anything else)

| Changes | Stays the same |
|---|---|
| `VLA_MODEL_FAMILY` env var (or config's `model_family`) | `RealVLAPolicyClient` (no code change) |
| `VLA_MODEL_ID_OR_PATH` / config's `model_id_or_path` | `generic_vla_server.py`'s endpoints/routing |
| Which `vla_adapters/<family>_adapter.py` is registered in `model_registry.py` | The `/predict` request/response schema |
| `vla_server/model_loader.py`'s per-family load/inference dispatch | `policy/vla_action_postprocessor.py` (client-side validation) |
| | `SafetyGate`/`SafetySupervisor`/`RobotBackend` (local, always outside the VLA policy) |

**The local robot control loop does not change.** That's the entire
point of this abstraction layer.

## Relationship to existing OpenVLA-specific files

`openvla_server_real/colab_vla_server.py` and
`notebooks/colab_vla_server_spike_v0.ipynb` are **not modified or
removed** -- they remain the place to experiment with OpenVLA
specifically (Google Drive cache, Colab GPU runtime, lazy `/load_model`
loading of a 7B checkpoint). They're now considered a deprecated/
optional path relative to this generic backend: `vla_server/
generic_vla_server.py` is the recommended default for new work,
because it's the one that doesn't hardcode a model family. Existing
`--policy-backend real-vla` + `configs/real_vla_backend_config.json`/
`configs/real_vla_backend_colab_config.json` workflows against
`openvla_server_dummy/real_vla_compatible_server.py` and
`colab_vla_server.py` are also untouched and keep working.

## Configs

- `configs/vla_backend_smolvla_config.json` -- `model_family: "smolvla"`,
  `model_id_or_path: "lerobot/smolvla_base"` (replace with a real
  checkpoint path/id once one is chosen), local server ports 9200.
- `configs/vla_backend_openvla_config.json` -- `model_family: "openvla"`,
  same `configs/real_vla_backend_config.json`-compatible shape, port
  9300, with a `note` field explaining it always returns
  `openvla_action_adapter_required` in this v0.

Both are `RealVLAPolicyClient`-compatible as-is: `--real-vla-config
configs/vla_backend_smolvla_config.json` (or the openvla one) works
with `run_full_recycling_cell_demo.py --policy-backend real-vla`
exactly like `configs/real_vla_backend_config.json` already does.

## Running it

```bash
# mock-action -- always works, no dependencies needed
VLA_MODEL_FAMILY=mock-action uvicorn vla_server.generic_vla_server:app --host 127.0.0.1 --port 9200

# smolvla -- attempts a real load; gracefully load_failed if lerobot/transformers
# or a GPU aren't available
VLA_MODEL_FAMILY=smolvla VLA_BACKEND_CONFIG_PATH=configs/vla_backend_smolvla_config.json \
  uvicorn vla_server.generic_vla_server:app --host 127.0.0.1 --port 9200

# openvla -- always ends in openvla_action_adapter_required regardless of load outcome
VLA_MODEL_FAMILY=openvla VLA_BACKEND_CONFIG_PATH=configs/vla_backend_openvla_config.json \
  uvicorn vla_server.generic_vla_server:app --host 127.0.0.1 --port 9300
```

```bash
python -m benchmark.probe_generic_vla_server --vla-config configs/vla_backend_smolvla_config.json --with-image

python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla --policy-backend real-vla \
  --real-vla-config configs/vla_backend_smolvla_config.json --real-vla-fallback-backend local-dummy \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-path data/test_images/recyclable_scene.jpg \
  --headless
```
