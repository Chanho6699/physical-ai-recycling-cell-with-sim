# Cloud SmolVLA Loading Spike (v0)

## Why SmolVLA before OpenVLA-7B

The [Colab OpenVLA spike](colab_vla_server_spike.md) proved the
plumbing (tunnel, `/health`, lazy `/load_model`, graceful failure)
works end to end, but actually loading OpenVLA-7B kept hitting real
environment limits: multi-GB shard downloads, a Colab GPU tier that
isn't guaranteed to have enough VRAM for a 7B model, and sessions that
can disconnect mid-download. None of that is a bug -- it's the actual
cost of a 7B model on free/shared compute. SmolVLA (LeRobot) targets
much more modest hardware, so it's the better next real (non-mock)
model to get an actual `action_len == 7` out of. See
[docs/generic_vla_backend.md](generic_vla_backend.md) for the fuller
version of this reasoning (this doc covers the cloud-loading spike
specifically; that one covers the adapter architecture).

## Why it's still designed to swap back to OpenVLA (or anything else)

This spike runs entirely through the [Generic VLA
Backend](generic_vla_backend.md) (`vla_server/generic_vla_server.py`),
not a SmolVLA-specific server. The only thing that names SmolVLA
anywhere in this spike is `VLA_MODEL_FAMILY=smolvla` (an env var) and
`configs/vla_backend_smolvla_config.json` (a config file). Swapping to
OpenVLA later means changing those two things, plus filling in
`OpenVLAActionAdapter._decode_openvla_action()`
(`vla_adapters/openvla_adapter.py`) once its raw action space is
verified -- the server, `RealVLAPolicyClient`, and the local robot
control loop never change either way.

## Colab Generic VLA Server structure

```text
Local PC (unchanged): camera, PyBullet, SafetySupervisor, RobotBackend,
                       RealVLAPolicyClient, episode recording
Colab: ONLY vla_server.generic_vla_server (model_family="smolvla")
       -- /health, /load_model, /predict. Never touches a robot.
```

`notebooks/colab_generic_vla_smolvla_spike_v0.ipynb` runs this server
inside a Colab GPU runtime and exposes it over a cloudflared tunnel,
mirroring the OpenVLA spike's shape but pointed at the
family-agnostic server instead of `openvla_server_real/colab_vla_server.py`.
That OpenVLA-specific notebook (and server file) is **not removed** --
it stays available for OpenVLA-specific experiments (its own Google
Drive cache flow, `openvla-dryrun` mode), just no longer the
recommended default.

## `/health` → `/load_model` → `/predict`, in order

1. **`GET /health`** -- always instant, never loads anything. Confirms
   the tunnel/HTTP path alone works before anything model-specific is
   attempted.

   ```json
   {
     "status": "ok",
     "model_family": "smolvla",
     "model_status": "not_loaded",
     "model_status_reason": "model load has not been requested",
     "model_id_or_path": "lerobot/smolvla_base",
     "local_files_only": false,
     "adapter": "SmolVLAActionAdapter",
     "version": "v0"
   }
   ```

2. **`POST /load_model`** -- the only thing that ever attempts a real
   SmolVLA load. `vla_server/model_loader.py` tries several LeRobot
   import paths in turn (`_SMOLVLA_IMPORT_CANDIDATES`, since LeRobot's
   package layout has moved across releases). Every failure mode is
   recorded into `model_status`/`model_status_reason` instead of
   crashing the server:

   | `model_status_reason` mentions | Interpretation |
   |---|---|
   | `SmolVLA policy import failed; VLM fallback disabled` | None of the tried LeRobot import paths matched this LeRobot version, and `VLA_ALLOW_VLM_FALLBACK` is unset/`0` (the default) -- **no download was attempted at all**. Use the notebook's section 4b `pkgutil` probe to find the real installed module path and add it to `_SMOLVLA_IMPORT_CANDIDATES` |
   | `missing_dependency` | `torch`/`lerobot` isn't installed (or, only when `VLA_ALLOW_VLM_FALLBACK=1`, `transformers` isn't installed either) |
   | a repo/model-not-found-style message | `VLA_MODEL_ID_OR_PATH` is wrong, private, or gated |
   | `CUDA out of memory` | Download and load actually worked -- this GPU tier just doesn't have enough VRAM |
   | anything else | Recorded verbatim; the reason string always includes `model_id_or_path`, `local_files_only`, `device`, `dtype` so the exact attempted configuration is visible |

   **Download success and model-loading success are different
   things** (same distinction as the OpenVLA spike): a `CUDA out of
   memory` failure means the download/import worked fine, it's a
   compute limit, not a data problem.

   ### SmolVLA action policy vs. SmolVLM/VLM backbone -- not the same thing

   **SmolVLA** (what this spike is actually trying to confirm) is a
   LeRobot **action policy** -- it should load via a LeRobot policy
   class such as `SmolVLAPolicy.from_pretrained(...)`, one of the
   `_SMOLVLA_IMPORT_CANDIDATES` paths. **SmolVLM/SmolVLM2** (e.g.
   `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) is a generic **vision-
   language backbone model** that SmolVLA happens to be built on top
   of -- it can answer questions about an image, but it has no action
   head and is not the thing this spike needs.

   `vla_server/model_loader.py`'s `_load_smolvla()` used to fall back to
   a plain `transformers.AutoModelForImageTextToText.from_pretrained(...)`
   load whenever none of the LeRobot import candidates matched. In
   practice this resolves to downloading the SmolVLM2 backbone (a
   multi-GB download that can hang a Colab session) instead of the
   actual SmolVLA action policy -- not useful for confirming SmolVLA's
   real import path, and not what "loaded" should mean for this spike.

   **This fallback is now disabled by default.** It only runs if
   `VLA_ALLOW_VLM_FALLBACK=1` is set explicitly (the notebook's section
   5 sets `VLA_ALLOW_VLM_FALLBACK=0` by default). With it left at `0`
   (or unset), a failed LeRobot import goes straight to
   `model_status=load_failed` with reason `SmolVLA policy import
   failed; VLM fallback disabled` -- no SmolVLM2 download is ever
   attempted. `/health` also reports the current setting as
   `allow_vlm_fallback`. Only set `VLA_ALLOW_VLM_FALLBACK=1` if you
   deliberately want to exercise the SmolVLM2 backbone fallback path
   itself (e.g. to confirm the download/load machinery works), not as
   part of confirming the real SmolVLA policy path -- use the
   notebook's section 4b `pkgutil` probe for that instead.

3. **`POST /predict`** with a dummy image -- if `model_status !=
   "loaded"`, returns the same structured `model_not_loaded` error
   `/health` already implied. If it is loaded, `vla_server/model_loader.py`
   tries `select_action`/`predict_action`/`generate` (whichever the
   loaded model actually exposes) to get a raw output, then
   `SmolVLAActionAdapter.normalize_model_output()` tries to interpret
   that raw output as a normalized 7-DoF action -- see below.

## SmolVLA raw output -> normalized action

`vla_adapters/smolvla_adapter.py`'s `SmolVLAActionAdapter` handles, in
order: a plain 7-number vector; a `{"action": ...}` dict; a chunked
`{"actions": [...]}` dict (action-horizon policies, selected by
`step_index`); a bare chunked `[T, 7]` list; a batched `[B, T, 7]` (or
`[B, 7]`) list; any of the above as a numpy array or torch tensor
instead of a plain list. It does this via a small bounded recursive
"peel one dimension, select by `step_index`" pass
(`_peel_to_vector()`) rather than assuming one fixed shape.

If none of that resolves to a flat 7-number vector, the adapter
returns a structured, fallback-triggering rejection instead of
guessing:

```json
{
  "error": "smolvla_raw_output_unrecognized_shape: <class 'dict'>",
  "model_family": "smolvla",
  "raw_model_output_available": true,
  "raw_output_summary": {"type": "dict", "dict_keys": ["logits", "hidden_states"]},
  "project_action_available": false,
  "reason": "smolvla_raw_output_unrecognized_shape: <class 'dict'>"
}
```

`raw_output_summary` (type + shape/keys, never the full tensor
contents) is exactly what tells you what to add to `_peel_to_vector()`
for a model whose output doesn't match any of the shapes already
handled -- extend that method, not the server or client.

## Success / failure interpretation, end to end

| Stage | Success looks like | Failure looks like (still not a crash) |
|---|---|---|
| Import `vla_server.generic_vla_server` | Finishes in <1s | (can't fail without a real bug -- no model touched here) |
| `/health` | `status: "ok"` immediately | Connection refused (tunnel/server not up) |
| `/load_model` | `model_status: "loaded"` | `model_status: "load_failed"` + reason (see table above) |
| `/predict` | HTTP 200, `action` has length 7 | HTTP 503 with structured `detail` (model not loaded, or adapter couldn't interpret raw output) |
| Local `probe_generic_vla_server.py` | `PASS` | `PASS_WITH_FALLBACK` (still counts as working -- fallback did its job) |
| Local full demo | `final_status: success`, `PASS` | Same, via `--real-vla-fallback-backend local-dummy` |

**A `PASS_WITH_FALLBACK` or `load_failed` result is a legitimate
outcome for this spike**, not a failure of the work -- the goal stated
up front is confirming `/load_model` and `/predict` raw output shapes,
not proving the full local demo succeeds (it already does, via
fallback, regardless).

## Next steps toward LeRobot dataset / fine-tuning

Once a SmolVLA checkpoint reliably loads and its raw output shape is
confirmed (or `SmolVLAActionAdapter` has been extended to handle it):

1. Record enough real (or simulated) episodes through this project's
   existing `TrajectoryRecorder`/LeRobot-style JSONL exporter (see
   [dataset_pipeline.md](dataset_pipeline.md)) to have something to
   fine-tune on.
2. Fine-tune SmolVLA on that dataset using LeRobot's own training
   scripts (out of scope for this repo directly -- this repo's role is
   the inference-time adapter/serving layer, not training
   infrastructure).
3. Point `VLA_MODEL_ID_OR_PATH` at the fine-tuned checkpoint -- no
   other code change needed, same as swapping any other checkpoint.

## Role separation: LLM Agent vs. VLA

Worth stating explicitly since both sit in the same "language in,
robot behavior out" pipeline but do very different jobs:

- **LLM Agent / TaskGoal parsing** (`llm_agent/rule_based_parser.py`,
  currently rule-based, not an LLM): turns a natural-language
  instruction into a structured `TaskGoal` (action, target object,
  target bin) **once per episode**, before perception/control ever
  starts. This is planning/task interpretation.
- **VLA (SmolVLA/OpenVLA/mock)**: turns `PolicyInput` (image + robot
  state + that same `TaskGoal` + step index) into a 7-DoF action
  **every control-loop step**. This is low-level continuous action
  prediction, not planning -- it never re-interprets the instruction.

Nothing in this spike changes that split. A future LLM-based multi-step
planner would still only ever produce `TaskGoal`s (or a sequence of
them); the VLA layer this spike is about stays the same one-step
action-prediction role regardless of which model backs it.

## See also

- [docs/generic_vla_backend.md](generic_vla_backend.md) -- the adapter/loader/registry architecture this spike runs on top of
- [docs/colab_vla_server_spike.md](colab_vla_server_spike.md) -- the OpenVLA-specific predecessor spike (still available, now optional)
- [docs/hardware_portability.md](hardware_portability.md) -- where this fits in the overall hardware-portability picture
