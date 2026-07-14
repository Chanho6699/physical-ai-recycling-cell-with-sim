# Colab VLA Server Spike (v0)

## Purpose

`--policy-backend real-vla` (see [docs/architecture.md](architecture.md#real-vla-backend-adapter-v0-policy-backend-real-vla))
is an adapter layer meant for a real VLA/OpenVLA server -- but this
project's local dev machine has no GPU worth running a 7B VLA model on.
This spike answers one narrow question: **can a temporary FastAPI VLA
server, hosted on a free Google Colab GPU runtime and exposed through a
public HTTPS tunnel, actually be called by the local
`RealVLAPolicyClient`?** Nothing about model quality or real OpenVLA
correctness is in scope here -- only the plumbing.

## Why Colab

Colab's free tier occasionally hands out a GPU (T4-class), which is
enough to *attempt* loading a 7B OpenVLA checkpoint -- something this
project's local machine cannot do at all. Colab is not being adopted as
infrastructure; it's a convenient, zero-install way to borrow a GPU for
an afternoon of adapter testing.

## This is a spike, not production

- **Session limits**: Colab disconnects on idle timeout or after its
  12/24h session cap, and closing the browser tab kills it immediately.
- **GPU is not guaranteed**: the free tier can hand out a CPU-only
  runtime, or no GPU may be available at all when you connect.
- **The tunnel URL changes every session.** ngrok's free tier and
  cloudflared's quick tunnels both mint a new random URL every time the
  notebook is (re)run -- there is no persistent address.
- **Latency is real**: every `/predict` call now round-trips over the
  public internet through a tunnel, on top of whatever the model itself
  takes.
- **None of this is meant to run unattended.** A real deployment would
  use a persistent server (a cloud GPU instance, not a notebook), not
  this spike.

None of these are bugs to fix -- they're why this is called a spike.
**A failed/unreachable Colab server is an environment limitation to
record, not a project failure**, precisely because `RealVLAPolicyClient`
already has a fallback path for exactly this (see
[docs/architecture.md](architecture.md#real-vla-backend-adapter-v0-policy-backend-real-vla)).

## Role split: local vs. Colab

```text
Local PC (unchanged, still fully in control):
  - external camera (iVCam/webcam) + wrist camera
  - Real2Sim mapping (ArUco/ROI)
  - PyBulletPandaBackend (RobotBackend)
  - SafetySupervisor (Safety Pause/Resume, hard-block SafetyGate)
  - action postprocessing/validation (policy/vla_action_postprocessor.py)
  - episode recording

Colab (temporary, replaceable, never in control of the robot):
  - FastAPI VLA server (openvla_server_real/colab_vla_server.py)
  - optional OpenVLA model loading
  - /health, /predict -- proposes actions only
```

**The Colab server never executes anything.** It returns a proposed
7-DoF action (or, in `openvla-dryrun` without a usable action, refuses
to). Whether that action is ever applied to the (simulated, for now)
robot is decided entirely on the local machine: `RealVLAPolicyClient`'s
own validation -> `policy/vla_action_postprocessor.py` (NaN/inf
rejection, clipping) -> `SafetyGate`/`SafetySupervisor` -> `RobotBackend`.
A compromised, buggy, or simply wrong Colab server can propose a bad
action; it cannot make the robot move on its own.

## Server modes (`openvla_server_real/colab_vla_server.py`)

Set via the `COLAB_VLA_SERVER_MODE` environment variable (default
`health-only`). Verify in this order -- each one is a strictly higher
bar than the last:

1. **`health-only`** -- no model loaded at all. `/health` reports
   `model_status=not_loaded`; `/predict` always fails with
   `model_not_loaded` (HTTP 503). Confirms the tunnel/HTTP path alone.
2. **`mock-action`** -- no real model; reuses `DummyOpenVLAPolicy` (the
   same phase engine `local-dummy`/`fastapi-dummy`/
   `real-vla-compatible-mock` already share) to return a deterministic,
   safe 7-DoF action. **This is the actual success bar for this spike.**
3. **`openvla-dryrun`** -- best-effort attempt to load a real OpenVLA
   model, but **only when you explicitly ask for it**. Any failure (no
   GPU, OOM, missing dependency, download stalling/failing) is
   recorded as `model_status=load_failed` with a `model_status_reason`,
   never a crash. If the model does load, `/predict` returns the raw
   model output for inspection but **still does not return an
   executable action** -- `project_action_available=false`,
   `reason=action_adapter_required`. OpenVLA's own action space/
   normalization has not been verified against this project's
   `delta_ee_7dof` schema, so it is never auto-converted and applied.

`openvla-direct` (a mode that would hand raw OpenVLA output straight to
the robot) is intentionally **not implemented**.

### Lazy model loading (`openvla-dryrun` only)

**Importing this module -- or starting the server in `openvla-dryrun`
mode -- never downloads or loads OpenVLA.** Importing/starting always
finishes in well under a second, and `/health` is usable immediately,
before any model load is ever attempted. This matters because an
OpenVLA checkpoint is several GB, and a shard download stalling (slow
Colab network, revoked GPU mid-download, ...) must never prevent the
FastAPI app itself from coming up -- otherwise you can't even reach
`/health` to tell what went wrong.

Model loading only happens when something explicitly calls
`POST /load_model`:

```text
GET  /health        always instant -- reports model_status without loading anything
POST /load_model    the ONLY thing that ever triggers a real OpenVLA download/load
POST /predict       (openvla-dryrun) uses whatever /load_model already produced;
                     never triggers a load itself
```

`/load_model` is idempotent and safe to call more than once: an
in-progress load reports back `model_status=loading` instead of
starting a second download; an already-finished load (success or
failure) reports back its final state instead of retrying. `model_status`
is one of `not_loaded` (never requested) / `loading` / `loaded` /
`load_failed` (attempted and failed -- see `model_status_reason`).

**Which model gets loaded** is controlled by environment variables,
read fresh every time (by `/health` and by `/load_model`), never
hardcoded:

```text
OPENVLA_MODEL_PATH        a local directory (e.g. a Google Drive mount
                          path like /content/drive/MyDrive/openvla-7b)
                          -- wins over OPENVLA_MODEL_ID if set
OPENVLA_MODEL_ID          a Hugging Face Hub repo id (default
                          "openvla/openvla-7b") -- used only if
                          OPENVLA_MODEL_PATH is not set
OPENVLA_LOCAL_FILES_ONLY  "1" to pass local_files_only=True to both
                          AutoProcessor.from_pretrained() and
                          AutoModelForVision2Seq.from_pretrained()
                          (skip any network call entirely); anything
                          else (including unset) means False
```

`/health` always reports the currently-resolved `model_id_or_path` and
`local_files_only` -- what `/load_model` would use right now -- so you
can confirm the env vars took effect before triggering an actual
(potentially multi-GB, multi-minute) load.

If a Colab GPU is unavailable, shard downloads stall, or VRAM runs out,
`/load_model` reports `model_status=load_failed` with a reason and the
server keeps running -- this is exactly the environment-limitation
case this spike is built to record cleanly instead of crash on, and
`/predict` continues to fail with a structured `model_not_loaded`/
`action_adapter_required` error so the local `RealVLAPolicyClient`
falls back.

## OpenVLA Drive cache workflow

OpenVLA-7B is ~15GB. Downloading it fresh at the start of every Colab
session is slow, wastes quota, and means a dropped session loses all
progress. `notebooks/colab_vla_server_spike_v0.ipynb`'s section 3b
splits storage into two tiers instead:

```text
Google Drive (/content/drive/MyDrive/openvla-cache/openvla-7b)
  -> permanent cache, survives across Colab sessions
  -> populated once via huggingface_hub.snapshot_download(..., resume_download=True)
       (resumable -- a dropped session just re-runs the same cell instead
       of restarting the download from 0%)

/content/openvla-7b (this session's local Colab VM disk)
  -> fast-loading copy for *this session only*, made by rsync-ing
     from the Drive cache
  -> wiped when the session ends; next session only needs to re-run
     the (fast, local) rsync copy, not snapshot_download again
```

`/load_model` only ever reads from a local path: it is wired up via
`OPENVLA_MODEL_PATH=/content/openvla-7b` and
`OPENVLA_LOCAL_FILES_ONLY=1` (see [Lazy model loading](#lazy-model-loading-openvla-dryrun-only)
above), so a working Drive cache means `/load_model` never re-downloads
anything from Hugging Face -- it either loads straight from
`/content/openvla-7b` or fails fast with a local-file error.

**Download success and model-loading success are two different
things**, and the notebook's section 6b explains how to tell them
apart:

- A **`CUDA out of memory`** failure means the cache is fine -- the
  download and copy worked -- but this Colab GPU tier doesn't have
  enough VRAM to hold the model. That's a compute limit, not a data
  problem; retrying `/load_model` won't help without a bigger GPU.
- A **missing-file/shard** failure under `local_files_only=True` means
  the Drive cache itself is incomplete. Re-run the `snapshot_download`
  cell (resumable, so this is cheap) and then the `rsync` copy cell
  again.

**Do not commit to Git**: model weights/checkpoints, the Drive cache
directory itself (it never leaves Colab/Drive, but just to be clear),
`results/`, `datasets/`, or any temporary `cloudflared`/ngrok tunnel
URL baked into a config file. `configs/real_vla_backend_colab_config.json`
is meant to be edited in place by `scripts/update_colab_vla_config.py`
each session -- if you commit a real tunnel URL into it, the next
person to clone the repo will just see a dead URL (Colab sessions
don't persist), which is harmless but pointless to track in Git.

## Using it

### 1. Run the notebook

Open `notebooks/colab_vla_server_spike_v0.ipynb` in Colab and run it top to bottom. Two paths:

- **Just checking connectivity (no GPU/OpenVLA needed)**: run section 1 (GPU check, informational only), section 2 (base deps), then skip straight to section 3 (repo clone) and section 4 -- set `SERVER_MODE = "mock-action"` there instead of the notebook's default. Skip sections 2b/2c/3b entirely. Continue from section 5 onward.
- **Real OpenVLA `openvla-dryrun` load (this notebook's default)**: run every section in order -- 1 (GPU check) -> 2/2b/2c (deps + import validation) -> 3 (repo clone) -> **3b (Drive cache: mount Drive, HF login, `snapshot_download` to Drive, copy to `/content/openvla-7b`, local-only processor smoke test)** -> 4 (sets `SERVER_MODE="openvla-dryrun"` + `OPENVLA_MODEL_PATH`/`OPENVLA_LOCAL_FILES_ONLY`, imports the app) -> 5 (start server + tunnel) -> 6 (curl commands, expected `/health` shown inline) -> **6b (`POST /load_model` -- the only cell that ever triggers a real model load; see "OpenVLA Drive cache workflow" above for how to read a failure here)**.

Either way, the server is fully up and `/health`-reachable as soon as section 5 finishes, regardless of `SERVER_MODE` -- section 6b is always optional and always separate from the server actually being alive.

### 2. Update the local config

```bash
python scripts/update_colab_vla_config.py \
  --base-url https://xxxx.ngrok-free.app \
  --config configs/real_vla_backend_colab_config.json
```

Rewrites only `server_url`/`health_url` in place (pretty-printed JSON,
every other key untouched). Rejects a `--base-url` missing an
`http(s)://` scheme or host with a clear error instead of silently
writing a broken config.

### 3. Probe

```bash
python -m benchmark.probe_colab_vla_server \
  --real-vla-config configs/real_vla_backend_colab_config.json \
  --with-image
```

Prints `health`, `server_mode`, `model_status`, `action_len`,
`fallback_used`, latency, and (in `openvla-dryrun`)
`raw_model_output_available`/`project_action_available`/`reason`.

### 4. Full demo

```bash
python -m benchmark.run_full_recycling_cell_demo \
  --policy dummy-openvla \
  --policy-backend real-vla \
  --real-vla-config configs/real_vla_backend_colab_config.json \
  --real-vla-fallback-backend local-dummy \
  --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
  --image-path data/test_images/recyclable_scene.jpg \
  --wrist-camera-mode refine \
  --wrist-refinement-policy blend \
  --policy-observation-source wrist \
  --record \
  --record-perception-metadata \
  --record-policy-observations \
  --headless
```

Expect `policy_backend: real-vla`, `model: colab-mock-action` (or
`openvla-dryrun`/fallback's model name), `final_status: success`,
`PASS` -- or, if the Colab session already ended, `PASS` still, via
`--real-vla-fallback-backend local-dummy` picking up the slack.

## Fallback must work for this to count as a success

Every one of the above steps is expected to *also* succeed with the
Colab server unreachable (session ended, wrong URL, etc.) -- that's the
entire point of `RealVLAPolicyClient`'s fallback path
(`--real-vla-fallback-backend local-dummy`, default). If fallback ever
stopped working, that would be the actual regression to worry about,
not a Colab session timing out.

## See also

- [docs/architecture.md](architecture.md#real-vla-backend-adapter-v0-policy-backend-real-vla) -- the `real-vla` adapter this spike targets
- [docs/hardware_portability.md](hardware_portability.md) -- where this fits in the overall hardware-portability picture
- [docs/demo_commands.md](demo_commands.md) -- the exact runnable commands above, in context with every other demo
