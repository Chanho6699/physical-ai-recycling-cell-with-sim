# Hardware Portability

This project's goal is not a PyBullet demo -- it is a **hardware-portable
Physical AI / VLA-ready software brain**: a perception -> Real2Sim ->
policy -> action -> safety -> logging pipeline built so that swapping the
simulator for a real robot arm, camera, and inference server changes as
little code as possible.

This document is the manifest of that portability: what's implemented
today, what a hardware port would swap out, and what calibration/config
work that swap actually requires. It complements
[docs/architecture.md](architecture.md) (how the pieces work) with a
narrower "what changes when this stops being a simulation" view.

## Current implementation (v0 interface boundaries)

| Boundary | Interface | Current implementation |
|---|---|---|
| Robot | `RobotBackend` (`robot_core/robot_backend.py`) | `PyBulletPandaBackend` (`robot_sim/pybullet_panda_backend.py`) |
| External camera | `CameraBackend` (`vision/camera_backend.py`) | `WebcamCameraBackend` (iVCam/webcam relay URL), `StaticImageCameraBackend` (`--image-path`) |
| Wrist camera | `CameraBackend` | `PyBulletWristCameraBackend` wrapping `PyBulletWristCamera` (virtual eye-in-hand camera) |
| Policy | `PolicyBackend` = `BasePolicy` (`policy/policy_backend.py`, `policy/base_policy.py`) | `DummyOpenVLAPolicy` (`--policy-backend local-dummy`), `FastAPIVLAPolicyClient` (`--policy-backend fastapi-dummy`, talks to `openvla_server_dummy/dummy_server.py`), `RealVLAPolicyClient` (`--policy-backend real-vla`, talks to `openvla_server_dummy/real_vla_compatible_server.py` today; the adapter layer a real OpenVLA/VLA server plugs into) |
| Safety (hard block) | `SafetyMonitor` + `SafetyGate` (`safety/safety_monitor.py`, `safety/safety_gate.py`) | `MockSafetyMonitor`, `ONNXRuntimeYOLOSafetyMonitor` (`--safety-monitor mock/onnx`) |
| Safety (pause/resume) | `SafetyMonitor` + `SafetySupervisor` (`safety/safety_supervisor.py`) | `MockHandIntrusionMonitor` (mock-timed, v0), `ExternalCameraHandSafetyMonitor` (real MediaPipe hand detection, v1) |
| Dataset | -- | Custom raw episode JSON (`TrajectoryRecorder`) + LeRobot-style JSONL exporter (`LeRobotDatasetExporter`) -- not official LeRobot parquet/video format |

Every row's "current implementation" column exists only to satisfy the
interface in the "Interface" column -- control-loop code in
`benchmark/run_full_recycling_cell_demo.py` is written against the
interfaces, and constructs a specific implementation only inside the
`create_robot_backend()` / `create_policy_backend()` /
`create_external_camera_backend()` / `create_hand_safety_monitor()`
factory functions. Swapping a row's implementation means changing that
one factory function, not the control loop.

## What a real hardware port would swap out

| Simulation piece | Hardware replacement |
|---|---|
| `PyBulletPandaBackend` | `RealRobotBackend` (vendor SDK/controller API) or `ROS2RobotBackend` (MoveIt2 + `follow_joint_trajectory`) -- both are unimplemented skeletons today (`robot_core/real_robot_backend.py`, `robot_core/ros2_robot_backend.py`), every method raises `NotImplementedError` with a docstring describing what to implement |
| PyBullet virtual wrist camera | A real wrist-mounted camera's `CameraBackend` (e.g. an Intel RealSense/similar over its own SDK or a `ROS2CameraBackend` subscribing to a wrist-mounted `/camera/.../image_raw`) |
| iVCam (phone camera over a Windows relay) | A calibrated, fixed external camera (proper intrinsics, fixed mount, ideally on the same network segment as the robot controller instead of a phone relay) |
`real-vla-compatible-server`'s `DummyOpenVLAPolicy` internals | A real OpenVLA (or other VLA) inference server behind the same `/predict`/`/health`/`/reset` contract and `configs/real_vla_backend_config.json` schema `RealVLAPolicyClient` already speaks -- only `openvla_server_dummy/real_vla_compatible_server.py`'s internals need to change, not the client or the control loop |
| `MockHandIntrusionMonitor` / `ExternalCameraHandSafetyMonitor`'s MediaPipe hands | A production-grade hand/person detector, ideally fused across external + wrist cameras, and always backed by a **hardware** e-stop -- `SafetySupervisor` deciding an action may be applied is a software safety layer, not a substitute for a physical interlock |
| Custom raw episode / LeRobot-style JSONL exporter | Official LeRobot parquet/video dataset format + HF Hub upload |

## Calibration a hardware port needs that simulation doesn't

None of this is implemented -- this is the checklist a real deployment
would have to work through, in roughly the order it becomes necessary:

- **Camera intrinsics** (external camera and, separately, wrist camera) -- PyBullet's virtual cameras use exact, noiseless intrinsics from `configs/wrist_camera_config.json`; a real camera needs an actual calibration (e.g. OpenCV `calibrateCamera`).
- **External camera <-> robot base extrinsics** -- today's ArUco homography (`real2sim/aruco_table_mapper.py`) implicitly re-derives a 2D table-plane mapping every frame; a real system typically wants an explicit, fixed 3D extrinsic (or keeps re-deriving it the same way, if the camera can't be rigidly fixed).
- **Wrist camera <-> end-effector transform** -- `configs/wrist_camera_config.json`'s `camera_local_position`/`camera_forward_local`/`camera_up_local` are today's stand-in for this; a real eye-in-hand camera needs a measured (not assumed) transform.
- **Bin positions in robot base frame** -- currently a fixed constant in `PyBulletPandaBackend`; a real cell needs this measured once per physical setup.
- **Workspace safety polygon** -- today's ArUco-marker-derived polygon (`build_hand_safety_workspace_polygon()` in `run_full_recycling_cell_demo.py`) assumes the external camera can see 4 taped markers; a fixed installation might instead hard-code the polygon in `configs/hand_safety_config.json`'s `roi` block once the camera is permanently mounted.

## What is explicitly still simulation-only / not done

- No real OpenVLA model connected. `FastAPIVLAPolicyClient` talks to a dummy server that runs `DummyOpenVLAPolicy` server-side; `RealVLAPolicyClient` (the adapter layer meant for a real VLA server) talks to `real_vla_compatible_server.py`, which does the same thing under a schema closer to what a real server would expect. Real model execution is deliberately optional in this v0 (no GPU inference requirement, no forced large-model download) -- see [docs/architecture.md](architecture.md#real-vla-backend-adapter-v0---policy-backend-real-vla).
- No ROS2 control (`ROS2RobotBackend`/`ROS2CameraBackend` are unimplemented skeletons; rclpy is imported lazily so the rest of the project works without ROS2 installed).
- No official LeRobot parquet/video format or HF Hub upload.
- No LLM-agent multi-step planning (`llm_agent/rule_based_parser.py` is a single rule-based instruction parser, not a planner).
- No custom-trained hand/object detector (YOLO weights and MediaPipe's hand landmark model are both off-the-shelf pretrained, not trained on this project's own data).

See [docs/architecture.md](architecture.md) for how each of these pieces
fits into the current control loop, and [docs/demo_commands.md](demo_commands.md)
for runnable commands against every implementation in the table above.
