# 02. Isaac Sim Minimal Scene Test

## Goal

Validate whether Isaac Sim can run on the local RTX 3050 8GB environment with a minimal scene.

## Hardware Constraint

Local GPU:

- NVIDIA GeForce RTX 3050
- VRAM: 8GB

Isaac Sim may be constrained on this hardware. Therefore, the first goal is not robot control, but minimal scene validation.

## Milestone 1: Minimal Scene

Success criteria:

- Isaac Sim launches successfully.
- An empty scene can be created.
- A single cube can be added.
- A single camera can be added.
- A screenshot or camera image can be saved.
- GPU memory usage is recorded.

## Out of Scope for This Step

- OpenVLA integration
- ROS 2 bridge
- Robot arm loading
- YOLO inference
- TensorRT optimization

## Next Step After Success

If the minimal scene runs successfully:

1. Add a table.
2. Add simple recyclable objects.
3. Add a lightweight robot arm.
4. Test scripted movement.
5. Connect ActionAdapter output to Isaac Sim command.

## Local Compatibility Check Result

Date: 2026-07-09

Command:

```bash
isaacsim isaacsim.exp.compatibility_check
Result:

System checking result: FAILED
Key findings:

GPU: NVIDIA GeForce RTX 3050 [supported]
Driver: 560.94 [supported]
VRAM: 8.59 GB [not enough]
Minimum VRAM: 10 GB

RAM: 16.67 GB [not enough]
Minimum RAM: 32 GB

Error:
No device could be created.
GPU Foundation is not initialized.
WSL/Vulkan device creation failed.
Decision:

Local WSL Isaac Sim execution will not be used as the primary development path.

The project will use:

Local PC for Python pipeline, ActionAdapter, Dummy/PyBullet backend, YOLO/TensorRT experiments
NVIDIA Brev or cloud GPU for Isaac Sim backend validation
