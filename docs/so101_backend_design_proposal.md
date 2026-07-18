# SO-101 Backend Design Proposal (exploration only -- not wired in)

This is a design proposal only. No production file is modified by this
document, and no `So101PyBulletBackend` class exists yet -- see the
chat report for what was actually built this task (inspection/smoke
scripts under `benchmark/`, vendored assets under `third_party/so101_arm/`).

## Relationship to the existing `RobotBackend` interface

`robot_core/robot_backend.py` already defines an ABC that
`PyBulletPandaBackend` implements: `reset()`, `get_state()`,
`apply_command()`, `move_end_effector_to()`, `open_gripper()`,
`close_gripper()`, `shutdown()`. That interface is Cartesian-EE-delta
centric (matches `action_adapter/adapter_v0.py`'s `RobotCommand`).

The interface sketched in this task's request is more granular --
it separates joint-space and Cartesian access explicitly
(`get_joint_positions()`, `command_joint_positions()`,
`get_end_effector_pose()`, `command_end_effector_delta()`) and adds
`is_object_grasped()` as its own query rather than folding it into a
generic state dict. Proposed shape, reconciling both:

```python
class RobotBackend(ABC):
    def reset(self) -> dict: ...                          # -> get_observation()'s own shape
    def get_observation(self) -> dict: ...                 # superset: joint positions, EE pose, gripper, camera frames, task/grasp status
    def get_joint_positions(self) -> list: ...             # arm joints only, backend-native order/count
    def get_end_effector_pose(self) -> tuple: ...           # (position[3], orientation) -- orientation representation is backend-declared (see table)
    def command_joint_positions(self, positions: list, **kwargs) -> dict: ...
    def command_end_effector_delta(self, delta_position, delta_orientation=None, **kwargs) -> dict: ...
    def set_gripper(self, command) -> dict: ...            # backend-native gripper command type -- see table, NOT assumed to be "open"/"close" strings
    def step(self, steps: int = 1) -> None: ...            # advance physics without a new command (distinct from apply-and-settle)
    def is_object_grasped(self) -> bool: ...
    def shutdown(self) -> None: ...
```

`PandaBackend` (today's `PyBulletPandaBackend`, unchanged) and a future
`So101PyBulletBackend` would both implement this same surface -- the
existing `apply_command()`/`move_end_effector_to()`/`open_gripper()`/
`close_gripper()` methods map onto `command_end_effector_delta()` /
`command_joint_positions()` / `set_gripper()` respectively, so adopting
this later is an additive/renaming change, not a rewrite of Panda's
internals.

## Panda vs. SO-101: confirmed differences

| Aspect | Panda (existing) | SO-101 (this exploration) |
|---|---|---|
| DOF (controllable) | 7 arm + 2 finger (prismatic, mirrored) = 9 | 5 arm + 1 gripper (revolute) = 6 |
| Action representation | EE-delta Cartesian (`[dx,dy,dz,drx,dry,drz,gripper]`, axis-angle) | Not yet decided -- joint-space is native; a Cartesian-delta layer would need its own IK wrapper (validated feasible this task, position-only) |
| Joint names | `panda_joint1..7`, `panda_finger_joint1/2` | `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` |
| EE link | `panda_grasptarget` (virtual point between fingers, link 11) | `gripper_frame_link` (dummy link, purpose-built by the URDF authors -- see its own "Gripper frame" comment) |
| Gripper command | Binary open/close string -> symmetric `[0.04,0.04]`/`[0,0]` prismatic targets | Single revolute joint, range `[-0.174533, 1.745329]` rad; **real LeRobot hardware uses a 0-100 linear scale (0=closed,100=open) that is NOT yet reflected in this URDF** (stated explicitly in the vendored README) -- any backend must own this mapping itself |
| Workspace | Hand-tuned box, `DEFAULT_WORKSPACE_BOUNDS_STR = "-0.1,0.9,-0.7,0.7,0.0,1.0"` | Not established -- this task's rough empirical anchor (current EE distance from base at neutral, ~0.45m, x1.3 margin) is a placeholder, not a validated spec |
| Home pose | `READY_JOINT_POSITIONS` (7 hand-picked joint angles) | `[0, 0, 0, 0, 0]` for all 5 arm joints -- this URDF's own "new calibration" already defines 0.0 as the middle of each joint's range, so no hand-tuning was needed (confirmed converges cleanly, see joint-control smoke test) |
| IK method | Full 6-DOF (position + orientation) `p.calculateInverseKinematics` | Position-only validated this task (5/5 targets, <3mm error); full 6-DOF not yet tested |
| Camera reference frame | Fixed main camera (world-space) + wrist camera rigidly attached to `end_effector_link_index` | Not defined at all -- no camera mount exists in the vendored URDF; a new design decision, not a reuse of Panda's convention |
| State dimension (for VLA obs) | 8D: `ee_pos[3] + ee_orientation_axis_angle[3] + gripper_qpos[2]` (2 mirrored finger channels) | Not defined -- SO-101's single-jaw gripper has no second mirrored channel, so an analogous vector would likely be 7D (`ee_pos[3] + ee_orientation[3] + gripper_qpos[1]`), but this is a proposal, not implemented |

## What would need deciding before writing `So101PyBulletBackend`

1. Gripper command mapping (URDF radians <-> LeRobot's 0-100 hardware scale) -- currently undefined upstream, must be this project's own decision.
2. A real workspace box (this task's radius estimate is a rough placeholder only).
3. Camera mount point/orientation (no precedent in the vendored assets).
4. Whether to expose Cartesian-delta control at all, or drive this arm joint-space-natively (SO-101's real hardware/LeRobot stack is typically joint-position-commanded, unlike Panda's EE-delta convention this project already uses) -- a real design choice, not a detail.
