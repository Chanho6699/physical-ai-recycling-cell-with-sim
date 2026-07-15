"""Semantic policy-adapter layer (v0).

Everything under this package is about *meaning*, not *shape*: given a
VLA checkpoint's raw output, decide whether its action space (joint vs
Cartesian, delta vs absolute, gripper convention, reference frame, ...)
actually matches this project's robot before ever letting it drive
anything -- matching array length alone (the mistake this package
replaces, see policy_semantics/adapters/legacy_shape_only_adapter.py)
is not the same question.

See canonical_command.py (the one normalized command every checkpoint
must be translated into or refused), manifest.py (what each checkpoint
claims about its own action/observation space), compatibility_gate.py
(checks a manifest against this project's target embodiment before
production use), and interfaces.py (the adapter shapes a real
checkpoint integration implements).

Nothing in vla_server/generic_vla_server.py's HTTP contract,
RealVLAPolicyClient, action_adapter/adapter_v0.py's RobotCommand, or
robot_sim/pybullet_panda_backend.py changes because of this package --
policy_semantics produces a CanonicalRobotCommand internally and
vla_adapters/smolvla_adapter.py bridges it back to the existing flat
[dx, dy, dz, droll, dpitch, dyaw, gripper] wire format those already
depend on.
"""
