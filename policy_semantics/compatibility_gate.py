"""CompatibilityGate -- checks a PolicyManifest against this project's
target embodiment (policy_semantics.manifest.PANDA_TARGET_EMBODIMENT)
before a checkpoint may drive production output.

Policy (all required by this task, do not relax any of these):
  - Any semantically-relevant field being UNKNOWN fails that check --
    UNKNOWN is never treated as "probably fine".
  - Matching action_dimension alone is never sufficient for `passed`.
  - `passed=False` still allows a caller to run a shape-only (dimension
    only, no claimed semantic correctness) path, but only when the
    caller explicitly passes smoke_test_mode=True -- see
    shape_only_allowed / semantic_action_valid below. `passed` and
    `semantic_action_valid` are always the same value; smoke_test_mode
    never makes either one True.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from policy_semantics.manifest import (
    PANDA_BACKEND_CAPABILITIES,
    PANDA_TARGET_EMBODIMENT,
    UNKNOWN,
    ActionSpace,
    PolicyManifest,
)


@dataclass
class CompatibilityResult:
    model_id: str
    target_model_id: str
    smoke_test_mode: bool
    passed: bool
    semantic_action_valid: bool
    shape_only_allowed: bool
    checks: Dict[str, dict]
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "target_model_id": self.target_model_id,
            "smoke_test_mode": self.smoke_test_mode,
            "passed": self.passed,
            "semantic_action_valid": self.semantic_action_valid,
            "shape_only_allowed": self.shape_only_allowed,
            "checks": self.checks,
            "reasons": self.reasons,
        }


def _embodiment_family(text: str) -> str:
    """Coarse embodiment grouping so 'Franka Panda (PyBullet ...)' and
    'LIBERO Franka Panda (robosuite/MuJoCo ...)' are recognized as the
    same robot family despite different exact strings, while 'SO-100/
    SO-101 ...' is correctly recognized as a different one. This is
    deliberately coarse (word-matching, not a robot ontology) -- good
    enough to separate "same arm" from "different arm", which is all
    this check needs."""
    lowered = (text or "").lower()
    if "panda" in lowered:
        return "franka_panda"
    if "so-100" in lowered or "so100" in lowered or "so-101" in lowered or "so101" in lowered:
        return "so-100/so-101"
    return UNKNOWN


class CompatibilityGate:
    @staticmethod
    def check(
        manifest: PolicyManifest,
        target: PolicyManifest = PANDA_TARGET_EMBODIMENT,
        smoke_test_mode: bool = False,
    ) -> CompatibilityResult:
        checks: Dict[str, dict] = {}
        reasons: List[str] = []

        def record(name: str, passed: bool, detail: str) -> None:
            checks[name] = {"passed": passed, "detail": detail}
            if not passed:
                reasons.append(f"{name}: {detail}")

        record(
            "action_dimension",
            manifest.action_dimension == target.action_dimension and manifest.action_dimension > 0,
            f"manifest={manifest.action_dimension}, target={target.action_dimension}",
        )

        record(
            "action_semantics_known",
            manifest.action_space != ActionSpace.UNKNOWN,
            f"action_space={manifest.action_space.value}",
        )

        manifest_family = _embodiment_family(manifest.source_embodiment)
        target_family = _embodiment_family(target.source_embodiment)
        record(
            "source_embodiment",
            manifest_family != UNKNOWN and manifest_family == target_family,
            f"manifest={manifest.source_embodiment!r} (family={manifest_family}), "
            f"target={target.source_embodiment!r} (family={target_family})",
        )

        record(
            "action_space_matches_target",
            manifest.action_space == target.action_space,
            f"manifest={manifest.action_space.value}, target={target.action_space.value}",
        )

        record(
            "relative_or_absolute",
            manifest.relative_or_absolute == target.relative_or_absolute,
            f"manifest={manifest.relative_or_absolute!r}, target={target.relative_or_absolute!r}",
        )

        record(
            "rotation_representation",
            manifest.rotation_representation == target.rotation_representation,
            f"manifest={manifest.rotation_representation!r}, target={target.rotation_representation!r}",
        )

        record(
            "reference_frame",
            manifest.reference_frame == target.reference_frame,
            f"manifest={manifest.reference_frame!r}, target={target.reference_frame!r}",
        )

        record(
            "gripper_convention",
            # A *known* (not UNKNOWN) gripper convention is sufficient here,
            # not exact equality with the target's polarity string -- a
            # known-but-opposite convention (e.g. robosuite's -1=open/
            # 1=closed vs. this project's 1.0=close/0.0=open) is fine as
            # long as a real, tested ActionAdapter performs the conversion
            # (see policy_semantics/adapters/smolvla_libero_adapter.py's
            # SmolVLALiberoActionAdapter.decode()). Requiring identical
            # polarity would reject every checkpoint that happens to use a
            # different (but well-understood) sign convention for no good
            # reason -- UNKNOWN is what's actually unsafe here, not "different".
            manifest.gripper_included and manifest.gripper_convention != UNKNOWN,
            f"manifest gripper_included={manifest.gripper_included}, "
            f"convention={manifest.gripper_convention!r} (target's own convention: "
            f"{target.gripper_convention!r} -- need not match verbatim, see comment)",
        )

        record(
            "gripper_index",
            manifest.gripper_index is not None and manifest.gripper_index == target.gripper_index,
            f"manifest={manifest.gripper_index!r}, target={target.gripper_index!r}",
        )

        record(
            "required_observations_declared",
            bool(manifest.required_camera_roles) and bool(manifest.state_fields),
            f"camera_roles={manifest.required_camera_roles!r}, state_fields={manifest.state_fields!r}",
        )

        record(
            "axis_convention_verified",
            manifest.axis_convention_verified is True,
            f"manifest.axis_convention_verified={manifest.axis_convention_verified} -- "
            + (
                "confirmed via real cross-simulator simulation, see "
                "docs/panda_axis_cross_verification.md/.json"
                if manifest.axis_convention_verified
                else "this checkpoint's source-simulator base-frame axis convention (+X/+Y/+Z direction) has "
                "not been independently cross-checked against this project's PyBullet Panda base frame; "
                "both claiming 'robot_base' is not the same as a verified match"
            ),
        )

        record(
            "backend_capabilities",
            # Full production compatibility requires the *executing*
            # backend to actually be able to carry out translation,
            # rotation, AND gripper commands -- a checkpoint whose
            # manifest is otherwise perfect is still not usable if the
            # backend would silently (or even loudly) drop one of them.
            # Before robot_sim/pybullet_panda_backend.py's v2 rotation
            # implementation, supports_cartesian_rotation was False here
            # and this check would have failed every manifest -- now it
            # reflects what the backend actually does, not what we hope
            # it does.
            PANDA_BACKEND_CAPABILITIES.supports_cartesian_translation
            and PANDA_BACKEND_CAPABILITIES.supports_cartesian_rotation
            and PANDA_BACKEND_CAPABILITIES.supports_gripper
            and PANDA_BACKEND_CAPABILITIES.rotation_representation == target.rotation_representation
            and PANDA_BACKEND_CAPABILITIES.reference_frame == target.reference_frame,
            f"backend_capabilities={PANDA_BACKEND_CAPABILITIES} (see robot_sim/pybullet_panda_backend.py's "
            "get_capabilities() for the live source of truth this mirrors)",
        )

        record(
            "processor_and_normalization",
            manifest.official_processor_available
            and manifest.official_processor_wired
            and manifest.normalization != UNKNOWN,
            f"official_processor_available={manifest.official_processor_available}, "
            f"official_processor_wired={manifest.official_processor_wired}, "
            f"normalization={manifest.normalization!r} -- this project's loader must call the "
            "checkpoint's own official processor/postprocessor before this can pass, per policy "
            "against inventing a meter/radian scale factor ourselves",
        )

        passed = all(check["passed"] for check in checks.values())

        return CompatibilityResult(
            model_id=manifest.model_id,
            target_model_id=target.model_id,
            smoke_test_mode=smoke_test_mode,
            passed=passed,
            semantic_action_valid=passed,
            shape_only_allowed=smoke_test_mode,
            checks=checks,
            reasons=reasons,
        )
