"""Semantic adapter interfaces (v0): PolicyAdapter, ObservationAdapter,
ActionAdapter.

These are a different, finer-grained split than two other same/similar-
sounding things already in this repo -- read this before implementing
either:

  - vla_adapters/base_vla_adapter.py's BaseVLAAdapter is the existing
    HTTP-facing contract vla_server/generic_vla_server.py and
    vla_server/model_registry.py already depend on
    (build_model_input/normalize_model_output/health_info). It is NOT
    replaced by this module -- a BaseVLAAdapter subclass (e.g.
    SmolVLAActionAdapter) composes an ObservationAdapter/ActionAdapter
    pair internally and bridges the resulting CanonicalRobotCommand back
    to BaseVLAAdapter's existing {"action": [...], ...} dict shape, so
    the server/client/robot-backend contracts never change.

  - action_adapter/adapter_v0.py's ActionAdapter is a *different, older*
    class with the same name in a different module: it converts an
    already-flat 7-float list into a RobotCommand for direct PyBullet
    execution and knows nothing about manifests/compatibility. This
    module's ActionAdapter is the semantic layer *above* that one --
    never import both under the same bare name in one file.

PolicyAdapter is the per-checkpoint composite: it owns a
PolicyManifest, an ObservationAdapter, and an ActionAdapter, and is what
a real checkpoint integration (e.g. a future SmolVLALiberoPolicyAdapter)
implements as a unit.
"""

from abc import ABC, abstractmethod
from typing import Optional

from policy_semantics.canonical_command import CanonicalRobotCommand
from policy_semantics.canonical_observation import CanonicalObservation
from policy_semantics.manifest import PolicyManifest
from policy_semantics.native_policy_action import NativePolicyAction


class ObservationAdapter(ABC):
    """CanonicalObservation -> whatever a specific checkpoint's official
    preprocessor needs, manifest-aware: it knows which camera roles/
    state fields the checkpoint's PolicyManifest declares as required,
    so it can report which of them CanonicalObservation actually has
    (see CanonicalObservation.missing_camera_roles()) instead of
    silently zero-filling and calling it done."""

    @abstractmethod
    def build_preprocessor_input(self, observation: CanonicalObservation, manifest: PolicyManifest) -> dict:
        raise NotImplementedError


class ActionAdapter(ABC):
    """NativePolicyAction (the checkpoint's own official postprocessor
    has already run -- see native_policy_action.py) -> CanonicalRobotCommand,
    or None. The semantic counterpart of BaseVLAAdapter.normalize_model_output(),
    except its input is already-unnormalized (never a raw model tensor)
    and its output is the typed CanonicalRobotCommand (see
    canonical_command.py) rather than a bare dict/list -- callers bridge
    that to the legacy wire format themselves (see SmolVLAActionAdapter).
    Must return None (never fabricate a command) whenever the manifest/
    compatibility result says this checkpoint's action semantics aren't
    verified for the target embodiment, exactly like BaseVLAAdapter's
    action=None contract."""

    @abstractmethod
    def decode(
        self, native_action: NativePolicyAction, manifest: PolicyManifest, context: dict
    ) -> Optional[CanonicalRobotCommand]:
        raise NotImplementedError


class PolicyAdapter(ABC):
    """The per-checkpoint composite: owns the PolicyManifest this
    checkpoint claims, plus the ObservationAdapter/ActionAdapter pair
    that actually implements it. A BaseVLAAdapter subclass (the thing
    model_registry.py actually instantiates) holds one of these and
    delegates to it -- this class itself is never registered with
    model_registry.py directly."""

    manifest: PolicyManifest
    observation_adapter: ObservationAdapter
    action_adapter: ActionAdapter

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError
