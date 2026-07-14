"""Maps model_family -> adapter class (v0).

generic_vla_server.py never hardcodes which adapter backs a given
model_family -- it asks this registry. Adding a new model family later
means adding one entry here (plus a vla_adapters/<family>_adapter.py
and a loader dispatch in vla_server/model_loader.py); nothing else in
the server/client/control-loop changes.
"""

from typing import Dict, Optional, Type

from vla_adapters.base_vla_adapter import BaseVLAAdapter
from vla_adapters.mock_vla_adapter import MockVLAAdapter
from vla_adapters.openvla_adapter import OpenVLAActionAdapter
from vla_adapters.smolvla_adapter import SmolVLAActionAdapter

VALID_MODEL_FAMILIES = ("mock-action", "smolvla", "openvla")

_ADAPTER_CLASSES: Dict[str, Type[BaseVLAAdapter]] = {
    "mock-action": MockVLAAdapter,
    "smolvla": SmolVLAActionAdapter,
    "openvla": OpenVLAActionAdapter,
}


def get_adapter(model_family: str, config: Optional[dict] = None) -> BaseVLAAdapter:
    if model_family not in _ADAPTER_CLASSES:
        raise ValueError(f"Unknown model_family={model_family!r}. Expected one of {VALID_MODEL_FAMILIES}.")
    adapter_class = _ADAPTER_CLASSES[model_family]
    return adapter_class(config=config)
