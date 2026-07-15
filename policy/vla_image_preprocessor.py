"""VLA image preprocessing (v0).

Turns a PolicyInput.image array into the wire-format payload a Real VLA
server expects (configs/real_vla_backend_config.json's "image_encoding"
section: optional resize, JPEG quality, base64 encoding), plus a debug
dict recording exactly what happened -- kept as a standalone module
(not a method on RealVLAPolicyClient) so image preprocessing can be
unit-tested and swapped independently of the HTTP client/fallback logic
around it.
"""

import base64
import io
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def encode_policy_image_for_vla(image: Optional[np.ndarray], config: dict) -> Tuple[Optional[dict], dict]:
    """Returns (image_payload, debug). image_payload is None (and debug
    reflects that) when `image` is None -- callers should still send the
    request without an image rather than skip the step, matching how
    FastAPIVLAPolicyClient already treats a missing image."""
    image_encoding_config = config.get("image_encoding", {}) or {}
    encoding = image_encoding_config.get("format", "jpg_base64")
    jpeg_quality = int(image_encoding_config.get("jpeg_quality", 90))
    resize_config = image_encoding_config.get("resize", {}) or {}
    resize_enabled = bool(resize_config.get("enabled", False))
    resize_width = int(resize_config.get("width", 224))
    resize_height = int(resize_config.get("height", 224))

    if image is None:
        return None, {
            "original_shape": None,
            "encoded_shape": None,
            "encoding": encoding,
            "jpeg_quality": jpeg_quality,
            "encoding_latency_ms": 0.0,
        }

    start = time.perf_counter()
    array = np.asarray(image)
    original_shape = list(array.shape)

    pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
    if resize_enabled:
        pil_image = pil_image.resize((resize_width, resize_height))

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    encoding_latency_ms = (time.perf_counter() - start) * 1000

    encoded_shape = [pil_image.height, pil_image.width, 3]
    image_payload = {"encoding": encoding, "shape": encoded_shape, "data": encoded_data}
    debug = {
        "original_shape": original_shape,
        "encoded_shape": encoded_shape,
        "encoding": encoding,
        "jpeg_quality": jpeg_quality,
        "encoding_latency_ms": round(encoding_latency_ms, 3),
    }
    return image_payload, debug


def encode_policy_images_by_role_for_vla(images_by_role: Optional[dict], config: dict) -> Tuple[dict, dict]:
    """Multi-camera counterpart of encode_policy_image_for_vla() -- same
    per-image encoding (resize/JPEG/base64), applied independently to
    each {role: np.ndarray} entry. Returns ({role: image_payload}, {role:
    debug}); ({}, {}) if images_by_role is None/empty, so a caller can
    always iterate the result without a None-check."""
    if not images_by_role:
        return {}, {}

    payloads = {}
    debugs = {}
    for role, image in images_by_role.items():
        payload, debug = encode_policy_image_for_vla(image, config)
        payloads[role] = payload
        debugs[role] = debug
    return payloads, debugs
