"""SO-101 bin visual-salience measurement (see this task's chat report,
"bin visual salience 개선안"). Pure measurement/diagnostic -- does NOT
change scene geometry, collision, expert behavior, or the dataset
schema. Reuses the EXACT front-camera parameters
(robot_sim.so101_pybullet_backend's own FRONT_CAMERA_* constants) the
real dataset recorder uses -- never retypes them -- so these numbers
describe the SAME image a recorded episode would actually produce.

Renders the SAME scene twice per seed:
  1. Plain RGB via backend.render_front_camera() -- byte-identical to
     what benchmark/collect_so101_bin_dataset.py records.
  2. A PyBullet per-pixel object-ID segmentation mask
     (ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX), using the SAME view/
     projection matrices -- this is the "object segmentation mask"
     this task's own section 3 asks to consider when exact
     segmentation is otherwise hard to get.

From the segmentation mask this computes, per seed: bin pixel count/
percentage, which of the 4 walls (+ bottom) are actually visible, bin-
vs-background contrast, bin-vs-object color distance, and a blank/
broken-frame check -- all read from the ACTUAL rendered pixels, never
assumed.

Run:
  .venv-vla/bin/python -m benchmark.measure_so101_bin_visual_salience --label before
  .venv-vla/bin/python -m benchmark.measure_so101_bin_visual_salience --label after
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from benchmark.evaluate_so101_expert_small_randomization import DEFAULT_X_RANGE, DEFAULT_Y_RANGE, sample_object_position
from robot_sim.so101_pybullet_backend import (
    FRONT_CAMERA_EYE,
    FRONT_CAMERA_FAR,
    FRONT_CAMERA_FOV,
    FRONT_CAMERA_HEIGHT,
    FRONT_CAMERA_NEAR,
    FRONT_CAMERA_TARGET,
    FRONT_CAMERA_UP,
    FRONT_CAMERA_WIDTH,
    So101PyBulletBackend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "so101_bin_visual_before_after"
PREVIEW_SEEDS = [0, 5, 10, 15, 19]
SEG_OBJECT_ID_MASK = (1 << 24) - 1  # low 24 bits -- object unique id, independent of link-index encoding (see PyBullet's own getCameraImage() segmentation docs)


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def render_rgb_and_segmentation(backend: So101PyBulletBackend) -> tuple:
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=FRONT_CAMERA_EYE, cameraTargetPosition=FRONT_CAMERA_TARGET, cameraUpVector=FRONT_CAMERA_UP,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=FRONT_CAMERA_FOV, aspect=FRONT_CAMERA_WIDTH / FRONT_CAMERA_HEIGHT, nearVal=FRONT_CAMERA_NEAR, farVal=FRONT_CAMERA_FAR,
    )
    _, _, rgba_pixels, _, seg_pixels = p.getCameraImage(
        width=FRONT_CAMERA_WIDTH, height=FRONT_CAMERA_HEIGHT, viewMatrix=view_matrix, projectionMatrix=projection_matrix,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, physicsClientId=backend.client_id,
    )
    rgb = np.array(rgba_pixels, dtype=np.uint8).reshape((FRONT_CAMERA_HEIGHT, FRONT_CAMERA_WIDTH, 4))[:, :, :3]
    seg = np.array(seg_pixels, dtype=np.int64).reshape((FRONT_CAMERA_HEIGHT, FRONT_CAMERA_WIDTH))
    seg_object_ids = np.where(seg >= 0, seg & SEG_OBJECT_ID_MASK, -1)
    return rgb, seg_object_ids


def ring_mask(mask: np.ndarray, dilate_px: int = 4) -> np.ndarray:
    """Pixels within `dilate_px` of `mask` but not in it -- a cheap
    "immediate surroundings" ring, no scipy dependency."""
    padded = np.pad(mask, dilate_px, mode="constant", constant_values=False)
    dilated = np.zeros_like(padded)
    for dy in range(-dilate_px, dilate_px + 1):
        for dx in range(-dilate_px, dilate_px + 1):
            dilated |= np.roll(np.roll(padded, dy, axis=0), dx, axis=1)
    dilated = dilated[dilate_px:-dilate_px, dilate_px:-dilate_px]
    return dilated & (~mask)


def compute_visibility_metrics(rgb: np.ndarray, seg: np.ndarray, backend: So101PyBulletBackend) -> dict:
    total_pixels = seg.size
    bin_ids = {name: bid for name, bid in backend.bin_body_ids.items() if name != "all"}

    per_wall_pixel_count = {name: int(np.sum(seg == bid)) for name, bid in bin_ids.items()}
    bin_mask = np.isin(seg, list(bin_ids.values()))
    object_mask = seg == backend.object_id
    table_mask = seg == backend.table_id
    background_mask = ~bin_mask & ~object_mask & ~table_mask & (seg != backend.robot_id)

    bin_pixel_count = int(np.sum(bin_mask))
    object_pixel_count = int(np.sum(object_mask))

    gray = rgb.astype(np.float64).mean(axis=2)
    bin_gray_mean = float(gray[bin_mask].mean()) if bin_pixel_count > 0 else None
    surround = ring_mask(bin_mask, dilate_px=4) & background_mask
    surround_gray_mean = float(gray[surround].mean()) if surround.sum() > 0 else None
    bin_background_grayscale_contrast = (
        abs(bin_gray_mean - surround_gray_mean) if bin_gray_mean is not None and surround_gray_mean is not None else None
    )

    bin_mean_rgb = rgb[bin_mask].astype(np.float64).mean(axis=0).tolist() if bin_pixel_count > 0 else None
    object_mean_rgb = rgb[object_mask].astype(np.float64).mean(axis=0).tolist() if object_pixel_count > 0 else None
    bin_object_rgb_distance = (
        float(np.linalg.norm(np.array(bin_mean_rgb) - np.array(object_mean_rgb)))
        if bin_mean_rgb is not None and object_mean_rgb is not None else None
    )
    surround_mean_rgb = rgb[surround].astype(np.float64).mean(axis=0).tolist() if surround.sum() > 0 else None
    bin_background_rgb_distance = (
        float(np.linalg.norm(np.array(bin_mean_rgb) - np.array(surround_mean_rgb)))
        if bin_mean_rgb is not None and surround_mean_rgb is not None else None
    )

    return {
        "total_pixels": int(total_pixels),
        "bin_pixel_count": bin_pixel_count,
        "bin_pixel_percentage": bin_pixel_count / total_pixels * 100.0,
        "object_pixel_count": object_pixel_count,
        "object_pixel_percentage": object_pixel_count / total_pixels * 100.0,
        "per_wall_pixel_count": per_wall_pixel_count,
        "visible_bin_parts": [name for name, count in per_wall_pixel_count.items() if count > 0],
        "bin_mean_rgb": bin_mean_rgb,
        "object_mean_rgb": object_mean_rgb,
        "surround_mean_rgb": surround_mean_rgb,
        "bin_background_grayscale_contrast": bin_background_grayscale_contrast,
        "bin_object_rgb_distance": bin_object_rgb_distance,
        "bin_background_rgb_distance": bin_background_rgb_distance,
        "frame_looks_blank_or_broken": bool(rgb.std() < 1.0),
    }


def colorize_segmentation(seg: np.ndarray, backend: So101PyBulletBackend) -> np.ndarray:
    palette = {
        -1: (20, 20, 20), backend.robot_id: (200, 200, 0), backend.table_id: (120, 70, 30), backend.object_id: (30, 120, 220),
    }
    for name, bid in backend.bin_body_ids.items():
        if name != "all":
            palette[bid] = (220, 30, 220)
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for body_id, color in palette.items():
        out[seg == body_id] = color
    return out


def analyze_seed(seed: int, bin_center_override_xy: list = None) -> dict:
    sampled_object_position = sample_object_position(seed, DEFAULT_X_RANGE, DEFAULT_Y_RANGE)
    kwargs = {"gui": False, "use_bin": True, "object_position": sampled_object_position}
    if bin_center_override_xy is not None:
        kwargs["bin_center_override_xy"] = bin_center_override_xy
    backend = So101PyBulletBackend(**kwargs)
    try:
        backend.reset()
        rgb, seg = render_rgb_and_segmentation(backend)
        metrics = compute_visibility_metrics(rgb, seg, backend)
        metrics["seed"] = seed
        return metrics, rgb, colorize_segmentation(seg, backend)
    finally:
        backend.close()


def build_contact_sheet(images_by_seed: dict, label: str, out_path: Path) -> None:
    thumb = 200
    grid = Image.new("RGB", (thumb * 2, thumb * len(images_by_seed)), (32, 32, 32))
    draw = ImageDraw.Draw(grid)
    for row, (seed, (rgb, seg_color)) in enumerate(images_by_seed.items()):
        grid.paste(Image.fromarray(rgb).resize((thumb, thumb)), (0, row * thumb))
        grid.paste(Image.fromarray(seg_color).resize((thumb, thumb)), (thumb, row * thumb))
        draw.text((4, row * thumb + 4), f"seed {seed} rgb", fill=(255, 255, 0))
        draw.text((thumb + 4, row * thumb + 4), f"seed {seed} seg", fill=(255, 255, 0))
    grid.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", type=str, required=True, choices=["before", "after"])
    args = parser.parse_args()

    out_dir = OUTPUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    images_by_seed = {}
    all_metrics = {}
    for seed in PREVIEW_SEEDS:
        metrics, rgb, seg_color = analyze_seed(seed)
        all_metrics[seed] = metrics
        images_by_seed[seed] = (rgb, seg_color)
        Image.fromarray(rgb).save(out_dir / f"seed_{seed}_rgb.png")
        Image.fromarray(seg_color).save(out_dir / f"seed_{seed}_segmentation.png")

    contact_sheet_path = out_dir / "contact_sheet_rgb_and_segmentation.png"
    build_contact_sheet(images_by_seed, args.label, contact_sheet_path)

    summary = {
        "label": args.label,
        "seeds": PREVIEW_SEEDS,
        "per_seed_metrics": all_metrics,
        "mean_bin_pixel_percentage": float(np.mean([m["bin_pixel_percentage"] for m in all_metrics.values()])),
        "mean_bin_background_grayscale_contrast": float(np.mean(
            [m["bin_background_grayscale_contrast"] for m in all_metrics.values() if m["bin_background_grayscale_contrast"] is not None]
        )),
        "mean_bin_object_rgb_distance": float(np.mean(
            [m["bin_object_rgb_distance"] for m in all_metrics.values() if m["bin_object_rgb_distance"] is not None]
        )),
        "any_frame_looks_blank_or_broken": any(m["frame_looks_blank_or_broken"] for m in all_metrics.values()),
        "visible_bin_parts_union": sorted(set(p for m in all_metrics.values() for p in m["visible_bin_parts"])),
    }
    summary_path = out_dir / "visibility_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"=== SO-101 bin visual salience measurement ({args.label}) ===")
    print(f"mean_bin_pixel_percentage: {summary['mean_bin_pixel_percentage']:.4f}%")
    print(f"mean_bin_background_grayscale_contrast: {summary['mean_bin_background_grayscale_contrast']:.2f}")
    print(f"mean_bin_object_rgb_distance: {summary['mean_bin_object_rgb_distance']:.2f}")
    print(f"visible_bin_parts_union: {summary['visible_bin_parts_union']}")
    print(f"any_frame_looks_blank_or_broken: {summary['any_frame_looks_blank_or_broken']}")
    print(f"\nContact sheet: {contact_sheet_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
