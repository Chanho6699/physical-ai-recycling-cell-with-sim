"""PyBullet virtual camera capture smoke test.

Confirms the "PyBullet scene -> RGB image on disk" path works on its own,
before any perception model is wired up.
"""

import time
from pathlib import Path

from robot_sim.camera_utils import capture_pybullet_camera, save_rgb_image
from robot_sim.pybullet_backend import PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "camera" / "pybullet_camera_capture.png"

GUI_MODE = True
KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def main() -> None:
    backend = PyBulletBackend(gui=GUI_MODE)
    try:
        state = backend.reset()
        print("=== Reset State ===")
        print(state)

        image = capture_pybullet_camera(physics_client_id=backend.client_id)
        saved_path = save_rgb_image(image, str(OUTPUT_PATH))
        print(f"\nSaved camera image to: {saved_path}")

        if KEEP_GUI_OPEN:
            print(f"Keeping PyBullet GUI open (up to {KEEP_SECONDS}s if no input)...")
            try:
                input("Press Enter to close PyBullet GUI...")
            except EOFError:
                time.sleep(KEEP_SECONDS)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
