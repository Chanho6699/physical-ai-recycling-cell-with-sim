"""Isaac Sim minimal scene smoke test.

Goal: verify that Isaac Sim can launch on this machine, build an empty
scene, add a cube and a camera, step the simulation a few times, and save
a camera image. No robot arm, no ROS 2, no OpenVLA here.

NOTE: Isaac Sim version에 따라 import 경로가 달라질 수 있으므로,
실제 설치된 Isaac Sim 버전에 맞게 아래 import 문을 수정해야 한다.
  - Isaac Sim 2023.1.x 이하: from omni.isaac.kit import SimulationApp
                             from omni.isaac.core import World
                             from omni.isaac.core.objects import DynamicCuboid
                             from omni.isaac.sensor import Camera
  - Isaac Sim 4.x 이후 (isaacsim 네임스페이스로 일부 이동):
                             from isaacsim import SimulationApp
                             from isaacsim.core.api import World
                             from isaacsim.core.api.objects import DynamicCuboid
                             from isaacsim.sensors.camera import Camera
Run this file with the Python interpreter bundled with Isaac Sim
(e.g. `./python.sh`), not with a regular system/venv Python.
"""

from pathlib import Path

# SimulationApp must be created before any other omni.isaac.* / isaacsim.* import.
from isaacsim import SimulationApp  # noqa: E402  (version-dependent, see note above)

simulation_app = SimulationApp({"headless": True})  # set headless=False to see the GUI

# All other Isaac Sim imports must come after SimulationApp() is constructed.
from omni.isaac.core import World  # noqa: E402  (version-dependent, see note above)
from omni.isaac.core.objects import DynamicCuboid  # noqa: E402
from omni.isaac.sensor import Camera  # noqa: E402

NUM_SIMULATION_STEPS = 60

# Where a captured camera frame would be written. Not created by this
# minimal script yet; wire up an actual image save (e.g. via the
# `Camera` sensor's `get_rgba()` + a PNG writer) once Isaac Sim access
# is confirmed to work end to end.
SCREENSHOT_DIR = Path(__file__).resolve().parents[1] / "results" / "isaacsim"
SCREENSHOT_PATH = SCREENSHOT_DIR / "minimal_scene_camera.png"


def build_scene() -> World:
    world = World()
    world.scene.add_default_ground_plane()

    world.scene.add(
        DynamicCuboid(
            prim_path="/World/dummy_cube",
            name="dummy_cube",
            position=[0.0, 0.0, 0.5],
            scale=[0.1, 0.1, 0.1],
        )
    )

    return world


def add_camera() -> Camera:
    camera = Camera(
        prim_path="/World/dummy_camera",
        position=[1.5, 0.0, 1.0],
        resolution=(640, 480),
    )
    camera.initialize()
    return camera


def main() -> None:
    world = build_scene()
    camera = add_camera()

    world.reset()

    for _ in range(NUM_SIMULATION_STEPS):
        world.step(render=True)

    # TODO: once this runs, save the last camera frame, e.g.:
    #   frame = camera.get_rgba()
    #   SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    #   <write `frame` to SCREENSHOT_PATH with e.g. PIL or matplotlib>
    print(f"[minimal_scene] would save camera frame to: {SCREENSHOT_PATH}")

    simulation_app.close()


if __name__ == "__main__":
    main()
