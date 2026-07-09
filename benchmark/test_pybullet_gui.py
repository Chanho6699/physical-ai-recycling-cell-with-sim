"""Minimal PyBullet GUI smoke test.

Connects to PyBullet with p.GUI directly (no PyBulletBackend, no
pipeline) so a WSL/display problem can be told apart from a code
problem. Creates a plane and a cube, steps simulation for ~30s, prints
connection state, then waits for Enter before disconnecting.
"""

import time

import pybullet as p
import pybullet_data

KEEP_SECONDS = 30


def main() -> None:
    client_id = p.connect(p.GUI)
    print(
        f"[test_pybullet_gui] connection_mode=p.GUI, client_id={client_id}, "
        f"isConnected={p.isConnected(client_id)}"
    )

    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, -9.8, physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], physicsClientId=client_id
    )
    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1.0, 0.0, 0.0, 1.0], physicsClientId=client_id
    )
    p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0.0, 0.0, 0.5],
        physicsClientId=client_id,
    )

    start = time.time()
    while time.time() - start < KEEP_SECONDS:
        p.stepSimulation(physicsClientId=client_id)
        time.sleep(1.0 / 240.0)

    print(f"[test_pybullet_gui] isConnected after {KEEP_SECONDS}s of stepping: {p.isConnected(client_id)}")

    try:
        input("Press Enter to close PyBullet GUI...")
    except EOFError:
        pass

    p.disconnect(client_id)


if __name__ == "__main__":
    main()
