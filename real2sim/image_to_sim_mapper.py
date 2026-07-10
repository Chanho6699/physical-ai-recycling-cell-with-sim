"""Image pixel coordinate -> approximate PyBullet table coordinate mapping (v0).

This is NOT camera calibration -- there is no intrinsics/extrinsics model,
no lens distortion correction, no depth estimation. It linearly stretches
image-space (x, y) onto a fixed rectangular region of the PyBullet table,
purely to validate the Real2Sim wiring (detect -> map -> place) end to
end. A precise Real2Sim mapping is left for a future step.
"""

from typing import Tuple


class ImageToSimMapper:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        sim_x_range: Tuple[float, float] = (-0.4, 0.6),
        sim_y_range: Tuple[float, float] = (-0.3, 0.4),
        object_z: float = 0.53,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.sim_x_range = sim_x_range
        self.sim_y_range = sim_y_range
        self.object_z = object_z

    def image_point_to_sim_position(self, x: float, y: float) -> list:
        x_ratio = x / self.image_width
        # Image y grows downward; PyBullet y is treated as growing "up" in
        # this mapping, so the ratio is flipped.
        y_ratio = 1.0 - (y / self.image_height)

        sim_x = self.sim_x_range[0] + x_ratio * (self.sim_x_range[1] - self.sim_x_range[0])
        sim_y = self.sim_y_range[0] + y_ratio * (self.sim_y_range[1] - self.sim_y_range[0])

        return [sim_x, sim_y, self.object_z]
