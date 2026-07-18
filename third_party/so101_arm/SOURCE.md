SO-101 simulation assets, vendored from the official repository.

- Source repository: https://github.com/TheRobotStudio/SO-ARM100
- Path within repo: `Simulation/SO101/`
- Commit: `fda892cba81032c46c40976a48c9ceadbf40a9ca`
- Commit date: 2026-02-26T14:29:08Z
- Fetched: 2026-07-18
- License: Apache License 2.0 (see `LICENSE` in this directory, copied unmodified from the source repo root)

Files copied unmodified:
- `LICENSE` (from repo root)
- `README.md` (from `Simulation/SO101/README.md`)
- `so101_new_calib.urdf` (default/recommended calibration per the SO101 README -- joint zero at the middle of each joint's range)
- `so101_old_calib.urdf` (alternate calibration -- joint zero at full horizontal extension; not used by this project's inspection/smoke scripts, kept for reference only)
- `assets/*.stl` (11 mesh files referenced by both URDFs via relative `assets/...stl` paths, already relative -- not `package://`, so no path rewriting was needed to load them from this project's own directory layout)

Not copied (not needed for a URDF+PyBullet pipeline):
- `*.part` CAD source files (Onshape/CAD-tool specific, not consumed by PyBullet)
- `*.xml` MuJoCo MJCF files and `scene.xml`/`joints_properties.xml` (this project uses PyBullet, not MuJoCo)

No modifications were made to the URDF or mesh files themselves.
