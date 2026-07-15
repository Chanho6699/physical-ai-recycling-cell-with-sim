# Panda Base-Frame Axis Cross-Verification (robosuite vs. PyBullet)

Generated: 2026-07-15T13:17:35.710842+00:00

Both checks are *real simulation results*, not a config-file comparison.

## 1. Delta-application test

Tolerance: intended-axis displacement must be positive and at least 5.0x larger than the largest cross-axis component.

| Axis | PyBullet delta (m) | robosuite delta (m) | Same sign | Passed |
|---|---|---|---|---|
| +X | [0.049935102462768555, -1.8838745745597407e-06, -6.48200511932373e-05] | [0.005483016342695213, -4.529797232073816e-05, -0.0004901650452084905] | True | True |
| +Y | [-6.827712059020996e-05, 0.049975586183791165, -6.207823753356934e-05] | [-6.196265631340514e-05, 0.0051660919022339085, -2.295402957508408e-05] | True | True |
| +Z | [-5.221366882324219e-05, -1.669784978730604e-06, 0.049926042556762695] | [-0.0002971492520658786, -2.4532260359533264e-05, 0.004706701580931227] | True | True |

## 2. Forward-kinematics test

Both simulators set to identical READY_JOINT_POSITIONS=[0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483], EE position read relative to robot base with no controller involved. Tolerance: 0.02 m per axis.

- PyBullet EE relative to base: [0.3068690001964569, -9.325392966275103e-06, 0.48526519536972046]
- robosuite EE relative to base: [0.3068905665929409, 6.5635692504717e-34, 0.49378205230283945]
- abs diff: [2.1566396483985173e-05, 9.325392966275103e-06, 0.008516856933118988]
- Passed: True

## Verdict

**axis_convention_verified = True**
