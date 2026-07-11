# 07. PyBullet Franka Panda Backend

## 1. 이 단계의 목적

지금까지의 `PyBulletBackend`는 end-effector를 구(sphere)로 표현하고 거리 기반으로 위치를 순간이동(teleport)시키는 manipulation abstraction이었습니다. 이번 단계는 PyBullet에 내장된 실제 Franka Panda URDF를 로드하고, IK(역기구학) + joint motor control로 실제 로봇팔처럼 움직이는 `PyBulletPandaBackend`를 추가해서, "진짜 로봇팔을 통한 pick-and-place"가 이 프로젝트의 인터페이스(`SimulatorBackend`) 안에서 동작하는지 검증합니다.

## 2. Simple PyBulletBackend와 PyBulletPandaBackend의 차이

| | `PyBulletBackend` (simple) | `PyBulletPandaBackend` (v1) |
|---|---|---|
| End-effector 표현 | 빨간 sphere (충돌 없음, mass=0) | 실제 Franka Panda URDF (7-DOF 팔 + 2-finger gripper) |
| 이동 방식 | `resetBasePositionAndOrientation`으로 즉시 순간이동 | `calculateInverseKinematics()` + `setJointMotorControlArray()`로 실제 관절을 물리 기반으로 구동 |
| Object 추종(held 상태) | 매 스텝 `object_position = ee_position + offset`로 수동 갱신 | `p.createConstraint()`로 물리적으로 부착 — 물리 엔진이 자동으로 따라가게 함 |
| Grasp 판정 | 거리 기반 (`GRASP_THRESHOLD`) | 거리 기반은 동일하지만, 실제 충돌 형상(gripper finger, object, table)이 있어 물리적 간섭이 발생할 수 있음 |
| `get_state()` | 6-DoF pose 중심 (`end_effector_position`, `gripper_state` 등) | 여기에 `joint_positions`, `joint_velocities`, `end_effector_orientation`, `gripper_width` 등 실제 로봇팔 상태 추가 |

두 backend 모두 같은 `SimulatorBackend` 인터페이스(`reset`/`apply_command`/`get_state`/`close`)를 따르므로, 상위 파이프라인(Task pipeline 등)은 어떤 backend를 쓰는지 몰라도 됩니다.

## 3. Panda URDF / IK / joint motor control 설명

PyBullet에 번들된 `franka_panda/panda.urdf`는 12개 joint를 가집니다. 실제로 `print_joint_info()`로 확인한 구조:

```text
joint_index=0..6   panda_joint1..7        (revolute)  -- 7-DOF 팔
joint_index=7      panda_joint8            (fixed)
joint_index=8      panda_hand_joint        (fixed)     -- hand 링크
joint_index=9,10   panda_finger_joint1/2   (prismatic) -- 그리퍼 손가락 (각각 0~0.04m)
joint_index=11     panda_grasptarget_hand  (fixed)     -- 손가락 사이 가상의 grasp 지점
```

- **`arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]`**, **`finger_joint_indices = [9, 10]`**
- **`end_effector_link_index = 11`** (`panda_grasptarget`) — link 8(`panda_hand`)이 아니라 11을 쓰는 이유는, 11이 실제로 손가락 사이의 "잡는 지점"에 위치해서 IK 타깃으로 삼기에 더 적절하기 때문입니다(직접 forward kinematics로 두 링크의 위치를 비교해 확인함).

`move_end_effector_to(target_position, target_orientation)`는 다음 순서로 동작합니다.

```text
target_position, target_orientation
→ p.calculateInverseKinematics(robot_id, end_effector_link_index, ...)
→ 7개 팔 관절 목표각 (IK가 반환하는 9개 값 중 앞 7개, 손가락 2개는 무시하고 별도 제어)
→ p.setJointMotorControlArray(POSITION_CONTROL, forces=URDF의 관절별 maxForce)
→ p.stepSimulation() 반복
```

`target_orientation`이 `None`이면, `reset()` 시점에 "ready pose"(Franka의 표준 대기 자세: `[0, -π/4, 0, -3π/4, 0, π/2, π/4]`)에서 forward kinematics로 읽은 실제 그리퍼-아래방향 orientation을 기본값으로 사용합니다.

## 4. grasp constraint를 사용하는 이유

Panda 그리퍼(손가락 2개, 각 20N)만으로 마찰력 기반 contact grasp을 구현하면, 물체 크기/마찰계수/근사 형상 차이 때문에 물체가 미끄러지거나 손가락 사이에서 튕겨 나가는 등 불안정해지기 쉽습니다. v1에서는 안정성을 우선해서:

```text
close_gripper() 이후, end-effector와 object 사이 거리가 GRASP_THRESHOLD(0.05m) 이하이면
→ p.createConstraint(JOINT_FIXED)로 object를 end-effector link에 물리적으로 고정
→ held_object=True, task_status="grasped"
```

이때 object의 **현재 위치를 기준으로 상대 오프셋을 계산**해서 constraint를 걸기 때문에(`invertTransform`/`multiplyTransforms` 사용), 물체가 그 순간 갑자기 end-effector 원점으로 순간이동(snap)하지 않습니다. `open_gripper()` 시 held 상태이면 constraint를 제거하고, bin과의 거리가 `PLACE_THRESHOLD(0.08m)` 이하면 `task_status="success"`, 아니면(엉뚱한 곳에서 놓으면) `task_status="released"`로 구분합니다.

## 5. 실행 명령

```bash
source .venv/bin/activate

# 기본 (GUI, 기본 좌표)
python -m benchmark.run_pybullet_panda_pick_place_demo

# headless + joint 정보 출력
python -m benchmark.run_pybullet_panda_pick_place_demo --headless --print-joint-info

# 좌표 직접 지정
python -m benchmark.run_pybullet_panda_pick_place_demo \
  --object-x 0.45 --object-y 0.0 --object-z 0.05 \
  --bin-x 0.3 --bin-y 0.35 --bin-z 0.05
```

## 6. 기대 결과

```text
=== Move to object ===
end_effector_position ≈ object_position

=== Close gripper ===
held_object=True, task_status=grasped, last_event=object_grasped
gripper_width ≈ 0.04 (손가락이 물체 폭에서 멈춤, 완전히 0까지 안 닫힘)

=== Move to bin ===
held_object=True 유지, object_position이 end-effector를 따라 bin 근처로 이동

=== Open gripper ===
held_object=False, task_status=success, last_event=object_placed_in_bin

=== Demo finished: task_status=success ===
PASS
```

실제로 실행해 위 결과를 확인했습니다. 참고로 "Move to bin"의 목표 z는 `bin_z` 그대로가 아니라 `bin_z + 0.05` (bin 위 약간 높은 지점)를 사용합니다 — bin이 속이 빈 통이 아니라 충돌체가 있는 solid box라서, 물체를 들고 그 높이까지 내리려 하면 bin과 충돌해 IK 동작이 목표에 도달하지 못하고 멈추는 현상을 실제로 확인했기 때문입니다.

## 7. 다음 단계: Real2Sim position → Panda target pose 연결

지금은 `--object-x/y/z`, `--bin-x/y/z`로 좌표를 직접 지정하지만, 다음 단계에서는:

```text
image → ONNXYOLODetector → RecyclableObjectMapper
→ ImageToSimMapper.image_point_to_sim_position()
→ PyBulletPandaBackend.set_object_position(sim_position)
→ move_end_effector_to(sim_position) 로 바로 접근
```

이렇게 연결하면 됩니다. `PyBulletPandaBackend`는 이미 `set_object_position`/`set_object_type`을 갖추고 있어 기존 `run_task_goal_real2sim_demo.py`의 Real2Sim 매핑 부분은 그대로 재사용하고, pick-and-place 실행 부분만 (`robot_sim.pick_place_policy.run_dynamic_pick_place` 대신) `move_end_effector_to`/`close_gripper`/`open_gripper` 시퀀스로 교체하면 됩니다. 다만 이번 단계에서는 이 연결을 하지 않았습니다(요구사항에 따라 다음 단계로 남김).
