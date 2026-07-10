# 06. Image-Based Safety Test

## 1. 이 단계의 목적

웹캠이나 Isaac Sim, 실제 로봇 없이도 **정적 이미지 파일만으로** `ONNXRuntimeYOLOSafetyMonitor`가 다음을 정확히 수행하는지 반복 검증하기 위한 구조입니다.

```text
person/hand 등 위험 이미지 → emergency_stop=True
사람이 없는 안전한 이미지  → emergency_stop=False
```

WSL 환경에서는 웹캠이 항상 잡히는 게 아니고, PyBullet 시뮬레이션 장면에는 실제 "person"이 나오지 않기 때문에, safety monitor의 hazard 감지 자체를 검증하려면 실제 사람이 있는 이미지가 필요합니다. 이미지 파일 기반 테스트는 이 부분을 가장 간단하고 재현 가능하게 확인하는 방법입니다.

## 2. 테스트 이미지 준비 방법

테스트 이미지는 다음 폴더에 둡니다.

```text
data/test_images/
```

이 폴더는 Git에 구조만 남고(`data/test_images/.gitkeep`), 실제 이미지 파일(`*.jpg`, `*.jpeg`, `*.png`)은 커밋되지 않습니다. 즉 이미지는 각자 로컬에 준비해서 넣어야 합니다. 준비 방법 예:

- 본인 휴대폰/웹캠으로 찍은 사진을 옮겨 담기
- 공개 라이선스의 사람이 포함된 샘플 이미지 사용 (예: Ultralytics 패키지에 번들된 `zidane.jpg`, `bus.jpg`)
- PyBullet 카메라로 캡처한 장면을 "안전 이미지"로 재사용 (`robot_sim.camera_utils.capture_pybullet_camera` + `save_rgb_image`)

## 3. 위험 이미지 예시

`data/test_images/person_hazard.jpg` — 사람이 뚜렷하게 나오는 사진. (이 프로젝트 검증 시에는 Ultralytics에 번들된 `zidane.jpg`를 그대로 복사해서 사용했습니다.)

## 4. 안전 이미지 예시

`data/test_images/safe_workspace.jpg` — 사람이 없는 장면. PyBullet 작업 공간 카메라 캡처(테이블/재활용 물체/수거함만 보이는 장면)를 그대로 사용해도 됩니다.

## 5. 실행 명령

```bash
source .venv/bin/activate

# 위험 이미지
python -m benchmark.run_onnx_yolo_safety_monitor_demo \
  --source image \
  --image-path data/test_images/person_hazard.jpg \
  --save-debug-image

# 안전 이미지
python -m benchmark.run_onnx_yolo_safety_monitor_demo \
  --source image \
  --image-path data/test_images/safe_workspace.jpg \
  --save-debug-image
```

## 6. 기대 결과

위험 이미지:

```text
emergency_stop: True
reason: hazard_detected:person
```

안전 이미지:

```text
emergency_stop: False
reason: safe
```

두 경우 모두 `--save-debug-image`를 주면 detection bbox가 그려진 이미지가 다음 위치에 저장됩니다 (매 실행마다 덮어씀).

```text
results/camera/onnx_yolo_safety_debug.png
```

## 7. 주의사항

- `--image-path`를 빠뜨리거나 존재하지 않는 경로를 주면 트레이스백 없이 안내 메시지만 출력하고 종료합니다.
- `results/camera/onnx_yolo_safety_debug.png`는 Git에 올라가지 않습니다(`.gitignore`에 `results/camera/*.png` 포함).
- 이번 단계는 이미지 파일 기반 검증까지입니다. 실제 웹캠 연결, Real2Sim 좌표 매핑, TensorRT 변환, 모델 학습, OpenVLA 실제 서버 연동, end-to-end latency 벤치마크는 다음 단계로 남겨둡니다.
