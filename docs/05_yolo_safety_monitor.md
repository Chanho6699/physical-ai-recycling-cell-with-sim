# 05. YOLO Safety Monitor

## YOLOSafetyMonitor의 역할

로봇이 `RobotCommand`를 실행하기 전에, 카메라 프레임을 보고 위험 요소(기본: `person`)가 있는지 확인해서 위험하면 command 적용 자체를 막는 안전 게이트입니다.

```text
FrameSource.get_frame()
→ SafetyMonitor.check(frame)
→ SafetyDecision(emergency_stop, reason, detections)
→ 위험하면 RobotCommand 적용 차단
→ 안전하면 PyBulletBackend.apply_command() 실행
```

`MockSafetyMonitor`로 게이트 구조 자체를 먼저 검증했고, 이제 `YOLOSafetyMonitor`가 실제 pretrained YOLO 모델(`weights/yolo26n.pt`)로 이 판단을 대신합니다.

## PyTorch baseline

`YOLOSafetyMonitor`는 현재 Ultralytics `YOLO(model_path)`를 그대로 로드해서 PyTorch로 추론합니다. 학습(fine-tuning)은 하지 않고 COCO pretrained nano 모델을 그대로 사용합니다. 이 프로젝트의 로컬 GPU(RTX 3050, 드라이버 이슈로 CUDA 미사용)에서는 CPU로 추론하며, 정확도/구조 검증이 목적이라 속도는 아직 최적화 대상이 아닙니다.

## ONNX export 목적

```text
weights/yolo26n.pt → weights/yolo26n.onnx (tools/export_yolo_to_onnx.py)
→ ONNX smoke test (benchmark/run_yolo_onnx_smoke_test.py)
```

ONNX export는 다음을 위한 중간 단계입니다.

- PyTorch 런타임 의존성 없이 추론 가능한 형태로 모델을 고정
- 나중에 ONNX Runtime, TensorRT 등 다른 백엔드로 넘어가기 위한 공통 포맷
- 이 프로젝트에 이미 있는 ONNX Model Evaluator 도구와 바로 연결 가능

## ONNX Evaluator와 연결하는 이유

이 프로젝트는 원래 YOLO + ONNX + TensorRT 기반 스마트 재활용 분류기에서 출발했고, `results/onnx_eval/`에 ONNX 모델 정확도/속도를 비교하는 evaluator가 이미 있습니다. `weights/yolo26n.onnx`를 그 evaluator에 그대로 입력으로 넘기면, 이번에 만든 safety 모델도 동일한 기준(정확도, latency)으로 다른 ONNX 모델들과 비교할 수 있습니다.

## TensorRT는 다음 단계로 남김

이번 단계에서는 TensorRT 변환을 하지 않습니다. ONNX 모델이 정상적으로 export/로드/추론되는 것까지만 확인하고, RTX 3050 8GB에서의 TensorRT FP16 엔진 변환 및 벤치마크는 다음 단계로 남겨둡니다.

## SafetyMonitor 인터페이스 덕분에 구현체를 교체할 수 있음

```text
SafetyMonitor (ABC)
├── MockSafetyMonitor       고정된 안전/위험 판정 (게이트 구조 검증용)
├── YOLOSafetyMonitor        PyTorch(.pt) 기반 실제 추론
└── (미래) ONNXSafetyMonitor / TensorRTSafetyMonitor
```

모든 구현체는 `check(frame: np.ndarray) -> SafetyDecision`만 지키면 되므로, `benchmark/run_pybullet_safety_gate_demo.py` 같은 게이트 코드는 어떤 구현체를 쓰든 전혀 수정할 필요가 없습니다. ONNX/TensorRT 기반 SafetyMonitor는 이번 단계에서 만들지 않았고, ONNX export + smoke test까지만 검증했습니다.
