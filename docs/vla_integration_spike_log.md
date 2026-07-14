# VLA Integration Spike Log

## 1. 목적

OpenVLA/SmolVLA 같은 실제 VLA(Vision-Language-Action) 모델을, 기존 로컬
robot control loop(PyBullet, `SafetyGate`/`SafetySupervisor`,
`RobotBackend`, episode recording)와 완전히 분리된 별도의 inference
server로 연결하는 것. 로컬 control loop는 어떤 모델이 실제로 응답하는지
전혀 몰라야 하고, 서버 쪽 모델을 바꾸는 것이 로컬 코드 변경으로 이어지지
않아야 한다는 것이 핵심 제약이다.

## 2. 현재 구현된 것

- **Generic VLA Server** (`vla_server/generic_vla_server.py`) --
  `model_family`에 관계없이 동일한 `/health`, `/load_model`, `/predict`,
  `/reset` 스키마로 응답하는 FastAPI 서버.
- **`model_family` 기반 adapter 구조** (`vla_adapters/*.py` +
  `vla_server/model_registry.py`) -- 모델별 raw output -> normalized
  action 변환만 담당하고, 모델 객체/네트워크는 절대 건드리지 않는다.
- **`SmolVLAActionAdapter`** -- 여러 가능한 raw output shape(플랫 7-vector,
  `{"action":...}`, 청크/배치 형태, numpy/torch tensor)을 재귀적으로 해석.
- **`OpenVLAActionAdapter`** -- 실제 action space 검증 전까지는 항상
  `action=None` + `openvla_action_adapter_required`를 반환하도록 의도적으로
  막아둔 상태.
- **`RealVLAPolicyClient`와 HTTP `/predict` 연결** -- 로컬 control loop가
  이 클라이언트 하나만 알면 되고, 서버가 mock-action/SmolVLA/OpenVLA 중
  무엇이든 동일한 요청/응답 스키마로 동작.
- **`VLA_ALLOW_VLM_FALLBACK=0`로 잘못된 SmolVLM fallback 방지** --
  `_load_smolvla()`가 LeRobot SmolVLA policy import에 실패했을 때, 기본값
  에서는 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 같은 VLM backbone을
  다운로드하는 fallback을 타지 않고 즉시 `model_status=load_failed`로
  끝난다 (`docs/smolvla_cloud_loading_spike.md` 참고).

## 3. Colab 실험 결과

- **mock-action 서버 연결 성공** -- Generic VLA Server를 Colab에서 띄우고
  `RealVLAPolicyClient`가 `/health`/`/predict`를 통해 정상적으로 통신하는
  것을 확인.
- **`SmolVLAPolicy` import path 확인 성공** -- LeRobot 설치본에서 실제
  SmolVLA policy 클래스가 위치한 import 경로를 `pkgutil` 기반 검색으로
  찾아냄.
- **`/load_model` 단계에서 Hugging Face backbone weight 다운로드가
  정체됨** -- 실제 SmolVLA 체크포인트 로드 시도 중 Hugging Face 쪽 weight
  다운로드가 멈추거나 매우 느려지는 현상이 Colab 환경에서 발생.

## 4. 결론

현재 병목은 **robot control code의 문제가 아니라 model weight
download/runtime 환경 문제**다. adapter/server/client 아키텍처는 이미
동작을 확인했고, 막혀 있는 지점은 순수하게 "무료/공유 Colab 환경에서 큰
체크포인트를 안정적으로 내려받고 로드할 수 있는가"라는 인프라 문제다.
Colab 대신 RTX 3050 로컬 환경 또는 향후 RunPod 서버에서 재검증할 예정.

## 5. 다음 단계

1. 로컬 Generic VLA Server 단독으로 `/load_model` 테스트 (Colab 없이,
   RTX 3050 환경에서 다운로드/로드가 끝까지 진행되는지 확인).
2. 성공 시 PyBullet headless demo와 연결해 실제 control loop에서 동작
   확인.
3. OOM(메모리 부족) 발생 시 RunPod 등 더 큰 GPU 인스턴스로 이동.
4. ONNX/TensorRT/INT8 최적화는 **실제 PyTorch 추론이 먼저 성공한 이후의
   최적화 단계**로 보류 -- 지금 단계에서 다루지 않는다.

## 6. 주의

- 아직 **SmolVLA 실추론 성공이라고 쓰지 않는다** -- 지금까지 확인된 것은
  import path와 서버/클라이언트 연결뿐, 실제 forward pass 결과는 아직
  없다.
- **TensorRT INT8 적용 완료라고 쓰지 않는다** -- 최적화 단계는 아직 시작
  전이다.
- 현재는 **integration spike 단계**라는 것을 명확히 한다 -- 이 문서의
  목적은 아키텍처가 동작한다는 것과, 남아 있는 병목이 어디인지를
  기록하는 것이지, 최종 성공을 보고하는 것이 아니다.

## See also

- [docs/generic_vla_backend.md](generic_vla_backend.md) -- adapter/loader/registry 아키텍처
- [docs/smolvla_cloud_loading_spike.md](smolvla_cloud_loading_spike.md) -- Colab SmolVLA 로딩 스파이크, `VLA_ALLOW_VLM_FALLBACK` 상세
- [docs/colab_vla_server_spike.md](colab_vla_server_spike.md) -- OpenVLA 전용 선행 스파이크
- [docs/hardware_portability.md](hardware_portability.md) -- 하드웨어 이식성 전체 그림
