# Split Deployment via Tailscale -- Success Log

## 1. 목적

Robot/Simulation Client와 VLA Inference Server를 **서로 다른 머신으로
분리 배포**하는 구조가 실제로 동작하는지 검증. 로컬 control loop
(PyBullet, safety, recorder)와 VLA 모델 추론이 물리적으로 다른 장치에서
돌아가도, `RealVLAPolicyClient`가 아는 것은 여전히 `/predict` HTTP
엔드포인트 하나뿐이라는 이 프로젝트의 핵심 제약이 네트워크 경계를
넘어서도 유지되는지가 이 검증의 초점이다.

## 2. 구성

```text
Notebook (client 쪽):
  PyBullet simulation, YOLO ONNX detection, Real2Sim mapping,
  RobotBackend, Safety(SafetyGate/SafetySupervisor), Recorder,
  RealVLAPolicyClient

Desktop RTX 3050 (server 쪽):
  Generic VLA Server (vla_server/generic_vla_server.py)
  model_family: mock-action / smolvla / openvla

Network:
  Tailscale private network (두 장치를 같은 tailnet에 연결)

Windows portproxy:
  Tailscale IP:9200 -> WSL 안의 VLA Server:9200
  (Generic VLA Server가 WSL에서 돌기 때문에, Windows 호스트가 Tailscale
   IP로 들어온 트래픽을 WSL 내부 IP:9200으로 넘겨줘야 함)
```

## 3. 성공 로그 요약

```text
policy_backend: real-vla
policy_server_url: http://100.75.147.72:9200/predict
model: generic-mock-action
fallback_used_count: 0
avg_inference_latency_ms: 20.383
final_status: success
PASS
```

`fallback_used_count: 0`은 매 스텝이 실제로 원격 서버의 `/predict`
응답을 사용했다는 뜻 -- 로컬 fallback(`local-dummy`)으로 넘어간 적이
없다.

## 4. 의미

- **로봇 제어 루프와 VLA 추론 서버가 네트워크 경계로 분리됨** -- 같은
  머신일 필요가 없고, Tailscale 같은 private network만 있으면 서로 다른
  물리 장치/OS(WSL) 사이에서도 기존 `/predict` 스키마가 그대로 동작한다.
- **VLA 서버 장애 시 fallback 가능** -- 서버가 죽거나 응답하지 않아도
  `RealVLAPolicyClient`의 client-side fallback(`local-dummy` 등)이
  전체 데모를 계속 진행시킨다 (이번 실행에서는 필요하지 않았지만, 구조
  자체는 검증된 기존 동작).
- **mock-action을 SmolVLA/OpenVLA로 교체 가능한 구조** -- 이번 성공은
  `model_family=mock-action`으로 배관(plumbing) 자체를 검증한 것이고,
  같은 서버/네트워크 구성에서 `VLA_MODEL_FAMILY=smolvla`(또는
  `openvla`)로 바꾸는 것만으로 실제 모델로 전환 가능하다 -- 로컬 코드는
  변경할 필요 없음.

## 5. 재현 순서

1. **데스크탑에서 VLA 서버 실행** (WSL, RTX 3050):
   ```bash
   VLA_MODEL_FAMILY=mock-action \
     uvicorn vla_server.generic_vla_server:app --host 0.0.0.0 --port 9200
   ```
2. **Tailscale IP 확인** (데스크탑 쪽):
   ```bash
   tailscale ip -4
   ```
3. **Windows portproxy 설정** (관리자 권한 PowerShell, Tailscale IP와
   WSL 내부 IP를 실제 값으로 교체):
   ```powershell
   netsh interface portproxy add v4tov4 `
     listenaddress=<Tailscale IP> listenport=9200 `
     connectaddress=<WSL 내부 IP> connectport=9200
   ```
4. **노트북에서 local config 작성** -- `server_url`/`health_url`을
   `http://<Tailscale IP>:9200/predict` / `/health`로 가리키는
   `configs/vla_backend_*_local.json` 작성 (Git에는 올리지 않음, 6번
   참고).
5. **노트북에서 full demo 실행**:
   ```bash
   python -m benchmark.run_full_recycling_cell_demo \
     --policy dummy-openvla --policy-backend real-vla \
     --real-vla-config configs/vla_backend_mock_action_local.json \
     --real-vla-fallback-backend local-dummy \
     --instruction "플라스틱 병을 플라스틱 수거함에 넣어줘" \
     --image-path data/test_images/recyclable_scene.jpg \
     --headless
   ```

## 6. 주의사항

- **`configs/*local*.json`은 Git에 올리지 않음** -- Tailscale IP처럼
  장치/세션마다 달라지는 값을 담고 있어 커밋 대상이 아니다
  (`.gitignore`에 `configs/*local*.json` 패턴 추가됨).
- **Tailscale Funnel 사용하지 않음** -- 이 구성은 tailnet 내부 전용
  private network만 쓰고, 외부 공개 터널(Funnel)은 사용하지 않는다.
- **WSL IP가 바뀌면 portproxy 재설정 필요** -- WSL의 내부 IP는 재부팅 시
  바뀔 수 있어, `netsh interface portproxy` 규칙을 새 IP로 다시 설정해야
  한다.
- **지금 성공한 것은 mock-action backend이며, SmolVLA 실모델 로딩은
  다음 단계** -- 이 문서는 네트워크/배포 구조가 동작한다는 것만
  검증한다. 실제 SmolVLA 추론 성공 여부는
  [docs/vla_integration_spike_log.md](vla_integration_spike_log.md)와
  [docs/smolvla_cloud_loading_spike.md](smolvla_cloud_loading_spike.md)
  에서 별도로 다룬다.

## See also

- [docs/generic_vla_backend.md](generic_vla_backend.md) -- Generic VLA Server의 adapter/loader/registry 아키텍처
- [docs/vla_integration_spike_log.md](vla_integration_spike_log.md) -- VLA 모델 통합 스파이크 전체 로그
- [docs/smolvla_cloud_loading_spike.md](smolvla_cloud_loading_spike.md) -- SmolVLA 로딩 스파이크, `VLA_ALLOW_VLM_FALLBACK` 상세
- [docs/hardware_portability.md](hardware_portability.md) -- 하드웨어 이식성 전체 그림
