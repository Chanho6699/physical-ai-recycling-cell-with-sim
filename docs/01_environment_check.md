# 01. Environment Check

## GPU

```text
GPU: NVIDIA GeForce RTX 3050
VRAM: 8192 MiB
Driver Version: 560.94
CUDA Version: 12.6
Current GPU Memory Usage: 753 MiB / 8192 MiB

Feasibility
ONNX / TensorRT

YOLO ONNX and TensorRT FP16 optimization are feasible on RTX 3050 8GB.

Isaac Sim

Isaac Sim may be constrained by 8GB VRAM. The first target is a minimal scene:

empty scene
single cube
single camera
lightweight robot arm
scripted movement
OpenVLA

OpenVLA 7B full local inference is likely infeasible on 8GB VRAM.

Initial OpenVLA strategy:

build dummy OpenVLA server first
validate image + instruction → action API
later replace dummy server with real OpenVLA server on cloud GPU