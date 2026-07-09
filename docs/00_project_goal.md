# Physical AI Recycling Cell

## Goal

This project redefines a YOLO-based smart recycling classifier as a simulated physical AI robot cell.

The final goal is to build a simulation-based robotic recycling pipeline using:

- Isaac Sim for digital twin robot simulation
- LLM Agent for high-level task planning
- OpenVLA for vision-language-action policy inference
- Action Adapter for converting VLA actions to robot commands
- YOLO + ONNX + TensorRT for optimized perception baseline
- ONNX Model Evaluator as an external model evaluation tool

## Architecture

User Command
→ LLM Agent
→ Task Goal JSON
→ Isaac Sim Camera Image
→ OpenVLA Inference Server
→ 7-DoF Action
→ Action Adapter
→ Isaac Sim Robot Control
→ Benchmark Logger

## Initial Strategy

Because the local GPU is RTX 3050 8GB, OpenVLA 7B will not be loaded locally at first.

The system will be split into:

- Local PC: Isaac Sim, ROS 2, Action Adapter, YOLO TensorRT baseline
- Cloud GPU: OpenVLA inference server

Before using the real OpenVLA model, a dummy FastAPI server will be implemented to validate the communication and action format.
