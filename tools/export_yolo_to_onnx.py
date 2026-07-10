"""Export a pretrained/finetuned Ultralytics YOLO checkpoint to ONNX.

  weights/yolo26n.pt -> weights/yolo26n.onnx

No TensorRT export here yet -- this is just the ONNX export step before
benchmark/run_yolo_onnx_smoke_test.py and (later) ONNX Model Evaluator.

Run directly (not as a module):
  python tools/export_yolo_to_onnx.py --model-path weights/yolo26n.pt
"""

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weights/yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", type=str, default="weights")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--opset", type=int, default=None)
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    model_path = resolve(args.model_path)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        print(
            "Check --model-path. If you haven't downloaded the pretrained "
            "weights yet, run benchmark.run_yolo_safety_monitor_demo once "
            "first (it downloads the .pt on first use), then move it into "
            "weights/."
        )
        sys.exit(1)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = YOLO(str(model_path))
        exported_path = Path(
            model.export(
                format="onnx",
                imgsz=args.imgsz,
                dynamic=args.dynamic,
                opset=args.opset,
            )
        )
    except Exception as exc:
        print(f"ONNX export failed: {exc}")
        print(
            "Check that 'onnx' and 'onnxslim' are installed "
            "(`pip install onnx onnxslim`) and that --model-path points to "
            "a valid Ultralytics .pt checkpoint."
        )
        sys.exit(1)

    final_path = output_dir / exported_path.name
    if exported_path.resolve() != final_path.resolve():
        shutil.move(str(exported_path), str(final_path))

    print(f"Exported ONNX model to: {final_path}")


if __name__ == "__main__":
    main()
