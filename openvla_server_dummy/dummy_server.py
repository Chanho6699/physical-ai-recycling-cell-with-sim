from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI(title="Dummy OpenVLA Action Server")


class PredictRequest(BaseModel):
    instruction: str
    image_path: Optional[str] = None


class PredictResponse(BaseModel):
    action: List[float]
    inference_ms: float
    model: str
    instruction: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "dummy-openvla",
    }


@app.post("/predict_action", response_model=PredictResponse)
def predict_action(req: PredictRequest):
    start = time.perf_counter()

    # Dummy 7-DoF action:
    # [dx, dy, dz, droll, dpitch, dyaw, gripper]
    action = [0.01, 0.00, 0.02, 0.00, 0.00, 0.00, 1.0]

    inference_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        action=action,
        inference_ms=inference_ms,
        model="dummy-openvla",
        instruction=req.instruction,
    )
