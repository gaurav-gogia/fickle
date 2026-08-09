from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


class LambdaLayer(nn.Module):
    """Wraps a callable as an nn.Module, mirroring Keras's Lambda layer."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ################
        # YOUR CODE HERE
        ################
        return self.fn(x)


def tool_ready_transform(t: torch.Tensor) -> torch.Tensor:
    scaled = t * 2.0
    shifted = scaled + 0.5
    return torch.tanh(shifted)


class SimpleTorchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            LambdaLayer(tool_ready_transform),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def export_to_onnx(path: Path) -> None:
    model = SimpleTorchModel().eval()
    dummy_input = torch.randn(1, 4, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["features"],
        output_names=["prediction"],
        dynamic_axes={"features": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=17,
    )


def load_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_inference(session: ort.InferenceSession) -> np.ndarray:
    sample = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    outputs = session.run(output_names=None, input_feed={"features": sample})
    return outputs[0]


if __name__ == "__main__":
    out_file = Path("model.onnx")
    export_to_onnx(out_file)

    session = load_session(out_file)
    pred = run_inference(session)

    print(f"ONNX model exported to: {out_file}")
    print(f"ONNX prediction shape: {pred.shape}")
