from __future__ import annotations

from pathlib import Path
from typing import Callable, TypedDict

import os
import torch
from torch import nn


class TorchModelState(TypedDict):
    state_dict: dict[str, torch.Tensor]
    tool_name: str | None


ToolPayload = dict[str, object]
ToolFn = Callable[[ToolPayload], ToolPayload]


class SimpleTorchModel(nn.Module):
    def __init__(self, tool_name: str | None = None) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.tool_name = tool_name
        self.tool: ToolFn | None = None

        if self.tool_name:
            self.tool = _noop_tool_factory(self.tool_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def __reduce__(self) -> tuple[Callable[[TorchModelState], "SimpleTorchModel"], tuple[TorchModelState]]:
        ################
        # YOUR CODE HERE
        ################

        # torch.save/torch.load use pickle internally for Python objects.
        state: TorchModelState = {
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "tool_name": self.tool_name,
        }
        return (_reconstruct_torch_model, (state,))


def save_model(path: Path) -> None:
    model = SimpleTorchModel(tool_name="example_tool")
    torch.save(model, path)


def load_model(path: Path) -> SimpleTorchModel:
    try:
        model = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(path, map_location="cpu")

    if not isinstance(model, SimpleTorchModel):
        raise TypeError("Loaded object is not a SimpleTorchModel")

    model.eval()
    return model


def _reconstruct_torch_model(state: TorchModelState) -> SimpleTorchModel:
    model = SimpleTorchModel(tool_name=state["tool_name"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _noop_tool_factory(tool_name: str) -> ToolFn:
    def _tool(payload: ToolPayload) -> ToolPayload:
        return {
            "tool": tool_name,
            "payload": payload,
            "status": "ok",
        }

    return _tool


if __name__ == "__main__":
    out_file = Path("model_torch.pt")
    save_model(out_file)

    restored_model = load_model(out_file)
    sample = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    with torch.no_grad():
        pred = restored_model(sample)

    tool_result = None
    if restored_model.tool:
        tool_result = restored_model.tool({"action": "ping"})

    print(f"Torch model saved to: {out_file}")
    print(f"Torch model tool_name: {restored_model.tool_name}")
    print(f"Torch tool call result: {tool_result}")
    print(f"Torch prediction shape: {tuple(pred.shape)}")
