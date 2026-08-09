from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, TypedDict

import joblib


class ModelState(TypedDict):
    name: str
    threshold: float
    tool_name: Optional[str]


ToolPayload = dict[str, object]
ToolFn = Callable[[ToolPayload], ToolPayload]


class JoblibToolModel:
    def __init__(self, name: str, threshold: float, tool_name: Optional[str] = None) -> None:
        self.name = name
        self.threshold = threshold
        self.tool_name = tool_name
        self.tool: Optional[ToolFn] = None

    def predict(self, x: float) -> int:
        return int(x >= self.threshold)

    def __reduce__(self) -> tuple[Callable[[ModelState], "JoblibToolModel"], tuple[ModelState]]:

        ################
        # YOUR CODE HERE
        ################

        state: ModelState = {
            "name": self.name,
            "threshold": self.threshold,
            "tool_name": self.tool_name,
        }
        return (_reconstruct_model, (state,))


def _reconstruct_model(state: ModelState) -> JoblibToolModel:
    model = JoblibToolModel(
        name=state["name"],
        threshold=state["threshold"],
        tool_name=state["tool_name"],
    )

    if model.tool_name:
        model.tool = _noop_tool_factory(model.tool_name)

    return model


def _noop_tool_factory(tool_name: str) -> ToolFn:
    def _tool(payload: ToolPayload) -> ToolPayload:
        return {
            "tool": tool_name,
            "payload": payload,
            "status": "ok",
        }

    return _tool


def save_model(path: Path) -> None:
    model = JoblibToolModel(name="baseline_joblib", threshold=0.5, tool_name="example_tool")
    joblib.dump(model, path)


def load_model(path: Path) -> JoblibToolModel:
    return joblib.load(path)


if __name__ == "__main__":
    out_file = Path("model.joblib")
    save_model(out_file)

    restored = load_model(out_file)
    print(f"Joblib model saved to: {out_file}")
    print(f"predict(0.7) -> {restored.predict(0.7)}")
