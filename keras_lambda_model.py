from __future__ import annotations

import os

from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="custom")
def tool_ready_transform(t: tf.Tensor) -> tf.Tensor:

    ################
    # YOUR CODE HERE
    ################

    scaled = t * 2.0
    shifted = scaled + 0.5
    return tf.math.tanh(shifted)


def build_model() -> keras.Model:
    inputs = keras.Input(shape=(4,), name="features")

    x = layers.Dense(8, activation="relu", name="dense_1")(inputs)

    # Named function keeps Lambda logic readable and easier to reuse.
    x = layers.Lambda(tool_ready_transform, name="tool_ready_transform")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="simple_lambda_model")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def save_model(path: Path) -> None:
    model = build_model()
    model.save(path)


def load_model(path: Path) -> Any:
    return keras.models.load_model(path)


if __name__ == "__main__":
    out_dir = Path("model.keras")
    save_model(out_dir)
    loaded_model = load_model(out_dir)
    print(f"Keras model saved to: {out_dir}")
    print(f"Keras model loaded: {loaded_model.name}")
