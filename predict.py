#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config


def load_model_and_labels(config):
    model_path = config.model_dir / "plant_disease_model.keras"
    labels_path = config.model_dir / "class_labels.json"
    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python3 train.py` after downloading PlantVillage."
        )
    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    return model, class_names


def preprocess_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image.numpy(), axis=0)


def predict_image(model, class_names, image_path: Path, image_size: tuple[int, int]) -> tuple[str, float]:
    batch = preprocess_image(image_path, image_size)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict plant disease from a leaf image.")
    parser.add_argument("image", type=Path, help="Path to a leaf image file")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predictions")
    args = parser.parse_args()

    config = load_config()
    model, class_names = load_model_and_labels(config)

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    batch = preprocess_image(args.image, config.image_size)
    probabilities = model.predict(batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][: args.top_k]

    print(f"Image: {args.image}")
    for rank, index in enumerate(top_indices, start=1):
        label = class_names[index]
        crop, disease = _split_label(label)
        print(
            f"{rank}. {label} | crop={crop} | condition={disease} | "
            f"confidence={probabilities[index]:.2%}"
        )


def _split_label(label: str) -> tuple[str, str]:
    if "___" in label:
        crop, disease = label.split("___", maxsplit=1)
        return crop.replace("_", " "), disease.replace("_", " ")
    return label, "unknown"


if __name__ == "__main__":
    main()
