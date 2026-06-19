from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    random_state: int
    raw_dir: Path
    image_size: tuple[int, int]
    validation_split: float
    max_samples_per_class: int | None
    batch_size: int
    epochs: int
    fine_tune_epochs: int
    learning_rate: float
    fine_tune_learning_rate: float
    fine_tune_from_layer: int
    model_dir: Path
    assets_dir: Path


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return AppConfig(
        project_name=raw["project"]["name"],
        random_state=raw["project"]["random_state"],
        raw_dir=ROOT / raw["data"]["raw_dir"],
        image_size=tuple(raw["data"]["image_size"]),
        validation_split=raw["data"]["validation_split"],
        max_samples_per_class=raw["data"].get("max_samples_per_class"),
        batch_size=raw["training"]["batch_size"],
        epochs=raw["training"]["epochs"],
        fine_tune_epochs=raw["training"]["fine_tune_epochs"],
        learning_rate=raw["training"]["learning_rate"],
        fine_tune_learning_rate=raw["training"]["fine_tune_learning_rate"],
        fine_tune_from_layer=raw["training"]["fine_tune_from_layer"],
        model_dir=ROOT / raw["paths"]["model_dir"],
        assets_dir=ROOT / raw["paths"]["assets_dir"],
    )
