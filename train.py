#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import build_datasets, find_dataset_root
from src.evaluate import evaluate_model, plot_training_history
from src.model import build_model, compile_model, unfreeze_base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("plant_disease.train")


def main() -> None:
    config = load_config()
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.assets_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = find_dataset_root(config.raw_dir)
    logger.info("Using PlantVillage dataset at %s", dataset_root)

    train_ds, val_ds, class_names = build_datasets(
        config.raw_dir,
        image_size=config.image_size,
        validation_split=config.validation_split,
        batch_size=config.batch_size,
        seed=config.random_state,
        max_samples_per_class=config.max_samples_per_class,
    )
    logger.info("Classes: %d", len(class_names))

    model = build_model(len(class_names), config.image_size)
    compile_model(model, config.learning_rate)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]

    logger.info("Phase 1 — training classification head (%d epochs)", config.epochs)
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Phase 2 — fine-tuning MobileNetV2 (%d epochs)", config.fine_tune_epochs)
    unfreeze_base(model, config.fine_tune_from_layer)
    compile_model(model, config.fine_tune_learning_rate)
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.fine_tune_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = evaluate_model(model, val_ds, class_names, config.assets_dir)
    logger.info(
        "Validation accuracy: %.3f | macro F1: %.3f",
        metrics["accuracy"],
        metrics["macro_f1"],
    )

    model_path = config.model_dir / "plant_disease_model.keras"
    labels_path = config.model_dir / "class_labels.json"
    metrics_path = config.assets_dir / "training_metrics.json"

    model.save(model_path)
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    combined_history = {
        key: history_head.history.get(key, []) + history_fine.history.get(key, [])
        for key in {"accuracy", "val_accuracy", "loss", "val_loss"}
    }
    plot_training_history(
        type("History", (), {"history": combined_history})(),
        config.assets_dir / "training_history.png",
    )

    logger.info("Saved model to %s", model_path)
    logger.info("Saved labels to %s", labels_path)
    logger.info("Saved plots to %s", config.assets_dir)


if __name__ == "__main__":
    tf.random.set_seed(load_config().random_state)
    main()
