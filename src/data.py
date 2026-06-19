from __future__ import annotations

import random
from pathlib import Path

import tensorflow as tf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_dataset_root(raw_dir: Path) -> Path:
    """Locate the folder that contains PlantVillage class subdirectories."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {raw_dir}\n"
            "Download PlantVillage and extract it under data/raw/. "
            "See README.md for instructions."
        )

    candidates: list[tuple[int, Path]] = []
    for path in [raw_dir, *raw_dir.rglob("*")]:
        if not path.is_dir():
            continue
        class_dirs = [child for child in path.iterdir() if child.is_dir() and _looks_like_class_dir(child)]
        if len(class_dirs) >= 5:
            image_count = sum(len(list_images(class_dir)) for class_dir in class_dirs[:5])
            candidates.append((len(class_dirs), path if image_count > 0 else path))

    if not candidates:
        raise FileNotFoundError(
            f"No PlantVillage class folders found under {raw_dir}. "
            "Expected subfolders like 'Potato___Early_blight/' with images inside."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _looks_like_class_dir(path: Path) -> bool:
    return any(child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir() if child.is_file())


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        child
        for child in class_dir.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
    )


def _maybe_limit(paths: list[Path], max_samples: int | None, seed: int) -> list[Path]:
    if max_samples is None or len(paths) <= max_samples:
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, max_samples)


def build_datasets(
    raw_dir: Path,
    *,
    image_size: tuple[int, int],
    validation_split: float,
    batch_size: int,
    seed: int,
    max_samples_per_class: int | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    dataset_root = find_dataset_root(raw_dir)
    class_names = sorted(
        child.name for child in dataset_root.iterdir() if child.is_dir() and _looks_like_class_dir(child)
    )

    train_records: list[tuple[str, str]] = []
    val_records: list[tuple[str, str]] = []

    for class_name in class_names:
        images = _maybe_limit(list_images(dataset_root / class_name), max_samples_per_class, seed)
        if not images:
            continue
        rng = random.Random(f"{seed}-{class_name}")
        rng.shuffle(images)
        split_at = max(1, int(len(images) * (1 - validation_split)))
        if split_at >= len(images):
            split_at = max(1, len(images) - 1)
        for path in images[:split_at]:
            train_records.append((str(path), class_name))
        for path in images[split_at:]:
            val_records.append((str(path), class_name))

    label_to_index = {name: index for index, name in enumerate(class_names)}

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    def make_dataset(records: list[tuple[str, str]], training: bool) -> tf.data.Dataset:
        paths = [record[0] for record in records]
        labels = [label_to_index[record[1]] for record in records]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(min(len(records), 1000), seed=seed)
            ds = ds.map(
                lambda image, label: (tf.image.random_flip_left_right(image), label),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_records, training=True)
    val_ds = make_dataset(val_records, training=False)
    return train_ds, val_ds, class_names
