from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(num_classes: int, image_size: tuple[int, int]) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
        name="mobilenetv2_backbone",
    )
    base.trainable = False

    inputs = base.input
    x = layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="plant_disease_mobilenet")


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_base(model: tf.keras.Model, from_layer: int) -> None:
    base = model.get_layer("mobilenetv2_backbone")
    base.trainable = True
    for layer in base.layers[:from_layer]:
        layer.trainable = False
