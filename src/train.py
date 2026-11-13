import json
from pathlib import Path
import tensorflow as tf

from src.model import CLASS_NAMES, IMAGE_SIZE, build_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "human_face_v1.keras"
METADATA_PATH = MODEL_DIR / "human_face_v1.json"
BATCH_SIZE = 32
SEED = 42


def load_split(split_name: str, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR / split_name,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    ).prefetch(tf.data.AUTOTUNE)


def main():
    if not (DATA_DIR / "train").exists() or not (DATA_DIR / "val").exists():
        raise SystemExit("Dataset missing in data/train or data/val")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val", shuffle=False)

    print("Building model...")
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(MODEL_DIR / "human_face_v1_training.csv"),
    ]

    print("Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
    )

    metadata = {
        "model_version": "1.0.0",
        "architecture": "MobileNetV2 transfer learning",
        "dataset": "LakoreAI/human-nonhuman-face-classification",
        "class_names": CLASS_NAMES,
        "positive_class_index": 1,
        "image_size": list(IMAGE_SIZE),
        "threshold": 0.5,
        "preprocessing": "RGB, resize 224x224, scale 0-255 to -1..1 inside saved model",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved best model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()