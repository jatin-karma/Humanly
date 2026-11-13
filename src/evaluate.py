import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.model import CLASS_NAMES
from src.train import MODEL_DIR, load_split


def main():
    model_path = MODEL_DIR / "human_face_v1.keras"
    metadata_path = MODEL_DIR / "human_face_v1.json"

    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    # 1. Load metadata to get the decision threshold (0.5)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])

    print("Loading test dataset...")
    # shuffle=False is critical so order of images matches predictions exactly
    test_ds = load_split("test", shuffle=False)

    print("Loading saved model...")
    model = tf.keras.models.load_model(model_path)

    print("Running evaluation on test split...")
    # Extract the true ground-truth labels (0 or 1) from the test dataset
    y_true = np.concatenate([labels.numpy().reshape(-1) for _, labels in test_ds]).astype(int)

    # Predict probabilities for all test images
    probabilities = model.predict(test_ds, verbose=1).reshape(-1)

    # Convert probability scores into 0 (human) or 1 (non_human) using threshold
    y_pred = (probabilities >= threshold).astype(int)

    # 2. Print evaluation metrics
    print("\n" + "=" * 50)
    print("Class order:", CLASS_NAMES, "(0 = human, 1 = non_human)")
    print("Decision threshold:", threshold)
    print("=" * 50)

    print("\nConfusion Matrix (Rows = True class, Columns = Predicted class):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4,
        )
    )


if __name__ == "__main__":
    main()