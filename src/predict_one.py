import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.model import CLASS_NAMES, IMAGE_SIZE

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def main(image_path: str):
    model_path = MODEL_DIR / "human_face_v1.keras"
    metadata_path = MODEL_DIR / "human_face_v1.json"

    if not model_path.exists():
        raise SystemExit(f"Model not found at {model_path}")

    model = tf.keras.models.load_model(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])

    # Preprocess image
    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    pixels = np.asarray(image, dtype=np.float32)[None, ...]  # Shape: (1, 224, 224, 3)

    p_non_human = float(model.predict(pixels, verbose=0)[0][0])
    predicted_index = int(p_non_human >= threshold)

    result = {
        "prediction": CLASS_NAMES[predicted_index],
        "human_probability": round(1.0 - p_non_human, 4),
        "non_human_probability": round(p_non_human, 4),
        "model_version": metadata["model_version"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.predict_one path/to/image.jpg")
    main(sys.argv[1])