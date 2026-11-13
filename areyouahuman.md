# Build guide: “Are you human?” image classifier

This is a learning project plan, not a security product. Using the selected starter dataset, you will build a service that classifies a face image as a real human face (`human`) or a non-human face-like image (`non_human`). It is useful for learning computer vision, APIs, containers, deployment, monitoring, and model lifecycle management.

> **Important limitation:** a two-class image model does not prove that a user is human, detect a live person, or resist spoofing. Do not use it as a CAPTCHA, identity check, access control, or safety decision. It can learn shortcuts such as backgrounds or camera style. Treat the prediction as a best-effort image classification demo.

## What you will have at the end

An image dataset and repeatable training process; a CNN checkpoint with evaluation results; an HTTP API that loads and serves a chosen model; a Docker image that runs the API consistently; a deployment that can be monitored; and a documented way to release and roll back model versions. The code walkthrough at the end gives you starter contents for the files. You create the files and type/run each stage yourself.

## The full flow

```text
Hugging Face dataset → export labeled images → preprocess → train CNN → evaluate → version model
                                                        ↓ approve
Client → deployed FastAPI → chosen model version → prediction + metrics
                            ↑ rollback to prior version if release is bad
```

## Suggested beginner stack

- **Python + TensorFlow/Keras:** image loading, preprocessing, CNN training, and inference.
- **FastAPI:** HTTP endpoints for image upload and prediction. It also generates interactive API docs at `/docs`.
- **Docker:** package the API and its Python dependencies into a repeatable runnable image.
- **Git:** version code and configuration.
- **DVC (later):** track dataset versions without putting large image files directly in Git.
- **MLflow (later):** track training runs, parameters, metrics, artifacts, and registered model versions.
- **Prometheus + Grafana (later):** collect and display service metrics and dashboards.

Learn in this order. Do not start by trying to install every tool at once.

## Step 0 — Understand the task and labels

Choose what “human” means for your dataset. The starter dataset linked below is a **face** dataset: it distinguishes real human faces from non-human face-like images. It does not answer whether an entire photo contains a person. Keep this narrower definition for the first version and write down edge cases (statues, masks, anime, blurry faces) so labels are consistent.

Why this matters: a model learns from labels, not from your intention. Unclear labels make evaluation and predictions unreliable. Keep the scope to ordinary photos and avoid collecting private or sensitive images without permission.

## Step 1 — Create the dataset

For the starter, download the public Hugging Face dataset in the code walkthrough below. You do not need to make a Hugging Face token for a public dataset. The preparation script exports its existing train, validation, and test splits into this folder layout:

```text
data/
  train/
    human/
    non_human/
  val/
    human/
    non_human/
  test/
    human/
    non_human/
```

- **Train** teaches the model.
- **Validation** helps compare choices while you develop.
- **Test** is held back until the end for an honest final estimate.

The dataset already provides splits; preserve them and do not reshuffle images between them. Check labels, image readability, class counts, and the dataset license. Record the source, label definition, and the date/version you downloaded. The folder names are there because Keras can infer labels from subfolders; your code can use another format if you write a matching loader.

**Why data versioning?** If the model changes, you need to know whether it was caused by code, settings, or a changed dataset. Git can hold a small manifest; DVC can track larger data using pointer files and remote storage. [DVC user guide](https://doc.dvc.org/user-guide)

## Step 2 — Build preprocessing

Images come in different file formats, sizes, and color modes. Your pipeline should:

1. Open the image safely and convert it to RGB.
2. Resize to the input size expected by the selected network (224×224 for this guide's MobileNetV2 model).
3. Convert pixels to numeric tensors and normalize them to the range expected by the pretrained CNN.
4. During **training only**, apply realistic augmentation such as modest crops or horizontal flips, if those transformations do not change the label.
5. Use the same deterministic resize and normalization at validation, test, and API inference.

Why: the model needs consistent numeric input. Training augmentation creates harmless variation so it is less likely to memorize exact training images. If training and API preprocessing differ, production predictions can be poor even though offline evaluation looked good. Save preprocessing settings alongside the model version.

## Step 3 — Choose and train a CNN

A CNN (convolutional neural network) is designed to learn visual patterns. Convolution filters slide across an image and learn local features such as edges and textures; deeper layers combine those into shapes. For this small dataset, this guide uses **MobileNetV2**, a compact pretrained CNN. Fine-tuning a network that has already learned general image features is more practical than teaching every feature from scratch. TensorFlow's image input guide explains Keras image loading and preprocessing: [Load and preprocess images](https://www.tensorflow.org/tutorials/load_data/images).

The starter code first freezes MobileNetV2's pretrained feature layers and trains a small new classification head. This keeps the first training run manageable. Once that works, you can optionally unfreeze the last layers and fine-tune them at a lower learning rate. TensorFlow provides the pretrained model and its architecture through Keras Applications.

Suggested learning sequence:

1. First train only the replacement classification layer (freeze the pretrained feature layers).
2. Evaluate it.
3. If you have enough data, unfreeze some/all layers and fine-tune with a smaller learning rate.
4. Save the best checkpoint based on **validation** performance, not test performance.

Track random seed, architecture, pretrained weights, image size, augmentation, batch size, optimizer, learning rate, epochs, class mapping, code commit, dataset version, and validation metrics. MLflow runs can store parameters, metrics, code versions, and output artifacts: [MLflow Tracking](https://mlflow.org/docs/latest/tracking). The starter code writes a model and a small metadata file first; add MLflow after you understand one training run.

## Step 4 — Evaluate before serving

Do not rely on accuracy alone. If classes are imbalanced, a model can achieve high accuracy by mostly guessing the larger class. Report:

- **Precision:** among images predicted human, how many really are human?
- **Recall:** among real human images, how many did it find?
- **F1:** a combined precision/recall score.
- **Confusion matrix:** counts of correct and incorrect predictions for both classes.
- **Per-class metrics:** reveals if one class performs much worse.

Choose a decision threshold using validation data and your stated goal. The default 0.5 is not automatically best. Inspect false positives and false negatives. Run the final evaluation only once on the held-out test set after decisions are done, and record the result. Keep a few “hard cases” to check whenever you change preprocessing or the model.

## Step 5 — Make a small inference program first

Before making a web service, use the saved Keras model to predict one image locally. The saved model includes resizing-compatible input and the pixel normalization layer, so the API must provide RGB pixels resized to 224×224 just like training. Confirm outputs for both classes and malformed files.

Keep model construction and preprocessing reusable: training and the API must call the same shared logic. Store metadata with the checkpoint: class names/order, architecture, input size, normalization, threshold, model version, and training run ID. Never infer class meaning from an unexplained output index.

## Step 6 — Learn FastAPI and make the API

An **API** is a defined way for another program to ask your service to do something. FastAPI maps HTTP requests to Python functions called route handlers. Start with a simple health route, then add prediction.

Recommended routes:

| Route | Purpose |
|---|---|
| `GET /health` | Liveness: process is running. |
| `GET /ready` | Readiness: model loaded and service can accept predictions. |
| `GET /model-info` | Return deployed model/version and preprocessing metadata. |
| `POST /predict` | Accept one image and return prediction, scores, model version, and request ID. |
| `GET /metrics` | Expose operational metrics for Prometheus (when you add monitoring). |

Learn these terms: **request** is what the client sends; **response** is what your API returns; **route/endpoint** is the URL plus HTTP method; **status code** indicates success or failure; **schema** describes expected input/output. Use an image upload field for `/predict`, validate file type and maximum size, handle decode errors clearly, and avoid logging or retaining uploaded image contents by default. FastAPI provides interactive docs at `/docs` and an OpenAPI description: [FastAPI first steps](https://fastapi.tiangolo.com/tutorial/first-steps/).

Load the model once when the service starts, not once per request. Return the model version in each prediction response so you can identify which release produced it. Keep a single worker at first; multiple server workers may each load a full copy of the model and consume more memory. A complete starter API file is included below.

## Step 7 — Learn Docker

Docker packages an application into an **image**; a running copy of that image is a **container**. A `Dockerfile` is the recipe: choose a Python base image, set a working directory, install pinned dependencies, copy in app code and required model assets, then set the startup command. `.dockerignore` excludes items such as local datasets, Git history, caches, and secrets from the build context.

Build and run locally, publish a port from container to your computer, and call `/health` and `/predict`. The point is repeatability: the same packaged dependencies and startup process can run on another machine. Learn Docker’s [first Dockerfile guide](https://docs.docker.com/learn/content/write-your-first-dockerfile/) and FastAPI’s [container deployment notes](https://fastapi.tiangolo.com/deployment/docker/).

Beginner pitfalls: a path on your computer is not automatically available inside a container; large datasets should not be copied into the API image; the service must listen on the container network interface; and secrets should be passed through deployment configuration, never baked into the image. Pin dependency versions and document how the model file gets into the runtime. Later, learn non-root users, smaller images, vulnerability scanning, and multi-stage builds.

## Step 8 — Add monitoring

Separate **service health** from **model quality**. A service can return HTTP 200 while its predictions are wrong.

**Operational signals:** request count/rate, error count/rate, response latency (especially p95), CPU/memory, restarts, and readiness. **Model signals:** prediction counts by class, confidence distribution, proportion near the decision threshold, and model version. Do not use per-image IDs, raw filenames, or other unbounded values as metric labels; that can create too many time series. Avoid storing image uploads unless you have a clear consent, privacy, access, and retention plan.

Prometheus periodically scrapes a metrics endpoint; Grafana displays the collected time series. Add structured logs with request ID, status, latency, and model version, but not image contents. For actual model quality, you need trustworthy labeled outcomes: periodically evaluate a reviewed sample or user-corrected labels (with consent) and compute precision/recall/F1. Confidence and input drift are warning signals, not proof that accuracy has changed. Prometheus documents its metrics exposition format [here](https://prometheus.io/docs/instrumenting/exposition_formats/).

## Step 9 — Version code, data, model, and deployment

These are separate versions and should be linked together:

| Item | How to identify it | Why |
|---|---|---|
| Source code | Git commit/tag | Recreate the API and training logic. |
| Dataset | DVC version/manifest/hash | Know which labeled data trained it. |
| Training run | MLflow run ID | Find parameters, metrics, logs, and artifacts. |
| Model | Immutable registry version (e.g. `human-check:7`) | Select and restore exact weights and preprocessing metadata. |
| Container | Immutable image tag or digest | Recreate the deployed runtime. |
| Release | Deployment record linking all of these | Trace a prediction back to its full build. |

Never overwrite a released model file in place as your only record. Keep released artifacts immutable. A moving alias such as `production` can point to one immutable registered version; MLflow supports model versions and aliases in its [Model Registry](https://mlflow.org/docs/latest/ml/model-registry/).

## Step 10 — Promotion, deployment, and rollback

Use a simple release flow:

1. **Develop:** change code or data on a branch; train and record a candidate run.
2. **Validate:** check data, preprocessing, metrics, class behavior, and API compatibility. Compare candidate and current production model on the same test/evaluation set.
3. **Register:** store the artifact and metadata as a new immutable model version. Mark it candidate/staging.
4. **Build:** build a container tied to a code commit and model version; record its digest.
5. **Deploy to staging:** run smoke checks: readiness, one valid image, one invalid image, and expected response fields.
6. **Promote gradually:** if your hosting platform supports it, send a small share of traffic to the new release (canary) or deploy to staging first. Compare error rate, latency, resource use, and model outputs.
7. **Promote to production:** move the production alias/configuration to the approved model version and deploy the corresponding release.
8. **Observe:** watch dashboards and logs after release. Keep the prior known-good model artifact and container available.

**Rollback means restoring a known-good release.** If errors rise, readiness fails, latency becomes unacceptable, or reviewed quality falls below your agreed floor: switch the production model reference back to the previous immutable version, redeploy/restart so the service loads it, verify `/ready` and a prediction, then record what happened. Do not delete the bad candidate; mark it rejected and preserve its run and logs to investigate. Keep the API and model compatible: a rollback needs the old weights, preprocessing metadata, and a runtime that can load them.

For a first deployment you can do this manually and document the exact steps. Later automate with CI/CD: on a code change, install pinned dependencies, run checks, build the container, publish it, deploy to staging, run smoke checks, and require an explicit promotion step. Training should not automatically replace production just because it completed.

## Step 11 — Deploy to a hosting service

The hosting provider runs your container and gives it a public URL. Pick a beginner-friendly container hosting service that supports environment variables, health checks, logs, and persistent or external model/artifact storage. Your release needs: a container image, a command/port configuration, model artifact access, CPU/memory limits, a health/readiness check, and secrets/settings configured outside source code.

First deploy a small CPU model and confirm it can cold start and load the weights within the provider’s limits. Check cost, sleep/scale-to-zero behavior, request size limits, and whether uploaded images are retained. Put a basic authentication or rate limit in front of a public demo if abuse is a concern. Do not promise availability or security guarantees for a portfolio demo.

## Step 12 — Make it a strong portfolio project

Your README should include: problem statement and limitation; dataset source/license and label rules; architecture diagram; setup and training instructions; preprocessing choices; experiment comparison; held-out test results and confusion matrix; API request/response example; Docker build/run steps; deployment link; monitoring screenshot; model release history; and a rollback example. Include what failed and what you learned. Never claim a metric you have not measured.

Possible resume bullet **after you have actually built and measured it**:

> Built and deployed a human/non-human face image classification API using TensorFlow/Keras MobileNetV2, FastAPI, and Docker; tracked data and model releases, monitored latency/errors and evaluation metrics, and documented rollback to a prior model version.

## A sensible milestone checklist

- [ ] I can explain my labels and dataset split.
- [ ] I can run preprocessing on a sample and inspect the result.
- [ ] I can train a baseline and explain precision, recall, and the confusion matrix.
- [ ] I can load a saved model and predict locally.
- [ ] I can upload an image through FastAPI `/docs` and explain the response.
- [ ] I can build/run the Docker image and explain image vs container.
- [ ] I can deploy it and find service logs/health status.
- [ ] I can identify the exact model and code version currently deployed.
- [ ] I have practiced rolling back and verified the earlier model serves predictions.

## Official references

- [TensorFlow load and preprocess images](https://www.tensorflow.org/tutorials/load_data/images)
- [Keras image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory)
- [FastAPI first steps](https://fastapi.tiangolo.com/tutorial/first-steps/)
- [FastAPI with Docker](https://fastapi.tiangolo.com/deployment/docker/)
- [Docker first Dockerfile](https://docs.docker.com/learn/content/write-your-first-dockerfile/)
- [DVC user guide](https://doc.dvc.org/user-guide)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/)

---

## File-by-file TensorFlow starter

This section is the hands-on part. Create these files yourself in the order below and paste the matching contents. Run each milestone before moving on; if something fails, fix that step before adding the next tool.

### 1. Create this project structure

```text
areyouahuman/
├── data/                         # downloaded/exported images; do not commit to Git
├── models/                       # trained model files; do not commit large files to Git
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare_data.py        # downloads from Hugging Face and exports image folders
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                # FastAPI app and inference endpoint
│   ├── model.py                   # defines CNN architecture and preprocessing layers
│   ├── train.py                   # reads train/val, trains, saves best model
│   ├── evaluate.py                # evaluates only on held-out test split
│   └── predict_one.py             # predicts one local image before building API
├── requirements.txt
├── .gitignore
├── .dockerignore
├── Dockerfile
└── README.md
```

**What goes where:** downloading and converting data belongs in `prepare_data.py`; image resizing and augmentation/model definition belongs in `model.py`; training belongs in `train.py`; final metrics belong in `evaluate.py`; the API only loads a trained model and serves predictions. Keeping these jobs separate makes it easier to understand and debug the pipeline.

### 2. Set up your Python environment

Open a terminal in the `areyouahuman` folder. TensorFlow supports specific Python versions; use Python 3.12 for this project rather than an unsupported newer version. A virtual environment isolates this project's packages from other Python projects.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or adjust your PowerShell policy only if you understand your machine's policy. Do not put a Hugging Face token in the code. The dataset below is public, so try it without a token; only authenticate if Hugging Face explicitly requires it.

### 3. Create `requirements.txt`

These are the project libraries. TensorFlow builds the CNN; `datasets` downloads the Hub dataset; Pillow handles images; FastAPI and Uvicorn serve requests; scikit-learn prints useful evaluation metrics.

```text
tensorflow>=2.16,<3
datasets>=3,<5
Pillow>=10,<13
numpy>=1.26,<3
scikit-learn>=1.4,<2
fastapi>=0.115,<1
uvicorn[standard]>=0.30,<1
python-multipart>=0.0.9,<1
```

Install them inside the active environment:

```powershell
python -m pip install -r requirements.txt
```

After you have a working install, record the exact resolved versions (`python -m pip freeze > requirements-lock.txt`) so you can recreate the same environment later. Use the lock file for repeatable deployments.

### 4. Download and export the dataset: `src/data/prepare_data.py`

The selected [Hugging Face dataset](https://huggingface.co/datasets/LakoreAI/human-nonhuman-face-classification) is a **face-realism** task, not whole-image person detection. The dataset card currently displays `LakoreAI/human-nonhuman-face-classification`; its sample code mentions an older/different ID. The script below uses the dataset's current page ID. If loading fails, check the ID shown in the page URL and dataset card before changing it. The card reports a public MIT license, around 8,500 examples, and existing splits. Read the license/card yourself before publishing a model or images.

Hugging Face's `datasets` library downloads files into its cache. This script then exports the images into the simple directory structure Keras understands. No token is required for a public dataset.

```python
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "LakoreAI/human-nonhuman-face-classification"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def choose_split(dataset, choices):
    """Return the first split name that exists, e.g. val or validation."""
    for name in choices:
        if name in dataset:
            return name
    raise ValueError(f"None of these splits exist: {choices}. Found: {list(dataset.keys())}")


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise SystemExit(
            f"{OUTPUT_DIR} already contains files. Move it aside or remove that generated "
            "folder yourself before exporting again."
        )

    print(f"Downloading/loading {DATASET_ID} (first run may take a few minutes)...")
    dataset = load_dataset(DATASET_ID)
    print("Available splits:", list(dataset.keys()))

    # Dataset repositories sometimes call validation 'val' and sometimes 'validation'.
    split_map = {
        "train": choose_split(dataset, ["train"]),
        "val": choose_split(dataset, ["val", "validation"]),
        "test": choose_split(dataset, ["test"]),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for output_split, source_split in split_map.items():
        rows = dataset[source_split]
        label_feature = rows.features["label"]
        if not hasattr(label_feature, "int2str"):
            raise TypeError("Expected the Hugging Face label column to be a ClassLabel feature.")

        print(f"Exporting {source_split} → {output_split} ({len(rows)} rows)")
        for index, row in enumerate(rows):
            label_name = label_feature.int2str(int(row["label"])).strip().lower()
            if label_name not in {"human", "non_human"}:
                raise ValueError(f"Unexpected label {label_name!r}; inspect the dataset card/labels.")

            image = row["image"].convert("RGB")
            class_dir = OUTPUT_DIR / output_split / label_name
            class_dir.mkdir(parents=True, exist_ok=True)
            image.save(class_dir / f"{index:06d}.jpg", format="JPEG", quality=95)

    print(f"Done. Keras-ready images are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

Run it from the project root:

```powershell
python -m src.data.prepare_data
```

The resulting folders should include `data/processed/train/human`, `data/processed/train/non_human`, and equivalent `val` and `test` folders. Check that the printed counts make sense before training. The validation and test splits are already supplied; do not randomly reshuffle them.

### 5. Define the CNN and shared image preprocessing: `src/model.py`

This defines a pretrained MobileNetV2 CNN plus a new one-unit sigmoid classifier. The sigmoid output is a score from 0 to 1 for class index 1 (`non_human`). `Rescaling` maps pixels from 0–255 to approximately -1–1, which is the input scale expected by MobileNetV2. The augmentation layers run only during training; during API prediction they automatically behave as identity operations.

```python
import tensorflow as tf
from tensorflow.keras import layers


IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["human", "non_human"]  # index 0 and index 1; keep this order everywhere


def build_model():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_image")

    # Random variation helps reduce memorization; it is inactive for validation/inference.
    x = layers.RandomFlip("horizontal", name="augment_flip")(inputs)
    x = layers.RandomRotation(0.04, name="augment_rotation")(x)

    # MobileNetV2 pretrained weights expect pixels scaled to [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenetv2_scale")(x)

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base.trainable = False  # first learn a new classifier head, keeping pretrained features fixed

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="non_human_probability")(x)

    return tf.keras.Model(inputs, outputs, name="human_face_classifier")
```

**Why MobileNetV2?** It is a CNN with fewer compute requirements than many larger image networks, and pretrained weights give a helpful starting point when your labeled dataset is modest. The first run freezes its feature extractor to keep training simpler. Later you can compare it with a small CNN trained from scratch to learn the difference.

### 6. Train and save the best model: `src/train.py`

`image_dataset_from_directory` reads class names from folder names. We explicitly set the class order so the label mapping is stable. Resizing happens as images are loaded; normalization and augmentation are part of the saved model. `ModelCheckpoint` keeps the epoch with the best validation loss. Validation data helps choose the model; test data is not used here.

```python
import json
from pathlib import Path

import tensorflow as tf

from src.model import CLASS_NAMES, IMAGE_SIZE, build_model


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "human_face_v1.keras"
METADATA_PATH = MODEL_DIR / "human_face_v1.json"
BATCH_SIZE = 32
SEED = 42


def load_split(split_name, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split_name,
        labels="inferred",
        label_mode="binary",       # labels are float 0 or 1 for binary cross-entropy
        class_names=CLASS_NAMES,    # human=0; non_human=1
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    ).prefetch(tf.data.AUTOTUNE)


def main():
    if not (DATA_DIR / "train").exists() or not (DATA_DIR / "val").exists():
        raise SystemExit("Dataset missing. Run: python -m src.data.prepare_data")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val", shuffle=False)

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
    print(f"Saved best model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
```

Run:

```powershell
python -m src.train
```

The first run may download MobileNetV2's ImageNet weights and take time. It saves the best model as `models/human_face_v1.keras`. Do not overwrite a released model to make a new version; use a new filename such as `human_face_v2.keras` and record what changed.

### 7. Evaluate once on test data: `src/evaluate.py`

This computes the confusion matrix and precision/recall/F1 on the held-out test split. Do not use this output to repeatedly tune the model; use validation results for that, then report the test result honestly.

```python
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.model import CLASS_NAMES, IMAGE_SIZE
from src.train import MODEL_DIR, load_split


def main():
    model_path = MODEL_DIR / "human_face_v1.keras"
    metadata = json.loads((MODEL_DIR / "human_face_v1.json").read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])

    test_ds = load_split("test", shuffle=False)
    model = tf.keras.models.load_model(model_path)

    y_true = np.concatenate([labels.numpy().reshape(-1) for _, labels in test_ds]).astype(int)
    probabilities = model.predict(test_ds, verbose=1).reshape(-1)
    y_pred = (probabilities >= threshold).astype(int)

    print("Class order:", CLASS_NAMES, "(0=human, 1=non_human)")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
```

Run `python -m src.evaluate`. If Python reports an unused import warning for `BATCH_SIZE` or `DATA_DIR`, that is harmless; those names are imported only through the shared split loader. You can also remove them from the import line.

### 8. Predict one image locally: `src/predict_one.py`

This lets you debug image decoding and class mapping before involving HTTP or Docker.

```python
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.model import CLASS_NAMES, IMAGE_SIZE


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def main(image_path):
    model = tf.keras.models.load_model(MODEL_DIR / "human_face_v1.keras")
    metadata = json.loads((MODEL_DIR / "human_face_v1.json").read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])

    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    pixels = np.asarray(image, dtype=np.float32)[None, ...]  # batch dimension: 1 x H x W x RGB
    p_non_human = float(model.predict(pixels, verbose=0)[0][0])
    predicted_index = int(p_non_human >= threshold)

    print({
        "prediction": CLASS_NAMES[predicted_index],
        "human_probability": round(1.0 - p_non_human, 4),
        "non_human_probability": round(p_non_human, 4),
        "model_version": metadata["model_version"],
    })


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.predict_one path\\to\\image.jpg")
    main(sys.argv[1])
```

Example: `python -m src.predict_one data\processed\test\human\000001.jpg`.

### 9. Create the API: `src/api/main.py`

FastAPI receives the uploaded image; Pillow decodes it; NumPy formats pixels as a batch; TensorFlow returns a score. The API does not download data or train a model. It loads the chosen model once during startup and serves it.

```python
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from src.model import CLASS_NAMES, IMAGE_SIZE


MODEL_PATH = os.getenv("MODEL_PATH", "models/human_face_v1.keras")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. Train the model first.")
    app.state.model = tf.keras.models.load_model(MODEL_PATH)
    yield
    app.state.model = None


app = FastAPI(title="Are You Human? Face Classifier", version=MODEL_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/model-info")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "classes": CLASS_NAMES,
        "image_size": list(IMAGE_SIZE),
        "threshold": THRESHOLD,
    }


def run_prediction(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    pixels = np.asarray(image, dtype=np.float32)[None, ...]
    p_non_human = float(model.predict(pixels, verbose=0)[0][0])
    predicted_index = int(p_non_human >= THRESHOLD)
    return {
        "prediction": CLASS_NAMES[predicted_index],
        "human_probability": round(1.0 - p_non_human, 4),
        "non_human_probability": round(p_non_human, 4),
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller")

    try:
        result = await run_in_threadpool(run_prediction, app.state.model, image_bytes)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image")
    finally:
        await file.close()

    return result
```

Run locally: `python -m uvicorn src.api.main:app --reload`. Open `http://127.0.0.1:8000/docs`, expand `POST /predict`, choose an image, and click **Execute**. `/health` only says the process is alive; `/ready` confirms the model loaded.

### 10. Ignore generated files: `.gitignore`

Git should contain your code and documentation, not downloaded images, credentials, virtual environments, or large model binaries.

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
data/
models/*.keras
models/*.h5
models/*training.csv
```

The model is ignored because it may be large. For deployment you still need to provide the trained model to Docker or fetch it from a model registry. In this first tutorial, you build the Docker image locally after training; `.dockerignore` below intentionally does **not** exclude the specific `.keras` file.

### 11. Control what Docker receives: `.dockerignore`

```dockerignore
.venv
__pycache__
*.py[cod]
.git
.env
data
models/*.h5
models/*training.csv
```

The entire `data` folder is excluded so thousands of training images do not bloat the API image. The trained `models/human_face_v1.keras` file remains in the build context for this beginner deployment.

### 12. Package the API: `Dockerfile`

Docker's `FROM` selects the starting environment; `WORKDIR` sets the working folder; `COPY` places project files in the image; `RUN` installs packages during image creation; `CMD` is the server startup command. `0.0.0.0` lets traffic reach the app inside the container.

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/human_face_v1.keras \
    MODEL_VERSION=1.0.0

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models/human_face_v1.keras ./models/human_face_v1.keras

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create empty `src/__init__.py`, `src/api/__init__.py`, and `src/data/__init__.py` files so Python treats these folders as packages. Before building, make sure `models/human_face_v1.keras` exists from training.

Build and run:

```powershell
docker build -t areyouahuman:1.0.0 .
docker run --rm -p 8000:8000 areyouahuman:1.0.0
```

Open `http://localhost:8000/docs`. `-p 8000:8000` connects port 8000 on your computer to port 8000 inside the container. Stop with Ctrl+C. To name the run explicitly with a model version: `docker run --rm -p 8000:8000 -e MODEL_VERSION=1.0.0 areyouahuman:1.0.0`.

### 13. Add a small `README.md`

Start with sections for project purpose and limitation, dataset source/license and class meaning, setup commands, download/train/evaluate commands, API usage, Docker commands, measured test results, deployed URL, monitoring, and rollback procedure. Fill in measured results only after you run the project; never invent accuracy.

### 14. Deployment pipeline: what to automate after local Docker works

First get the local path working from dataset download to a container prediction. Then add a GitHub Actions workflow or your host's deployment integration. A beginner CI/CD pipeline should do these steps in order:

1. On a push, install the pinned dependencies and run code checks (later, add tests).
2. Build the Docker image and tag it with the Git commit, for example `areyouahuman:<commit-sha>`; do not use `latest` as your only release identifier.
3. Publish that image to a container registry.
4. Deploy the image to a **staging** service and check `/health`, `/ready`, and one prediction.
5. Promote the same image to production after you review evaluation metrics and staging behavior.
6. Keep the previous production image tag and model artifact. If the new version fails, redeploy the previous image tag and verify `/ready` and `/predict`.

For every release, write down: Git commit, dataset version, model version, container tag/digest, evaluation metrics, and deployment date. Initially perform promotion and rollback manually so you learn what each action does; automate them only after the manual procedure works. The hosting provider differs by choice, so add provider-specific deployment files after selecting one.

### Run order recap

From the active virtual environment and project root:

```powershell
python -m src.data.prepare_data
python -m src.train
python -m src.evaluate
python -m src.predict_one data\processed\test\human\000001.jpg
python -m uvicorn src.api.main:app --reload
```

Once the local API works, build/run Docker using the commands in Step 12. If a path fails, check the current directory first: run these commands from the top-level `areyouahuman` folder, not from inside `src`.
