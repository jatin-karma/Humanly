---
title: Humanly - Face Realism Classifier
emoji: ✦
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Humanly: “Are You Human or Not?” Face Realism Image Classifier

An end-to-end beginner machine learning project that classifies whether a given face image belongs to a real human face (`human`) or a non-human face-like object (`non_human` — e.g. statues, digital art, anime, CGI, gaming avatars, or masks).

This repository covers the complete machine learning lifecycle: dataset acquisition, shared preprocessing layers, CNN transfer learning using MobileNetV2, rigorous test set evaluation, an asynchronous prediction API with FastAPI, and containerized deployment with Docker.

---

## 1. Project Overview & Scope

### What This Model Predicts
* **Class 0 (`human`):** Real, natural human face photographs.
* **Class 1 (`non_human`):** Synthetic, artificial, or artistic depictions of faces (statues, paintings, anime characters, gaming renders, and masks).

### Important Limitations
* **Not a Security or Identity System:** This model is **not** a CAPTCHA, a biometric identity verifier, a liveness detection system, or proof that an actual human is sitting at a computer.
* **Narrow Focus:** The model was trained specifically on cropped face images. It does not determine whether an entire uncropped photo contains a human body.
* **Potential Shortcuts:** Like all visual CNNs, the model can pick up superficial image shortcuts (such as textures, lighting artifacts, or artistic background rendering) rather than truly understanding human biology.

### Learning Goals
1. Learn how to structure a production-ready machine learning codebase with separate modules for data loading, model architecture, training, and inference.
2. Understand transfer learning with a pretrained Convolutional Neural Network (MobileNetV2).
3. Embed preprocessing and augmentation layers directly into the model graph to avoid train-serve skew.
4. Measure performance honestly using held-out test splits, precision, recall, and a confusion matrix.
5. Wrap a trained model into a production HTTP service using **FastAPI** and package it into a reproducible **Docker** container.

---

## 2. End-to-End System Workflow

```text
1. DATASET
   Hugging Face Hub (LakoreAI/human-nonhuman-face-classification)
   └── Downloaded / Exported to: data/train, data/val, data/test (8,534 images)
                                  │
2. TRAINING & MODEL ARTIFACT      ▼
   src/model.py  ──> Preprocessing (Rescaling [-1, 1], Augmentation) + MobileNetV2
   src/train.py  ──> Transfer Learning (Adam, BinaryCrossentropy, EarlyStopping)
                     └── Output: models/human_face_v1.keras (9.6 MB)
                     └── Output: models/human_face_v1.json (Metadata)
                                  │
3. EVALUATION                     ▼
   src/evaluate.py ──> Evaluated on data/test (855 unseen images)
                       └── Confusion Matrix & Classification Report (96.96% Accuracy)
                                  │
4. SERVING & PACKAGING            ▼
   src/api/main.py ──> FastAPI server (/health, /ready, /model-info, /predict)
   index.html      ──> Interactive browser frontend (upload & preview)
   Dockerfile      ──> Container image packaging code + model for deployment
```

---

## 3. Repository Structure

```text
AreU_Human/
├── data/                         # Local dataset directory (not committed to Git)
│   ├── train/                    # Training images (5,973 images: human, non_human)
│   ├── val/                      # Validation split (1,706 images)
│   └── test/                     # Final evaluation held-out test split (855 images)
├── models/                       # Model release artifacts
│   ├── human_face_v1.keras       # Trained Keras 3 model weights & architecture (9.6 MB)
│   └── human_face_v1.json        # Release metadata, threshold, and preprocessing spec
├── src/
│   ├── __init__.py               # Python package marker
│   ├── model.py                  # MobileNetV2 CNN architecture & preprocessing layers
│   ├── train.py                  # Training loop with callbacks and data loaders
│   ├── evaluate.py               # Evaluates final model on held-out test data
│   ├── predict_one.py            # CLI script to predict on a single local image file
│   └── api/
│       ├── __init__.py           # API package marker
│       └── main.py               # FastAPI application with CORS and inference routes
├── index.html                    # Modern standalone web frontend
├── Dockerfile                    # Container definition (based on tensorflow/tensorflow:2.16.1)
├── requirements.txt              # Production dependency specifications
├── .dockerignore                 # Excludes raw data, caches, and virtual environments
├── .gitignore                    # Prevents committing datasets, models, and virtualenvs
├── test_metrics.json             # Exact measured test evaluation metrics
├── README_METRICS.md             # Markdown snippet of test results
└── README.md                     # Project documentation
```

---

## 4. Dataset Details

* **Source:** [LakoreAI/human-nonhuman-face-classification](https://huggingface.co/datasets/LakoreAI/human-nonhuman-face-classification)
* **License:** MIT License (Public)
* **Total Images:** 8,534 images across 3 pre-defined splits.
* **Label Mapping:**
  * Index `0`: `human` (Real human faces)
  * Index `1`: `non_human` (Statues, anime, gaming characters, paintings, masks)

### Split Counts
| Split | Human (`0`) | Non-Human (`1`) | Total Images | Purpose |
|---|:---:|:---:|:---:|---|
| **Train** | 3,024 | 2,949 | **5,973** | Optimizes the classification head weights |
| **Validation** | 864 | 842 | **1,706** | Evaluates epoch-by-epoch loss to trigger `ModelCheckpoint` |
| **Test** | 433 | 422 | **855** | Held-out split evaluated only once for unbiased reporting |
| **Total** | **4,321** | **4,213** | **8,534** | Balanced (~50.6% human, 49.4% non-human) |

---

## 5. Environment Setup & Dependency Installation

### Local Setup (Windows PowerShell)

#### Prerequisites
* Python 3.11 installed.
* Docker Desktop installed and running.

#### 1. Clone the repository
```powershell
git clone https://github.com/YOUR_USERNAME/AreU_Human.git
cd AreU_Human
```

#### 2. Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### 3. Install dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 6. Step-by-Step Execution Guide

### Step A: Download & Organize Dataset
Images can be downloaded directly from the Hugging Face hub:
```powershell
git clone https://huggingface.co/datasets/LakoreAI/human-nonhuman-face-classification data
```
*(Ensure images are organized into `data/train`, `data/val`, and `data/test` with `human/` and `non_human/` subfolders).*

### Step B: Train the Model
```powershell
python -m src.train
```
* Fits the classification head for up to 10 epochs.
* Automatically saves the best checkpoint to `models/human_face_v1.keras`.
* Writes release metadata to `models/human_face_v1.json`.

### Step C: Evaluate on Test Data
```powershell
python -m src.evaluate
```
* Loads `models/human_face_v1.keras`.
* Evaluates exclusively against `data/test` (`shuffle=False`).
* Prints the Confusion Matrix, Accuracy, Precision, and Recall.

### Step D: Predict on a Single Local Image
```powershell
python -m src.predict_one data\test\human\000000.jpg
```
Output:
```json
{
  "prediction": "human",
  "human_probability": 0.9854,
  "non_human_probability": 0.0146,
  "model_version": "1.0.0"
}
```

### Step E: Run the FastAPI Server Locally
```powershell
python -m uvicorn src.api.main:app --reload --port 8000
```
Open interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 7. Machine Learning Architecture & Design Choices

### Preprocessing & Data Augmentation Pipeline (`src/model.py`)
To prevent **train-serve skew** (where API preprocessing differs from training preprocessing), image transformations are embedded directly into the Keras model:
1. **Input Shape:** Fixed tensor dimensions of `(224, 224, 3)`.
2. **Data Augmentation (Active only during training):**
   * `RandomFlip("horizontal")`: Flips faces horizontally to account for left/right symmetry.
   * `RandomZoom((-0.2, 0.2))`: Zooms in or out by up to 20%.
   * `RandomRotation(0.05)`: Tilts faces slightly by $\pm 5\%$.
3. **Rescaling / Normalization:**
   * MobileNetV2 expects pixel values normalized between `[-1.0, 1.0]`.
   * Implemented via `layers.Rescaling(1.0 / 127.5, offset=-1.0)`, mapping raw `[0, 255]` pixel integers using:
     $$\text{pixel}_{\text{scaled}} = \left(\frac{\text{pixel}}{127.5}\right) - 1.0$$

### Model Architecture
```python
# 1. Base Feature Extractor (Pretrained on ImageNet)
base = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base.trainable = False  # Pretrained feature layers are frozen

# 2. Classification Head
x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
x = layers.Dropout(0.2, name="dropout")(x)
outputs = layers.Dense(1, activation="sigmoid", name="non_human_probability")(x)
```
* **Why MobileNetV2?** It is lightweight (~14 MB), compute-efficient for CPU/Docker inference, and has strong general feature representations learned from 1.4M ImageNet photos.
* **Why Freeze Base Layers?** With only ~6,000 training images, training all 2.2M parameters from scratch risks severe overfitting. Freezing limits trainable parameters to just **1,281 weights** in the final Dense layer.

### Training Strategy & Callbacks (`src/train.py`)
* **Optimizer:** `Adam(learning_rate=1e-3)`
* **Loss Function:** `BinaryCrossentropy()`
* **Batch Size:** 32 images per batch
* **Callbacks:**
  * `ModelCheckpoint`: Saves only the single best epoch based on `val_loss`.
  * `EarlyStopping`: Halts training if `val_loss` does not improve for 3 consecutive epochs (`patience=3`, `restore_best_weights=True`).
  * `CSVLogger`: Logs per-epoch history to `models/human_face_v1_training.csv`.

---

## 8. Measured Evaluation Results

The model was evaluated on the **855 images** of the held-out `test` split (`data/test`).

* **Overall Test Accuracy:** **`96.96%`** (829 out of 855 correct)
* **Decision Threshold:** `0.5`

### Confusion Matrix
$$\begin{array}{c|cc}
\text{Actual \ Predicted} & \text{Predicted Human (0)} & \text{Predicted Non-Human (1)} \\
\hline
\text{\textbf{Actual Human (0)}} & \mathbf{415} & 18 \\
\text{\textbf{Actual Non-Human (1)}} & 8 & \mathbf{414}
\end{array}$$

* **True Positives (Human correctly identified):** 415
* **True Negatives (Non-Human correctly identified):** 414
* **False Negatives (Real humans misclassified as non-human):** 18 (4.16% error rate)
* **False Positives (Non-humans misclassified as human):** 8 (1.89% error rate)

### Detailed Per-Class Metrics
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Human (`0`)** | `0.9811` (98.1%) | `0.9584` (95.8%) | `0.9696` | 433 images |
| **Non-Human (`1`)** | `0.9583` (95.8%) | `0.9810` (98.1%) | `0.9696` | 422 images |
| **Accuracy** | | | **`0.9696`** | **855 images** |
| **Macro Average** | `0.9697` | `0.9697` | `0.9696` | 855 images |
| **Weighted Average** | `0.9699` | `0.9696` | `0.9696` | 855 images |

---

## 9. API Specification & Endpoints

The API is implemented in [src/api/main.py](src/api/main.py) with CORS middleware enabled.

### Endpoints

#### 1. Liveness Probe: `GET /health`
* **Purpose:** Confirms the HTTP server process is running.
* **Response (HTTP 200):**
  ```json
  {
    "status": "alive"
  }
  ```

#### 2. Readiness Probe: `GET /ready`
* **Purpose:** Confirms the neural network model is loaded in memory and ready for inference.
* **Response (HTTP 200):**
  ```json
  {
    "status": "ready",
    "model_version": "1.0.0"
  }
  ```

#### 3. Model Metadata: `GET /model-info`
* **Response (HTTP 200):**
  ```json
  {
    "model_version": "1.0.0",
    "classes": ["human", "non_human"],
    "image_size": [224, 224],
    "threshold": 0.5
  }
  ```

#### 4. Face Prediction: `POST /predict`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`
* **Payload:** `file` (Binary image file: JPEG, PNG, or WebP; maximum 5 MB)
* **Response (HTTP 200 OK):**
  ```json
  {
    "prediction": "human",
    "human_probability": 0.9854,
    "non_human_probability": 0.0146,
    "model_version": "1.0.0"
  }
  ```
* **Validation Errors:**
  * HTTP 415: `{"detail": "Upload a JPEG, PNG, or WebP image"}`
  * HTTP 413: `{"detail": "Image must be 5 MB or smaller"}`
  * HTTP 400: `{"detail": "The uploaded file is not a readable image"}`

---

## 10. Containerization with Docker

The application is containerized using **Docker** for cross-platform portability.

### `Dockerfile` Architecture
* **Base Image:** `tensorflow/tensorflow:2.16.1` (pre-packages Linux-optimized TensorFlow and NumPy).
* **Dependencies:** `fastapi`, `uvicorn`, `pillow`, `python-multipart`, and `keras>=3.0`.
* **Port Exposed:** `8000`
* **Start Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`

### Build & Run Commands

```powershell
# 1. Build the Docker container image
docker build -t areyouahuman:1.0.0 .

# 2. Run container mapped to port 8000
docker run -p 8000:8000 areyouahuman:1.0.0
```

Once running, access:
* **Interactive UI:** Open `index.html` in your browser.
* **Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 11. Lifecycle, Versioning & Deployment Status

| Capability | Status | Implementation Details |
|---|:---:|---|
| **Code Versioning** | **Implemented** | Git repository tracking code and configurations. Large data and binaries excluded via `.gitignore`. |
| **Model Artifact Versioning** | **Implemented** | Release metadata tracked via `models/human_face_v1.json` with semantic versioning (`1.0.0`). |
| **Container Versioning** | **Implemented** | Tagged Docker images (`areyouahuman:1.0.0`). |
| **Rollback Strategy** | **Implemented (Manual)** | Rolling back entails changing the container tag/model path back to the prior known-good version (e.g. `1.0.0`). |
| **Cloud Deployment** | *Ready / Configured* | Containerized and prepared for deployment to Hugging Face Spaces (Docker SDK). |
| **Live Operational Monitoring** | *Future Improvement* | Exposing Prometheus `/metrics` (latency, request count, confidence distribution) and Grafana dashboards. |
| **Data Versioning (DVC)** | *Future Improvement* | Tracking large image dataset hashes via DVC pointer files instead of raw Git. |
| **CI/CD Pipeline** | *Future Improvement* | Automated GitHub Actions workflow to build, test, and push images on `git push`. |

---

## 12. Troubleshooting & Common Pitfalls

1. **Windows Device Guard / WDAC Policy Blocks DLLs (`_pywrap_tensorflow_internal`):**
   * *Symptom:* `ImportError: DLL load failed: An Application Control policy has blocked this file.`
   * *Cause:* Windows security policies on work/school laptops block compiled C++ binaries executing from user directories (`AppData`, `D:\`).
   * *Fix:* Run training on Google Colab or inside Docker Desktop, where code executes inside a Linux kernel free of Windows DLL restrictions.

2. **Git LFS Pointer Text Files Causing Invalid Image Errors:**
   * *Symptom:* `InvalidArgumentError: Unknown image file format. One of JPEG, PNG, GIF required.`
   * *Cause:* Cloning Hugging Face repos via raw Git without `git-lfs` downloads 130-byte text pointer files instead of image binaries.
   * *Fix:* Export images using Python's `datasets` library or run `git lfs pull` inside the dataset folder.

3. **Keras 3 Deserialization Error (`data_format` or `get_tensor`):**
   * *Symptom:* `Unrecognized keyword arguments passed to RandomFlip: {'data_format': 'channels_last'}`
   * *Cause:* Model saved using Keras 3 (in Colab) loaded by older Keras 2 (`tf.keras`).
   * *Fix:* Install `keras>=3.0` inside Docker and load using `import keras; keras.models.load_model(MODEL_PATH)`.

4. **Network Read Timeout during Docker pip install:**
   * *Symptom:* `pip._vendor.urllib3.exceptions.ReadTimeoutError: Read timed out.`
   * *Fix:* Use the pre-built `tensorflow/tensorflow:2.16.1` base image and pass `--default-timeout=100` to pip.

---

## 13.  Future Improvements


### Future Improvements
1. **Fine-Tuning:** Unfreeze the top 20 layers of MobileNetV2 with a smaller learning rate (`1e-5`) to improve subtle boundary discrimination.
2. **ONNX Runtime Export:** Convert the `.keras` file to `.onnx` for 3x faster, lightweight CPU inference (~15 MB memory footprint).
3. **Automated CI/CD:** Add GitHub Actions to automate unit tests and image publishing to Docker Hub on every release tag.
4. **Monitoring:** Add a Prometheus `/metrics` endpoint to observe prediction confidence drift and response latency over time.

---

## 👨‍💻 Author & Credits

* **Developer:** **Jatin Karma**
* **Project Name:** **Humanly**
* **Dataset Credit:** LakoreAI ([Hugging Face Hub](https://huggingface.co/datasets/LakoreAI/human-nonhuman-face-classification))
* **Base Architecture:** Google MobileNetV2 (ImageNet weights)
