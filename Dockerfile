FROM tensorflow/tensorflow:2.16.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/human_face_v1.keras \
    MODEL_VERSION=1.0.0

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn python-multipart
RUN pip install --no-cache-dir pillow
RUN pip install --no-cache-dir "keras>=3.8,<4"

COPY src ./src
COPY models ./models
COPY index.html ./index.html

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]