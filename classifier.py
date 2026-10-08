"""Validated image decoding and a lazily loaded Keras ImageNet classifier."""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
import threading
import time
import warnings

# Keep both model downloads and Keras settings inside this project.
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".model-cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "20000000")

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_PIXELS = 20_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
logger = logging.getLogger(__name__)


class ImageValidationError(ValueError):
    """The uploaded file cannot safely be used as an image."""


class ModelUnavailableError(RuntimeError):
    """The model or its class labels could not be loaded."""


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Check image headers before allocating pixels, then decode with OpenCV."""
    if not image_bytes:
        raise ImageValidationError("The image is empty. Please choose another file.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as header:
                if header.format not in ALLOWED_FORMATS:
                    raise ImageValidationError("Please upload a JPEG, PNG, or WebP image.")
                width, height = header.size
                if width < 1 or height < 1 or width * height > MAX_PIXELS:
                    raise ImageValidationError("The image is too large. Please use an image under 20 megapixels.")
                if getattr(header, "n_frames", 1) != 1:
                    raise ImageValidationError("Please choose a still image instead of an animated image.")
                header.verify()
        # IMREAD_COLOR normalizes grayscale/alpha and respects EXIF orientation.
        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ImageValidationError("This image could not be read. Try saving it as a new JPEG or PNG.")
        if decoded.shape[0] * decoded.shape[1] > MAX_PIXELS:
            raise ImageValidationError("The image is too large. Please use an image under 20 megapixels.")
        return decoded
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("The image is too large. Please use an image under 20 megapixels.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, cv2.error) as exc:
        raise ImageValidationError("This is not a readable JPEG, PNG, or WebP image.") from exc


class ImageClassifier:
    """One cached model per process; serialize initialization and inference."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._preprocess = None
        self._decode_predictions = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        try:
            from tensorflow.keras.applications.mobilenet_v2 import (
                MobileNetV2, decode_predictions, preprocess_input,
            )

            model = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
            # Fetch and cache labels at initialization, too; fail cleanly if offline.
            decode_predictions(np.zeros((1, 1000), dtype=np.float32), top=1)
            model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
            self._preprocess = preprocess_input
            self._decode_predictions = decode_predictions
            self._model = model
        except Exception as exc:
            logger.exception("Could not initialize MobileNetV2")
            raise ModelUnavailableError(
                "The model could not start. On first use, connect to the internet so the model can download, "
                "then try again. If this continues, run setup.bat and check the server log."
            ) from exc

    def warm_up(self) -> None:
        with self._lock:
            self._load_locked()

    def predict(self, image_bytes: bytes) -> dict:
        started = time.perf_counter()
        decoded = decode_image(image_bytes)
        height, width = decoded.shape[:2]
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        with self._lock:
            self._load_locked()
            scores = self._model(self._preprocess(batch), training=False).numpy()
            matches = self._decode_predictions(scores, top=5)[0]
        predictions = [
            {"class_id": class_id, "label": label.replace("_", " "), "confidence": round(float(score) * 100, 2)}
            for class_id, label, score in matches
        ]
        return {
            "predictions": predictions,
            "image": {"width": width, "height": height},
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "model": "MobileNetV2",
            "low_confidence": predictions[0]["confidence"] < 40,
        }
