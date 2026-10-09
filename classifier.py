"""Portrait detection and ImageNet classification with bounded image resizing."""
from __future__ import annotations

from dataclasses import dataclass
import io
import logging
import os
from pathlib import Path
import threading
import time
import warnings

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".model-cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from object_detector import ObjectDetector

MODEL_NAME = "EfficientNetV2B3"
MODEL_SIZE = 300
MAX_SOURCE_PIXELS = 120_000_000
MAX_SOURCE_SIDE = 20_000
MAX_WORKING_SIDE = 1600
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
# Retain a decompression-bomb guard while accepting high-resolution photos.
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS
FACE_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
FACE_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
logger = logging.getLogger(__name__)


class ImageValidationError(ValueError):
    """The uploaded file cannot safely be used as an image."""


class ModelUnavailableError(RuntimeError):
    """One of the pretrained models or class labels could not be loaded."""


@dataclass(frozen=True)
class DecodedImage:
    pixels: np.ndarray
    original_width: int
    original_height: int

    @property
    def metadata(self) -> dict:
        height, width = self.pixels.shape[:2]
        return {
            "width": self.original_width,
            "height": self.original_height,
            "processed_width": width,
            "processed_height": height,
            "resized": (width, height) != (self.original_width, self.original_height),
        }


def _source_limit_error() -> ImageValidationError:
    return ImageValidationError(
        "This image exceeds the safety limit of 120 megapixels or 20,000 pixels per side. "
        "Photos within these limits are resized automatically."
    )


def read_image(image_bytes: bytes) -> DecodedImage:
    """Verify headers, shrink photos, and preserve shape, orientation and alpha."""
    if not image_bytes:
        raise ImageValidationError("The image is empty. Please choose another file.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as header:
                if header.format not in ALLOWED_FORMATS:
                    raise ImageValidationError("Please upload a JPEG, PNG, or WebP image.")
                width, height = header.size
                if min(width, height) < 1 or width * height > MAX_SOURCE_PIXELS or max(width, height) > MAX_SOURCE_SIDE:
                    raise _source_limit_error()
                if getattr(header, "n_frames", 1) != 1:
                    raise ImageValidationError("Please choose a still image instead of an animated image.")
                header.verify()

            with Image.open(io.BytesIO(image_bytes)) as source:
                orientation = source.getexif().get(274, 1)
                if orientation in (5, 6, 7, 8):
                    width, height = height, width
                # JPEG draft decoding reduces memory before pixels are allocated.
                if source.format == "JPEG":
                    source.draft("RGB", (MAX_WORKING_SIDE, MAX_WORKING_SIDE))
                source.thumbnail((MAX_WORKING_SIDE, MAX_WORKING_SIDE), Image.Resampling.LANCZOS, reducing_gap=3.0)
                oriented = ImageOps.exif_transpose(source)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    rgb = Image.alpha_composite(canvas, rgba).convert("RGB")
                else:
                    rgb = oriented.convert("RGB")
                pixels = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                return DecodedImage(pixels, width, height)
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise _source_limit_error() from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, cv2.error) as exc:
        raise ImageValidationError("This is not a readable JPEG, PNG, or WebP image.") from exc


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Return the bounded BGR image for callers needing only pixels."""
    return read_image(image_bytes).pixels


def prepare_model_input(bgr: np.ndarray) -> np.ndarray:
    """Fit the whole image into a square without stretching or cutting off objects."""
    height, width = bgr.shape[:2]
    scale = MODEL_SIZE / max(width, height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, target, interpolation=interpolation)
    canvas = np.full((MODEL_SIZE, MODEL_SIZE, 3), 127, dtype=np.float32)
    x = (MODEL_SIZE - target[0]) // 2
    y = (MODEL_SIZE - target[1]) // 2
    canvas[y:y + target[1], x:x + target[0]] = resized
    # EfficientNetV2 includes normalization; input must stay in [0, 255].
    return np.expand_dims(canvas, axis=0)


class ImageClassifier:
    """Cache both models; serialize decoding and inference to bound memory use."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._decode_predictions = None
        self._face_detector = None
        self._object_detector = None

    @property
    def ready(self) -> bool:
        return self._model is not None and self._face_detector is not None and self._object_detector is not None

    def _load_locked(self) -> None:
        if self.ready:
            return
        try:
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3, decode_predictions
            from tensorflow.keras.utils import get_file

            face_path = get_file(
                "face_detection_yunet_2023mar.onnx", FACE_MODEL_URL,
                file_hash=FACE_MODEL_SHA256, hash_algorithm="sha256", cache_subdir="models",
            )
            face_detector = cv2.FaceDetectorYN.create(face_path, "", (320, 320), 0.9, 0.3, 5000)
            object_detector = ObjectDetector.from_pretrained()
            model = EfficientNetV2B3(weights="imagenet", include_top=True, include_preprocessing=True)
            decode_predictions(np.zeros((1, 1000), dtype=np.float32), top=1)
            model(np.zeros((1, MODEL_SIZE, MODEL_SIZE, 3), dtype=np.float32), training=False)
            self._decode_predictions = decode_predictions
            self._face_detector = face_detector
            self._object_detector = object_detector
            self._model = model
        except Exception as exc:
            logger.exception("Could not initialize image recognition models")
            raise ModelUnavailableError(
                "The models could not start. Connect to the internet and run setup.bat to download the new models."
            ) from exc

    def warm_up(self) -> None:
        with self._lock:
            self._load_locked()

    def _portrait_face_count(self, image: np.ndarray) -> int:
        height, width = image.shape[:2]
        scale = min(1.0, 640 / max(width, height))
        small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
        self._face_detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._face_detector.detect(small)
        if faces is None or len(faces) == 0:
            return 0
        # A tiny face in a car/window/background must not override the main object.
        areas = np.maximum(faces[:, 2], 0) * np.maximum(faces[:, 3], 0) / (small.shape[0] * small.shape[1])
        return int(np.count_nonzero(areas >= 0.04))

    def predict(self, image_bytes: bytes) -> dict:
        started = time.perf_counter()
        with self._lock:
            decoded = read_image(image_bytes)
            self._load_locked()
            face_count = self._portrait_face_count(decoded.pixels)
            objects = [] if face_count else self._object_detector.detect(decoded.pixels)
            if face_count:
                # Detection is not a probability over ImageNet's categories.
                predictions = [{"class_id": "person", "label": "person" if face_count == 1 else "people", "confidence": None}]
                method = "face_detection"
                result_model = "YuNet"
            elif objects:
                predictions = objects
                method = "object_detection"
                result_model = "YOLOX"
            else:
                scores = self._model(prepare_model_input(decoded.pixels), training=False).numpy()
                matches = self._decode_predictions(scores, top=5)[0]
                predictions = [
                    {"class_id": class_id, "label": label.replace("_", " "), "confidence": round(float(score) * 100, 2)}
                    for class_id, label, score in matches
                ]
                method = "classification"
                result_model = MODEL_NAME
        return {
            "predictions": predictions,
            "image": decoded.metadata,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "model": result_model,
            "method": method,
            "face_count": face_count,
            "low_confidence": method == "classification" and predictions[0]["confidence"] < 40,
        }
