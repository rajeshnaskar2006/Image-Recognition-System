"""Exercise real image decoding without loading TensorFlow or model weights."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import pytest

import classifier
from classifier import ImageClassifier, ImageValidationError, decode_image, read_image, prepare_model_input, MODEL_NAME, MODEL_SIZE


def encode_image(format_name="PNG", size=(17, 13), mode="RGB"):
    image = Image.new(mode, size, color=127 if mode == "L" else (20, 80, 180))
    output = BytesIO()
    image.save(output, format=format_name)
    return output.getvalue()


def test_decodes_opencv_png_to_bgr_uint8():
    original = np.zeros((13, 17, 3), dtype=np.uint8)
    original[:, :] = (180, 80, 20)
    encoded, data = cv2.imencode(".png", original)
    assert encoded
    decoded = decode_image(data.tobytes())
    assert decoded.shape == (13, 17, 3)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, original)


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "WEBP"])
def test_supported_image_formats_decode(format_name):
    result = decode_image(encode_image(format_name))
    assert result.shape == (13, 17, 3)
    assert result.dtype == np.uint8


def test_grayscale_image_becomes_three_channels():
    result = decode_image(encode_image(mode="L"))
    assert result.shape == (13, 17, 3)
    np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
    np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])


def test_transparent_png_becomes_three_channels():
    output = BytesIO()
    Image.new("RGBA", (17, 13), (20, 80, 180, 100)).save(output, format="PNG")
    assert decode_image(output.getvalue()).shape == (13, 17, 3)


def test_jpeg_exif_orientation_is_applied():
    output = BytesIO()
    image = Image.new("RGB", (17, 13), (20, 80, 180))
    exif = Image.Exif()
    exif[274] = 6  # Rotate a landscape image 90 degrees clockwise.
    image.save(output, format="JPEG", exif=exif)
    assert decode_image(output.getvalue()).shape == (17, 13, 3)


def test_animated_png_is_rejected():
    output = BytesIO()
    first = Image.new("RGB", (17, 13), "red")
    second = Image.new("RGB", (17, 13), "blue")
    first.save(output, format="PNG", save_all=True, append_images=[second], duration=100)
    with pytest.raises(ImageValidationError):
        decode_image(output.getvalue())


@pytest.mark.parametrize("image_bytes", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_corrupt_image_is_rejected(image_bytes):
    with pytest.raises(ImageValidationError):
        decode_image(image_bytes)


@pytest.mark.parametrize("format_name", ["GIF", "BMP", "TIFF"])
def test_unsupported_actual_format_is_rejected(format_name):
    with pytest.raises(ImageValidationError):
        decode_image(encode_image(format_name))


def test_truncated_png_is_rejected():
    image_bytes = encode_image()
    with pytest.raises(ImageValidationError):
        decode_image(image_bytes[: len(image_bytes) // 2])


def test_source_limit_is_checked_before_pixels_are_loaded(monkeypatch):
    monkeypatch.setattr(classifier, "MAX_SOURCE_PIXELS", 100)

    def fail_if_called(*args, **kwargs):
        pytest.fail("OpenCV must not allocate an oversized image")

    monkeypatch.setattr(Image.Image, "thumbnail", fail_if_called)
    with pytest.raises(ImageValidationError):
        decode_image(encode_image(size=(11, 10)))


def test_image_at_pixel_limit_is_accepted(monkeypatch):
    monkeypatch.setattr(classifier, "MAX_SOURCE_PIXELS", 100)
    assert decode_image(encode_image(size=(10, 10))).shape == (10, 10, 3)


@pytest.mark.parametrize(("score", "low_confidence"), [(0.91, False), (0.2, True)])
def test_inference_converts_pixels_and_formats_model_output(score, low_confidence):
    predictor = ImageClassifier()
    assert not predictor.ready
    observed = {}

    class Scores:
        def numpy(self):
            return np.zeros((1, 1000), dtype=np.float32)

    class Model:
        def __call__(self, batch, training):
            observed["batch"] = batch
            observed["training"] = training
            return Scores()

    def decode_predictions(scores, top):
        assert scores.shape == (1, 1000)
        assert top == 5
        return [[("n02099601", "golden_retriever", score)]]

    predictor._model = Model()
    predictor._face_detector = NoFaces()
    predictor._object_detector = NoObjects()
    predictor._decode_predictions = decode_predictions
    result = predictor.predict(encode_image())

    assert predictor.ready
    assert observed["training"] is False
    assert observed["batch"].shape == (1, MODEL_SIZE, MODEL_SIZE, 3)
    assert observed["batch"].dtype == np.float32
    # The model expects RGB even though OpenCV decodes into BGR.
    np.testing.assert_allclose(
        observed["batch"][0, MODEL_SIZE // 2, MODEL_SIZE // 2], np.array([20, 80, 180]), atol=1e-6
    )
    assert result["predictions"] == [
        {"class_id": "n02099601", "label": "golden retriever", "confidence": round(score * 100, 2)}
    ]
    assert result["image"]["width"] == 17
    assert result["image"]["height"] == 13
    assert result["image"]["resized"] is False
    assert result["model"] == MODEL_NAME
    assert result["low_confidence"] is low_confidence
    assert result["elapsed_ms"] >= 0


class NoFaces:
    def setInputSize(self, size):
        self.size = size

    def detect(self, image):
        return 0, None


class NoObjects:
    def detect(self, image):
        return []


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "WEBP"])
def test_large_photo_is_resized_automatically(format_name, monkeypatch):
    monkeypatch.setattr(classifier, "MAX_WORKING_SIDE", 100)
    image = read_image(encode_image(format_name, size=(800, 400)))
    assert image.pixels.shape == (50, 100, 3)
    assert image.metadata == {"width": 800, "height": 400, "processed_width": 100, "processed_height": 50, "resized": True}


def test_24_megapixel_jpeg_no_longer_requires_manual_resizing():
    image = read_image(encode_image("JPEG", size=(6000, 4000)))
    assert image.metadata["resized"] is True
    assert (image.original_width, image.original_height) == (6000, 4000)
    assert max(image.pixels.shape[:2]) <= classifier.MAX_WORKING_SIDE
    assert abs(image.pixels.shape[1] / image.pixels.shape[0] - 1.5) < 0.01


def test_extreme_dimensions_rejected_before_resize(monkeypatch):
    monkeypatch.setattr(classifier, "MAX_SOURCE_SIDE", 100)
    with pytest.raises(ImageValidationError, match="safety limit"):
        read_image(encode_image(size=(101, 2)))


def test_transparent_pixels_are_composited_on_white():
    output = BytesIO()
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(output, format="PNG")
    assert np.all(decode_image(output.getvalue()) == 255)


def test_resize_preserves_exif_orientation_and_original_dimensions(monkeypatch):
    monkeypatch.setattr(classifier, "MAX_WORKING_SIDE", 100)
    output = BytesIO()
    source = Image.new("RGB", (800, 400), "red")
    exif = Image.Exif()
    exif[274] = 6
    source.save(output, format="JPEG", exif=exif)
    image = read_image(output.getvalue())
    assert image.pixels.shape == (100, 50, 3)
    assert (image.original_width, image.original_height) == (400, 800)


def test_whole_wide_image_is_retained_without_stretching():
    source = np.full((100, 400, 3), (30, 50, 200), dtype=np.uint8)
    batch = prepare_model_input(source)
    assert batch.shape == (1, MODEL_SIZE, MODEL_SIZE, 3)
    assert np.all(batch[0, 0] == 127)
    # Both horizontal ends survive; the image remains a 4:1 rectangle.
    np.testing.assert_array_equal(batch[0, 150, 0], [200, 50, 30])
    np.testing.assert_array_equal(batch[0, 150, -1], [200, 50, 30])
    assert np.count_nonzero(batch[0, :, 150, 0] == 200) == 75
    assert batch.max() > 1  # Built-in model normalization must not run twice.


def test_portrait_returns_person_without_inventing_confidence():
    class FaceDetector(NoFaces):
        def detect(self, image):
            face = np.zeros((1, 15), dtype=np.float32)
            face[0, 2:4] = (8, 8)
            face[0, 14] = 0.97
            return 1, face

    predictor = ImageClassifier()
    predictor._face_detector = FaceDetector()
    predictor._object_detector = NoObjects()
    def forbidden_model(*args, **kwargs):
        pytest.fail("Portrait must not be forced into an ImageNet category")
    predictor._model = forbidden_model
    result = predictor.predict(encode_image())
    assert result["method"] == "face_detection"
    assert result["predictions"] == [{"class_id": "person", "label": "person", "confidence": None}]
    assert result["low_confidence"] is False
    assert result["face_count"] == 1


def test_tiny_background_face_does_not_override_main_object():
    class TinyFace(NoFaces):
        def detect(self, image):
            face = np.zeros((1, 15), dtype=np.float32)
            face[0, 2:4] = (2, 2)
            return 1, face

    predictor = ImageClassifier()
    predictor._face_detector = TinyFace()
    assert predictor._portrait_face_count(np.zeros((100, 200, 3), dtype=np.uint8)) == 0


def test_car_detector_takes_priority_over_part_classifier():
    class CarDetector:
        def detect(self, image):
            return [{"class_id": "coco:2", "label": "car", "confidence": 91.2}]

    predictor = ImageClassifier()
    predictor._face_detector = NoFaces()
    predictor._object_detector = CarDetector()
    def forbidden_model(*args, **kwargs):
        pytest.fail("A detected whole car must not be replaced by an ImageNet part label")
    predictor._model = forbidden_model
    result = predictor.predict(encode_image())
    assert result["method"] == "object_detection"
    assert result["predictions"][0]["label"] == "car"
    assert result["predictions"][0]["confidence"] == 91.2
    assert result["model"] == "YOLOX"
