"""Exercise real image decoding without loading TensorFlow or model weights."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import pytest

import classifier
from classifier import ImageClassifier, ImageValidationError, decode_image


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


def test_pixel_limit_is_checked_before_opencv_decode(monkeypatch):
    monkeypatch.setattr(classifier, "MAX_PIXELS", 100)

    def fail_if_called(*args, **kwargs):
        pytest.fail("OpenCV must not allocate an oversized image")

    monkeypatch.setattr(cv2, "imdecode", fail_if_called)
    with pytest.raises(ImageValidationError):
        decode_image(encode_image(size=(11, 10)))


def test_image_at_pixel_limit_is_accepted(monkeypatch):
    monkeypatch.setattr(classifier, "MAX_PIXELS", 100)
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
    predictor._preprocess = lambda batch: batch / 127.5 - 1
    predictor._decode_predictions = decode_predictions
    result = predictor.predict(encode_image())

    assert predictor.ready
    assert observed["training"] is False
    assert observed["batch"].shape == (1, 224, 224, 3)
    assert observed["batch"].dtype == np.float32
    # The model expects RGB even though OpenCV decodes into BGR.
    np.testing.assert_allclose(
        observed["batch"][0, 0, 0], np.array([20, 80, 180]) / 127.5 - 1, atol=1e-6
    )
    assert result["predictions"] == [
        {"class_id": "n02099601", "label": "golden retriever", "confidence": round(score * 100, 2)}
    ]
    assert result["image"] == {"width": 17, "height": 13}
    assert result["model"] == "MobileNetV2"
    assert result["low_confidence"] is low_confidence
    assert result["elapsed_ms"] >= 0
