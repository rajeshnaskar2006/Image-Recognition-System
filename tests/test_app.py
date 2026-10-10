"""HTTP tests use a predictor stub so no model download is necessary."""

from io import BytesIO

import pytest

from app import create_app
from classifier import ImageValidationError, ModelUnavailableError, MODEL_NAME


class StubClassifier:
    ready = True

    def __init__(self, error=None):
        self.error = error
        self.received = []

    def predict(self, image_bytes):
        self.received.append(image_bytes)
        if self.error is not None:
            raise self.error
        return {
            "predictions": [
                {"label": "golden retriever", "class_id": "n02099601", "confidence": 92.5}
            ],
            "image": {"width": 640, "height": 480},
            "elapsed_ms": 24.2,
            "model": MODEL_NAME,
            "low_confidence": False,
        }


@pytest.fixture
def classifier():
    return StubClassifier()


@pytest.fixture
def client(classifier):
    app = create_app({"TESTING": True}, classifier=classifier)
    return app.test_client()


def upload(client, contents=b"fixture-image", filename="photo.png"):
    return client.post(
        "/api/predict",
        data={"image": (BytesIO(contents), filename)},
        content_type="multipart/form-data",
    )


def assert_json_error(response, status):
    assert response.status_code == status
    assert response.is_json
    assert response.get_json().get("error")


def test_home_renders_upload_interface(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.content_type
    assert b"image" in response.data.lower()


def test_health_is_available(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.is_json


def test_vercel_upload_limit_reserves_multipart_overhead(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    predictor = StubClassifier()
    app = create_app({"TESTING": True}, classifier=predictor)
    assert app.config["MAX_CONTENT_LENGTH"] < 4_500_000
    response = upload(app.test_client(), b"x" * 4_000_001)
    assert_json_error(response, 413)
    assert not predictor.received
    assert "image picker" in response.get_json()["error"]


def test_local_upload_limit_stays_at_25_mib(monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    app = create_app({"TESTING": True}, classifier=StubClassifier())
    assert app.config["MAX_IMAGE_BYTES"] == 25 * 1024 * 1024


def test_success_returns_predictions_and_passes_file_bytes(client, classifier):
    response = upload(client)
    assert response.status_code == 200
    result = response.get_json()
    assert result["model"] == MODEL_NAME
    assert result["predictions"][0]["label"] == "golden retriever"
    assert result["predictions"][0]["confidence"] == 92.5
    assert result["image"] == {"width": 640, "height": 480}
    assert classifier.received == [b"fixture-image"]


def test_missing_file_returns_validation_error(client, classifier):
    assert_json_error(client.post("/api/predict", data={}), 400)
    assert not classifier.received


def test_empty_file_returns_validation_error(client, classifier):
    assert_json_error(upload(client, b""), 400)
    assert not classifier.received


def test_bad_image_returns_validation_error():
    predictor = StubClassifier(ImageValidationError("Invalid or corrupted image."))
    client = create_app({"TESTING": True}, classifier=predictor).test_client()
    assert_json_error(upload(client, b"not an image"), 400)


def test_file_byte_limit_applies_below_request_limit():
    predictor = StubClassifier()
    client = create_app(
        {"TESTING": True, "MAX_IMAGE_BYTES": 16, "MAX_CONTENT_LENGTH": 4096},
        classifier=predictor,
    ).test_client()
    assert_json_error(upload(client, b"x" * 17), 413)
    assert not predictor.received


def test_multipart_request_limit_returns_json():
    predictor = StubClassifier()
    client = create_app(
        {"TESTING": True, "MAX_CONTENT_LENGTH": 512}, classifier=predictor
    ).test_client()
    assert_json_error(upload(client, b"x" * 1024), 413)
    assert not predictor.received


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ModelUnavailableError("private download URL token=secret"), 503),
        (RuntimeError("C:/private/path secret-debug-information"), 500),
    ],
)
def test_service_errors_are_json_without_internal_details(error, status):
    client = create_app(
        {"TESTING": True}, classifier=StubClassifier(error)
    ).test_client()
    response = upload(client)
    assert_json_error(response, status)
    assert str(error) not in response.get_data(as_text=True)
