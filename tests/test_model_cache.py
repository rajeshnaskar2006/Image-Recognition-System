"""Prepared deployments must load models without writable caches or downloads."""
import hashlib
from io import BytesIO
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import classifier
from classifier import ImageClassifier, ModelUnavailableError


@pytest.fixture
def prepared_cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    models = cache / "models"
    models.mkdir(parents=True)
    hashes = {}
    for name in classifier.BUNDLED_MODEL_HASHES:
        contents = name.encode()
        if name.endswith(".json"):
            contents = json.dumps({str(i): [f"n{i:08}", f"label_{i}"] for i in range(1000)}).encode()
        (models / name).write_bytes(contents)
        hashes[name] = ("sha256", hashlib.sha256(contents).hexdigest())
    monkeypatch.setenv("KERAS_HOME", str(cache))
    monkeypatch.setattr(classifier, "BUNDLED_MODEL_HASHES", hashes)
    return models


def stub_backends(monkeypatch, download):
    observed = {}

    class Model:
        def __call__(self, batch, training):
            scores = np.zeros((1, 1000), dtype=np.float32)
            scores[0, 7] = 0.75
            return SimpleNamespace(numpy=lambda: scores)

    def efficientnet(**kwargs):
        observed["weights"] = kwargs["weights"]
        return Model()

    def keras_decoder(scores, top):
        return [[("n00000007", "label_7", 0.75)]]

    class Faces:
        def setInputSize(self, size):
            pass

        def detect(self, image):
            return 0, None

    class Net:
        def setInput(self, blob):
            pass

        def forward(self):
            return np.zeros((1, 8400, 85), dtype=np.float32)

    def face_model(path, *args):
        observed["face_path"] = path
        return Faces()

    def object_model(path):
        observed["object_path"] = path
        return Net()

    monkeypatch.setitem(sys.modules, "tensorflow.keras.applications.efficientnet_v2", SimpleNamespace(
        EfficientNetV2B3=efficientnet, decode_predictions=keras_decoder,
    ))
    monkeypatch.setitem(sys.modules, "tensorflow.keras.utils", SimpleNamespace(get_file=download))
    monkeypatch.setattr(classifier.cv2.FaceDetectorYN, "create", face_model)
    monkeypatch.setattr(classifier.cv2.dnn, "readNetFromONNX", object_model)
    return observed


def test_complete_readonly_bundle_runs_prediction_without_downloads(prepared_cache, monkeypatch):
    def forbidden_download(*args, **kwargs):
        pytest.fail("A prepared deployment must not download models or labels")

    observed = stub_backends(monkeypatch, forbidden_download)
    monkeypatch.setattr(classifier.os, "access", lambda *args: False)
    source = BytesIO()
    Image.new("RGB", (32, 24), "white").save(source, format="PNG")
    result = ImageClassifier().predict(source.getvalue())
    assert observed == {
        "weights": str(prepared_cache / "efficientnetv2-b3.h5"),
        "face_path": str(prepared_cache / "face_detection_yunet_2023mar.onnx"),
        "object_path": str(prepared_cache / "object_detection_yolox_2022nov.onnx"),
    }
    assert result["method"] == "classification"
    assert result["predictions"][0] == {"class_id": "n00000007", "label": "label 7", "confidence": 75.0}
    assert len(result["predictions"]) == 5


def test_incomplete_writable_cache_retains_local_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("KERAS_HOME", str(tmp_path))
    downloads = []

    def download(name, *args, **kwargs):
        downloads.append(name)
        return str(tmp_path / name)

    observed = stub_backends(monkeypatch, download)
    ImageClassifier().warm_up()
    assert downloads == ["face_detection_yunet_2023mar.onnx", "object_detection_yolox_2022nov.onnx"]
    assert observed["weights"] == "imagenet"


def test_incomplete_readonly_bundle_fails_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv("KERAS_HOME", str(tmp_path))
    monkeypatch.setattr(classifier.os, "access", lambda *args: False)
    stub_backends(monkeypatch, lambda *args, **kwargs: pytest.fail("Unexpected download"))
    with pytest.raises(ModelUnavailableError) as error:
        ImageClassifier().warm_up()
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_corrupt_bundle_is_rejected_before_loading_models(prepared_cache, monkeypatch):
    (prepared_cache / "efficientnetv2-b3.h5").write_bytes(b"corrupt")
    monkeypatch.setattr(classifier.os, "access", lambda *args: False)
    observed = stub_backends(monkeypatch, lambda *args, **kwargs: pytest.fail("Unexpected download"))
    with pytest.raises(ModelUnavailableError) as error:
        ImageClassifier().warm_up()
    assert "checksum" in str(error.value.__cause__)
    assert observed == {}


def test_corrupt_writable_cache_can_be_repaired_by_existing_downloaders(prepared_cache):
    (prepared_cache / "efficientnetv2-b3.h5").write_bytes(b"corrupt")
    assert classifier._bundled_model_files() is None


def test_bundled_labels_preserve_top_scores_and_batch_order(prepared_cache):
    decoder = classifier._bundled_prediction_decoder(prepared_cache / "imagenet_class_index.json")
    scores = np.zeros((2, 1000), dtype=np.float32)
    scores[0, [3, 7, 11]] = [0.25, 0.6, 0.1]
    scores[1, 2] = 0.91
    result = decoder(scores, top=2)
    assert [entry[0] for entry in result[0]] == ["n00000007", "n00000003"]
    assert result[0][0][2] == scores[0, 7]
    assert result[1][0][0] == "n00000002"
    with pytest.raises(ValueError, match="shape"):
        decoder(np.zeros((1, 999)))
