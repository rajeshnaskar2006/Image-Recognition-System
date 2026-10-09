import numpy as np
import pytest

from object_detector import ObjectDetector, INPUT_SIZE


def encoded_detection(detector, box=(0, 0, 200, 100), category=2, score=0.9, objectness=0.95, index=0):
    output = np.zeros((len(detector.grid), 85), dtype=np.float32)
    left, top, width, height = box
    scale = 1.6  # 400 x 200 source -> 640 x 320, bottom padded.
    center = np.array([left + width / 2, top + height / 2]) * scale
    output[index, :2] = center / detector.strides[index] - detector.grid[index]
    output[index, 2:4] = np.log(np.array([width, height]) * scale / detector.strides[index])
    output[index, 4] = objectness
    output[index, 5 + category] = score
    return output


def test_detection_scores_are_not_inflated():
    detector = ObjectDetector(None)
    result = detector._decode(encoded_detection(detector), 400, 200, 1.6)
    assert result == [{"class_id": "coco:2", "label": "car", "confidence": 85.5}]


@pytest.mark.parametrize("box", [(0, 0, 10, 10), (0, 250, 150, 100)])
def test_tiny_and_padding_only_boxes_do_not_override_the_subject(box):
    detector = ObjectDetector(None)
    assert detector._decode(encoded_detection(detector, box=box), 400, 200, 1.6) == []


def test_low_score_and_nonfinite_detections_are_rejected():
    detector = ObjectDetector(None)
    assert detector._decode(encoded_detection(detector, score=0.3), 400, 200, 1.6) == []
    assert detector._decode(encoded_detection(detector, score=float("nan")), 400, 200, 1.6) == []


def test_duplicate_objects_of_one_class_are_not_added_into_a_fake_confidence():
    detector = ObjectDetector(None)
    output = encoded_detection(detector)
    second = encoded_detection(detector, box=(205, 0, 190, 100), index=1)
    output[1] = second[1]
    result = detector._decode(output, 400, 200, 1.6)
    assert result == [{"class_id": "coco:2", "label": "car", "confidence": 85.5}]


def test_input_channels_and_padding_match_the_published_model():
    class Net:
        def setInput(self, blob):
            self.blob = blob

        def forward(self):
            return np.zeros((1, 8400, 85), dtype=np.float32)

    net = Net()
    detector = ObjectDetector(net)
    image = np.full((200, 400, 3), (20, 40, 80), dtype=np.uint8)
    assert detector.detect(image) == []
    assert net.blob.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    np.testing.assert_array_equal(net.blob[0, :, 100, 100], [80, 40, 20])
    assert np.all(net.blob[0, :, 400, 0] == 114)
