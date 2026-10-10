"""COCO whole-object detection using OpenCV's published YOLOX-S weights.

Model/reference: https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox
The model is distributed under Apache-2.0; it is downloaded, not bundled here.
"""
import cv2
import numpy as np

MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
)
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
INPUT_SIZE = 640
SCORE_THRESHOLD = 0.5
MIN_AREA_FRACTION = 0.04


class ObjectDetector:
    def __init__(self, net):
        self.net = net
        grids, strides = [], []
        for stride in (8, 16, 32):
            side = INPUT_SIZE // stride
            y, x = np.indices((side, side))
            grids.append(np.column_stack((x.ravel(), y.ravel())))
            strides.append(np.full((side * side, 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    @classmethod
    def from_pretrained(cls, model_path: str | None = None):
        if model_path is None:
            from tensorflow.keras.utils import get_file

            model_path = get_file("object_detection_yolox_2022nov.onnx", MODEL_URL,
                                  file_hash=MODEL_SHA256, hash_algorithm="sha256", cache_subdir="models")
        return cls(cv2.dnn.readNetFromONNX(model_path))

    def detect(self, image: np.ndarray) -> list[dict]:
        height, width = image.shape[:2]
        scale = INPUT_SIZE / max(height, width)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[:size[1], :size[0]] = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        # This export expects RGB 0..255, top-left aligned padding, and NCHW.
        self.net.setInput(cv2.dnn.blobFromImage(padded))
        output = self.net.forward()[0]
        return self._decode(output, width, height, scale)

    def _decode(self, output, width, height, scale) -> list[dict]:
        class_scores = output[:, 5:] * output[:, 4:5]
        categories = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        selected = np.flatnonzero(np.isfinite(scores) & (scores >= SCORE_THRESHOLD))
        boxes, kept_scores, kept_categories = [], [], []
        for index in selected:
            center = (output[index, :2] + self.grid[index]) * self.strides[index]
            extent = np.exp(np.clip(output[index, 2:4], -20, 20)) * self.strides[index]
            left, top = np.maximum((center - extent / 2) / scale, 0)
            right, bottom = np.minimum((center + extent / 2) / scale, (width, height))
            box_width, box_height = max(0, right - left), max(0, bottom - top)
            if box_width * box_height < width * height * MIN_AREA_FRACTION:
                continue
            boxes.append([float(left), float(top), float(box_width), float(box_height)])
            kept_scores.append(float(scores[index]))
            kept_categories.append(int(categories[index]))
        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxesBatched(boxes, kept_scores, kept_categories, SCORE_THRESHOLD, 0.5)
        # Show distinct detected subjects, using the best actual score for each class.
        best = {}
        for index in np.asarray(keep).reshape(-1):
            category = kept_categories[index]
            best[category] = max(best.get(category, 0), kept_scores[index])
        return [
            {"class_id": f"coco:{category}", "label": LABELS[category], "confidence": round(score * 100, 2)}
            for category, score in sorted(best.items(), key=lambda item: item[1], reverse=True)[:5]
        ]
