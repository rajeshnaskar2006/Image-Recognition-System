# Image Recognition System

A complete local web app for recognizing objects in uploaded images. Built with **Python, TensorFlow/Keras, OpenCV, and Flask**, with a responsive browser interface named **Lens**.

## Start on Windows

1. Double-click **setup.bat** once. It creates `.venv`, installs the dependencies, and downloads MobileNetV2 weights and ImageNet labels. Internet access is required for setup; TensorFlow is a large download.
2. Double-click **start.bat**.
3. Open **http://127.0.0.1:5000** in your browser.
4. Choose or drop a JPEG, PNG, or WebP image and select **Analyze image**. The page shows the five strongest matches, confidence scores, dimensions, and processing time.

Stop the server with **Ctrl+C** in its terminal. After setup, recognition works locally without an internet connection. Uploaded images are not retained by this application. No cloud prediction API or API key is needed.

The setup supports 64-bit Python 3.10–3.12; Python 3.12 is recommended. It looks for Python on your computer and can also use the compatible runtime bundled with Codex. If neither is available, install [Python 3.12](https://www.python.org/downloads/). Moving a project containing `.venv` to another folder can break that environment; run setup again in the final location with a fresh `.venv`.

## Features

- Drag and drop or browse for an image, with an immediate preview.
- Five ranked predictions and confidence bars, without reloading the page.
- Visible loading and error states, file size checks, and a low-confidence notice.
- Actual file-content validation, a 10 MiB upload limit, and a 20-megapixel decoded-image limit.
- OpenCV decoding, EXIF orientation, BGR-to-RGB conversion, resizing, and Keras preprocessing.
- Pretrained MobileNetV2 loaded once per server process and reused for subsequent predictions.
- Responsive, keyboard-accessible interface with no external font or JavaScript dependencies.
- Local Waitress server, with debug mode disabled.

## How recognition works

The server verifies the image, uses OpenCV to decode and resize it to 224 × 224 pixels, converts its channels to RGB, and applies MobileNetV2 preprocessing. Keras runs the pretrained neural network and returns the top five ImageNet categories.

This project uses existing ImageNet-trained weights; it does **not** claim to train a new model. The model recognizes 1,000 predefined categories. It classifies the whole image and does not locate or count individual objects, draw bounding boxes, identify people, or recognize every possible object. Confidence scores are model scores, not guarantees of correctness. The model can assign a label even when none of its known categories fits. Use a clear photo with one prominent object for better results. The top five scores need not total 100% because the remaining categories also receive scores.

Actual latency depends on your CPU and image. First-time model initialization is slower; subsequent requests reuse the loaded model. Processing time includes validation, preprocessing, model initialization if needed, and inference, but excludes browser upload time. Windows runs this project on CPU by default.

## Files

```text
app.py                  Flask endpoints and local server
classifier.py           Image validation, OpenCV processing, Keras prediction
prepare_model.py        Download/cache the model and check an inference
templates/index.html    Web interface
static/styles.css       Responsive styling
static/app.js           Uploads, previews, progress, and result rendering
setup.bat / setup.ps1   Windows setup
start.bat               Windows launcher
requirements.txt        Pinned application dependencies
requirements-dev.txt    Test dependencies
tests/                  API and image-validation tests
.model-cache/           Downloaded weights and labels, created during setup
.venv/                  Project-specific Python environment, created during setup
```

## Manual setup

From the project directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_model.py
.\.venv\Scripts\python.exe app.py
```

On macOS or Linux, use `python3 -m venv .venv` and `.venv/bin/python` for the equivalent commands. The batch launchers are Windows-specific.

## API

`GET /api/health` reports whether the server is running and whether its in-process model is loaded. `model_ready: false` before the first prediction is normal, even if weights are already downloaded.

`POST /api/predict` accepts multipart form data with an `image` field:

```powershell
curl.exe -F "image=@C:\path\to\photo.jpg" http://127.0.0.1:5000/api/predict
```

A successful response contains `predictions` (five entries with `class_id`, `label`, and percentage `confidence`), `image` (width and height), `elapsed_ms`, `model`, and `low_confidence`. Invalid images return HTTP 400, oversized uploads 413, model initialization failures 503, and unexpected prediction failures 500. Error responses include an `error` message.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests exercise the real Flask routes and OpenCV image decoding, using an injected predictor for HTTP tests so no model downloads are needed during the test suite. `prepare_model.py` separately verifies that the actual TensorFlow model can load and perform inference.

## Troubleshooting

- **The first prediction is slow:** model initialization is still running. Model downloads happen once and are cached in `.model-cache`.
- **Model could not start:** run `setup.bat` with internet access, then restart the app. Check the terminal for the underlying error.
- **TensorFlow DLL error:** install the Microsoft Visual C++ x64 Redistributable linked from the [TensorFlow installation guide](https://www.tensorflow.org/install/pip), then restart the terminal.
- **Port 5000 is busy:** in PowerShell, set `$env:PORT = '5001'`, then run `.\.venv\Scripts\python.exe app.py` and open `http://127.0.0.1:5001`.
- **Incorrect prediction:** try a clear image with a single object. ImageNet's category set is limited and scores can be wrong.

## Model references

- [Keras MobileNet documentation](https://keras.io/api/applications/mobilenet/)
- [TensorFlow installation guide](https://www.tensorflow.org/install/pip)

The included server binds to `127.0.0.1` for use on this computer. Public deployment would need its own hosting configuration, HTTPS, and request limits appropriate to the service.
