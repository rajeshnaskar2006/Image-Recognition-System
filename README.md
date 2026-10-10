# Image Recognition System

A complete local web app for recognizing objects in uploaded images. Built with **Python, TensorFlow/Keras, OpenCV, and Flask**, with a responsive browser interface named **Lens**.

## Start on Windows

1. Double-click **setup.bat** once. It creates `.venv`, installs the dependencies, and downloads EfficientNetV2B3, ImageNet labels, the YOLOX object detector, and the YuNet face detector. Internet access is required for setup; TensorFlow is a large download.
2. Double-click **start.bat**.
3. Open **http://127.0.0.1:5000** in your browser.
4. Choose or drop a JPEG, PNG, or WebP image and select **Analyze image**. Large photos resize automatically. The page shows detected objects, a **Person / People** result for prominent human faces, or up to five classification matches when no prominent object is detected.

**Updating an existing installation:** stop the running server, get the updated files, and run `.\.venv\Scripts\python.exe prepare_model.py` once to download the replacement models. Then restart the server and refresh the browser. No new Python dependencies are required for this update.

Stop the server with **Ctrl+C** in its terminal. After setup, recognition works locally without an internet connection. Uploaded images are not retained by this application. No cloud prediction API or API key is needed.

The setup supports 64-bit Python 3.10–3.12; Python 3.12 is recommended. It looks for Python on your computer and can also use the compatible runtime bundled with Codex. If neither is available, install [Python 3.12](https://www.python.org/downloads/). Moving a project containing `.venv` to another folder can break that environment; run setup again in the final location with a fresh `.venv`.

## Features

- Drag and drop or browse for an image, with an immediate preview.
- Five ranked object predictions and confidence bars, without reloading the page.
- Human-face detection for portraits, so a detected face is reported as a person instead of being forced into an unrelated ImageNet label.
- Whole-object detection for 80 common categories, including cars, people, animals, and everyday items, before the more detailed ImageNet classifier is used.
- Visible loading and error states, file size checks, and a low-confidence notice.
- Automatic resizing to a maximum 1,600-pixel working edge, preserving proportions and the original file.
- Actual file-content validation, a 25 MiB upload limit, and a 120-megapixel / 20,000-pixel-per-side source safety limit. Ordinary high-resolution photos no longer hit the old 20-megapixel rejection.
- Pillow validation, reduced JPEG decoding, EXIF orientation, white compositing for transparency, and OpenCV color conversion and aspect-preserving model input.
- Pretrained EfficientNetV2B3, YOLOX, and YuNet loaded once per server process and reused for subsequent predictions.
- Responsive, keyboard-accessible interface with no external font or JavaScript dependencies.
- Local Waitress server, with debug mode disabled.

## How recognition works

The server checks the source dimensions before decoding, verifies its format, and reduces large photos to a working image with a maximum edge of 1,600 pixels. JPEGs use reduced-resolution decoding where possible. Orientation and proportions are preserved, and transparent pixels are composited on white. The response reports both original and processed dimensions; the source file is never overwritten. A safety cap remains necessary because decoding arbitrarily large or malicious files can exhaust memory. Images are decoded and predicted one at a time per process.

YuNet checks for human faces at a detection threshold of 0.9. If a detected face occupies at least 4% of the image area, the result is **Person** (or **People** for multiple prominent faces). Tiny background faces do not replace the main object prediction. This is face detection only: the app does not identify anyone. A detection result has no ImageNet percentage; the interface explicitly says **Face detected** instead of inventing a confidence score. Occluded, tiny, or unusual faces may still be missed.

Next, YOLOX-S detects whole objects across 80 COCO categories. It uses a 640 × 640 input with preserved proportions, suppresses overlapping duplicate detections, and keeps detections scoring at least 0.5 and occupying at least 4% of the source image. Up to five distinct detected categories are shown. These are detected subjects, not competing labels for one object. Scores are the detector's objectness multiplied by class score; scores from separate objects are not added or normalized. A Car result can therefore take precedence over a classifier's focus on a wheel. A Dog result names the broad object rather than claiming a specific breed. Detection scores and classification scores measure different things and should not be treated as directly comparable probabilities.

When neither detector finds a prominent subject, OpenCV fits the entire RGB image into a 300 × 300 square with neutral padding. It does not stretch the subject or crop away the ends of a wide car. EfficientNetV2B3 performs its own normalization on RGB values in the 0–255 range, and returns five ImageNet categories. [Keras reports](https://keras.io/api/applications/) 82.0% ImageNet top-1 accuracy for EfficientNetV2B3 compared with 71.3% for the previous MobileNetV2; these are dataset benchmark results, not a guarantee for an individual photo.

This project uses pretrained weights, not a model trained on your own photos. The object classifier still recognizes 1,000 predefined categories and cannot recognize every possible object. Its scores are reported without boosting, sharpening, or renormalizing the top five. Low-confidence results remain visible and explicitly marked as uncertain. The scores need not total 100%, because the other categories also receive scores. A clear photo with one prominent object works best; unusual subjects may need a different dataset or a custom trained model.

Actual latency depends on your CPU and image. First-time model initialization is slower; subsequent requests reuse the loaded model. Processing time includes validation, preprocessing, model initialization if needed, and inference, but excludes browser upload time. Windows runs this project on CPU by default.

## Files

```text
app.py                  Flask endpoints and local server
classifier.py           Image validation, OpenCV processing, Keras prediction
object_detector.py      YOLOX preprocessing and whole-object detection
prepare_model.py        Download/cache the models and check an inference
templates/index.html    Web interface
public/static/styles.css Responsive styling
public/static/app.js      Uploads, previews, progress, and result rendering
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

## Deploy on Vercel (free Hobby plan)

This app includes a Vercel configuration for Python 3.12 and Flask. Vercel's Hobby plan is for personal, non-commercial projects and has monthly usage limits. Check the current [Hobby terms and allowances](https://vercel.com/docs/plans/hobby).

- Import this GitHub repository into Vercel and select the **Hobby** plan.
- Keep the detected **Flask** framework and the checked-in build command, `python build_vercel.py`.
- Enable **Fluid compute** and **Large Functions**. New projects are eligible for Large Functions by default; if necessary, set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in the project environment and redeploy. The large-function feature is in public beta and is needed for the TensorFlow dependencies. See [Vercel's function limits](https://vercel.com/docs/functions/limitations).
- Model weights and labels are prepared during the build and included with the deployment. Static assets are tracked in `public/static` for Vercel's CDN and served at the same `/static` URLs locally. The models load on the first prediction in each new function instance, so the first result can take longer.
- Verify both the home page and an actual image prediction after deployment. `/api/health` can report `model_ready: false` until that instance processes its first image.

The browser accepts photos up to 25 MiB. On Vercel, larger uploads are prepared in the browser to fit under the platform's 4.5 MB request limit; the direct prediction API accepts at most 4,000,000 image bytes plus a small multipart allowance. The original file stays unchanged. Recognition still uses the same three pretrained models. Local and Render deployments retain the original 25 MiB API limit.

The deployment uses the free plan's 2 GB memory allocation. Actual memory and execution time depend on the image; this is intended for a personal demo with light usage. Uploaded photos are sent to the hosting server for analysis and are not retained by the app.

## Deploy on Render (paid alternative)

The repository includes a `render.yaml` Blueprint for a Python web service. It uses Python 3.12, the existing Waitress server, and one process so the recognition models are not duplicated in memory.

**Compute cost:** the Blueprint selects Render's paid **1 CPU / 2 GB** plan (`1c-2g`, formerly Standard), currently **$25/month** for compute. Review [Render's pricing](https://render.com/pricing) before creating the service; other usage charges can apply. The 512 MB Free and Starter plans are not recommended for this TensorFlow app. Monitor actual memory after deployment; unusually large images or heavy traffic may require more RAM.

1. Sign in to [Render](https://dashboard.render.com/) and connect this GitHub repository.
2. Choose **New > Blueprint**, select the repository and `master` branch, and review the service and cost before deploying.
3. Render installs dependencies and prepares all three models during the build. Startup loads the cached models before the service accepts requests.
4. Once the service is live, open its `https://...onrender.com` address and analyze an image. `/api/health` should report `model_ready: true`.

For a manually created **Web Service**, use these equivalent settings:

| Setting | Value |
| --- | --- |
| Language | Python 3 |
| Branch | `master` |
| Region | Singapore (or your preferred nearby region) |
| Build command | `pip install -r requirements.txt && python prepare_model.py` |
| Start command | `python app.py` |
| Health check | `/api/health` |
| Environment | `HOST=0.0.0.0`, `PRELOAD_MODELS=1`, `PYTHONUNBUFFERED=1`, `TF_NUM_INTRAOP_THREADS=1`, `TF_NUM_INTEROP_THREADS=1`, `OMP_NUM_THREADS=1` |

The `.python-version` file selects Python 3.12. Leave `PORT` unset: Render provides it. Keep model preparation in the build command, not the pre-deploy command. The model cache is generated in the project directory as part of the build; no persistent disk is required. Future pushes to the linked branch automatically redeploy by default.

Hosted uploads are sent to the Render server for analysis; the app does not retain them. Render provides HTTPS for its public URL. Without authentication, anyone with that URL can use the app. Local use still defaults to `127.0.0.1:5000`.

## API

`GET /api/health` reports whether the server is running and whether its in-process model is loaded. `model_ready: false` before the first prediction is normal, even if weights are already downloaded.

`POST /api/predict` accepts multipart form data with an `image` field:

```powershell
curl.exe -F "image=@C:\path\to\photo.jpg" http://127.0.0.1:5000/api/predict
```

A successful response contains `method`, `predictions`, `image`, `elapsed_ms`, `model`, `face_count`, and `low_confidence`.

- `method: "classification"`: five predictions with `class_id`, `label`, and percentage `confidence`; `model` is `EfficientNetV2B3`.
- `method: "object_detection"`: one to five distinct prominent detected subjects with percentage detection `confidence`; `class_id` uses a `coco:` prefix and `model` is `YOLOX`. These scores do not sum to 100%.
- `method: "face_detection"`: one Person / People result with `confidence: null`; `model` is `YuNet`, and `face_count` is the number of prominent detected faces. Clients must handle null confidence rather than displaying 0% or inventing a percentage.
- `image` includes original oriented `width` and `height`, `processed_width`, `processed_height`, and a `resized` flag.

Invalid images or sources beyond the safety caps return HTTP 400, oversized file uploads 413, model initialization failures 503, and unexpected prediction failures 500. Error responses include an `error` message.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests exercise Flask routes, decoding and automatic resizing (including a real 24-megapixel JPEG), transparent images, EXIF orientation, aspect-preserving input, face-result semantics, background-face filtering, object detector preprocessing, duplicate suppression, score handling, and whole-object priority. Injected predictors keep the unit suite independent of model downloads. `prepare_model.py` separately verifies that the models can load and the TensorFlow classifier can run.

## Troubleshooting

- **The first prediction is slow:** model initialization is still running. Model downloads happen once and are cached in `.model-cache`.
- **Model could not start:** run `setup.bat` with internet access, then restart the app. Check the terminal for the underlying error.
- **TensorFlow DLL error:** install the Microsoft Visual C++ x64 Redistributable linked from the [TensorFlow installation guide](https://www.tensorflow.org/install/pip), then restart the terminal.
- **Port 5000 is busy:** in PowerShell, set `$env:PORT = '5001'`, then run `.\.venv\Scripts\python.exe app.py` and open `http://127.0.0.1:5001`.
- **Incorrect prediction:** try a clear image with a single object. ImageNet's category set is limited and scores can be wrong.

## Model references

- [Keras EfficientNetV2 documentation](https://keras.io/api/applications/efficientnet_v2/efficientnet_v2_models/)
- [Keras model benchmarks](https://keras.io/api/applications/)
- [OpenCV YuNet model and MIT license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [OpenCV YOLOX-S model and Apache-2.0 license](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox)
- [TensorFlow installation guide](https://www.tensorflow.org/install/pip)

The included server defaults to `127.0.0.1` for local use. The Render configuration sets `HOST=0.0.0.0` and uses Render's assigned `PORT` and HTTPS endpoint.
