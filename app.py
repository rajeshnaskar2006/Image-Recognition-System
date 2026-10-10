"""Image recognition web app for local use and hosted Flask deployments."""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from classifier import ImageClassifier, ImageValidationError, ModelUnavailableError, MODEL_NAME, MAX_UPLOAD_BYTES


def create_app(config: dict | None = None, classifier=None) -> Flask:
    app = Flask(__name__, static_folder="public/static", static_url_path="/static")
    # Leave room for multipart headers below Vercel's 4.5 MB request limit.
    request_upload_bytes = 4_000_000 if os.environ.get("VERCEL") == "1" else MAX_UPLOAD_BYTES
    app.config.update(
        MAX_CONTENT_LENGTH=request_upload_bytes + 128 * 1024,
        MAX_IMAGE_BYTES=request_upload_bytes,
        MAX_FORM_MEMORY_SIZE=128 * 1024,
        MAX_FORM_PARTS=10,
    )
    if config:
        app.config.update(config)
    predictor = classifier if classifier is not None else ImageClassifier()
    app.extensions["classifier"] = predictor

    @app.get("/")
    def index():
        return render_template(
            "index.html", upload_max_bytes=MAX_UPLOAD_BYTES,
            request_upload_bytes=app.config["MAX_IMAGE_BYTES"],
        )

    def upload_limit_message():
        if app.config["MAX_IMAGE_BYTES"] < MAX_UPLOAD_BYTES:
            return (
                "This upload is too large. Use the image picker to automatically prepare "
                "photos up to 25 MB, or upload an image under 4 MB directly."
            )
        return "The image exceeds the 25 MB upload limit. Large photo dimensions are resized automatically."

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model=MODEL_NAME, model_ready=predictor.ready)

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Choose an image to classify."), 400
        image_bytes = upload.read(app.config["MAX_IMAGE_BYTES"] + 1)
        if not image_bytes:
            return jsonify(error="The image is empty. Please choose another file."), 400
        if len(image_bytes) > app.config["MAX_IMAGE_BYTES"]:
            return jsonify(error=upload_limit_message()), 413
        try:
            return jsonify(predictor.predict(image_bytes))
        except ImageValidationError as exc:
            return jsonify(error=str(exc)), 400
        except ModelUnavailableError:
            app.logger.exception("Model initialization failed")
            return jsonify(error=(
                "The recognition models are unavailable. Please try again shortly. "
                "If this continues, the app owner should check the deployment or server log."
            )), 503
        except Exception:
            app.logger.exception("Image prediction failed")
            return jsonify(error="The prediction could not be completed. Please try another image."), 500

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error=upload_limit_message()), 413

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data:; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; "
            "base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


app = create_app()


if __name__ == "__main__":
    from waitress import serve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    if os.environ.get("PRELOAD_MODELS") == "1":
        logging.info("Loading recognition models before accepting requests")
        app.extensions["classifier"].warm_up()
    print(f"\n  Image Recognition System\n  Listening on http://{host}:{port}\n  Press Ctrl+C to stop.\n", flush=True)
    serve(app, host=host, port=port, threads=4, max_request_body_size=app.config["MAX_CONTENT_LENGTH"])
