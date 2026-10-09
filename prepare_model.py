"""Download model weights and class labels, then run one warm-up inference."""
from classifier import ImageClassifier, MODEL_NAME

if __name__ == "__main__":
    print(f"Preparing {MODEL_NAME}, YOLOX, and YuNet. First setup downloads pretrained weights and labels.", flush=True)
    ImageClassifier().warm_up()
    print("Model is ready. Start the app with start.bat.", flush=True)
