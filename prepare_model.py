"""Download model weights and class labels, then run one warm-up inference."""
from classifier import ImageClassifier

if __name__ == "__main__":
    print("Preparing MobileNetV2. First setup downloads pretrained weights and labels.", flush=True)
    ImageClassifier().warm_up()
    print("Model is ready. Start the app with start.bat.", flush=True)
