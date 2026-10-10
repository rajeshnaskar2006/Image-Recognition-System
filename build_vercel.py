"""Prepare the model bundle during a Vercel build."""
from classifier import ImageClassifier


def main():
    print("Preparing recognition models for deployment.", flush=True)
    ImageClassifier().warm_up()
    print("Models are ready.", flush=True)


if __name__ == "__main__":
    main()
