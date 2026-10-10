"""Prepare the model bundle and publish static assets during a Vercel build."""
from pathlib import Path
import shutil

from classifier import ImageClassifier


def main():
    root = Path(__file__).resolve().parent
    print("Preparing recognition models for deployment.", flush=True)
    ImageClassifier().warm_up()
    # Vercel serves public/ directly; preserve the app's existing /static URLs.
    shutil.copytree(root / "static", root / "public" / "static", dirs_exist_ok=True)
    print("Models and static assets are ready.", flush=True)


if __name__ == "__main__":
    main()
