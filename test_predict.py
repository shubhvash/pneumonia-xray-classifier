"""Send chest X-rays to the running service and print the predictions.

Usage:
    uv run python test_predict.py                                  # local service, sample images
    uv run python test_predict.py --url https://<your-app>.onrender.com
    uv run python test_predict.py path/to/image.jpeg ...
"""
import argparse
import json

import requests

DEFAULT_IMAGES = [
    "data/chest_xray/NORMAL/IM-0031-0001.jpeg",
    "data/chest_xray/INFECTED/person1_virus_11.jpeg",
    "data/chest_xray/INFECTED/person100_bacteria_475.jpeg",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="*", default=DEFAULT_IMAGES)
    p.add_argument("--url", default="http://localhost:9696")
    args = p.parse_args()
    base = args.url.rstrip("/")

    print("health:", json.dumps(requests.get(f"{base}/health", timeout=60).json()))
    for path in args.images:
        with open(path, "rb") as f:
            r = requests.post(f"{base}/predict", files={"file": (path, f, "image/jpeg")}, timeout=60)
        r.raise_for_status()
        print(f"{path}: {r.json()}")


if __name__ == "__main__":
    main()
