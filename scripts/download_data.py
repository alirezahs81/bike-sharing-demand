"""Download the UCI Bike Sharing dataset into data/raw/."""

from __future__ import annotations

import os
import zipfile

import requests

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
RAW_DIR = os.path.join("data", "raw")
ZIP_PATH = os.path.join("data", "bike+sharing+dataset.zip")


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(os.path.join(RAW_DIR, "hour.csv")):
        print("Dataset already present in data/raw/ - skipping download")
        return
    print(f"Downloading {URL} ...")
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(RAW_DIR)
    print("Extracted to data/raw/:", os.listdir(RAW_DIR))


if __name__ == "__main__":
    main()
