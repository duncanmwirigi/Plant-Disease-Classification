#!/usr/bin/env python3
"""Download or set up the PlantVillage dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data" / "raw"
GITHUB_MIRROR = "https://github.com/spMohanty/PlantVillage-Dataset.git"
DEFAULT_CLONE_DIR = TARGET / "PlantVillage-Dataset"


def clone_dataset(dest: Path = DEFAULT_CLONE_DIR) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    if dest.exists() and any(dest.iterdir()):
        print(f"Dataset already present at {dest}")
        return
    print(f"Cloning PlantVillage (~2 GB, may take several minutes)...")
    print(f"Destination: {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", GITHUB_MIRROR, str(dest)],
        check=True,
    )
    print(f"Done. Class images are under: {dest / 'raw' / 'color'}/")


def print_instructions() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    print("PlantVillage dataset setup")
    print("=" * 60)
    print()
    print("Fastest option — clone the public mirror:")
    print("   python download_data.py --clone")
    print()
    print("Or manually from Kaggle:")
    print("   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print()
    print("Manual git clone:")
    print(f"   git clone --depth 1 {GITHUB_MIRROR} data/raw/PlantVillage-Dataset")
    print()
    print("2. Class folders should end up under:")
    print(f"   {TARGET}/")
    print("   Example:")
    print(f"   {TARGET}/PlantVillage-Dataset/raw/color/Potato___Early_blight/*.JPG")
    print()
    print("3. Train:")
    print("   source .venv/bin/activate")
    print("   python train.py")
    print()
    print("4. Predict on a leaf image:")
    print("   python predict.py path/to/leaf.jpg")
    print()
    print(f"Target directory ready: {TARGET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the PlantVillage dataset")
    parser.add_argument(
        "--clone",
        action="store_true",
        help="Clone the GitHub mirror into data/raw/PlantVillage-Dataset",
    )
    args = parser.parse_args()

    if args.clone:
        try:
            clone_dataset()
        except subprocess.CalledProcessError as exc:
            print(f"Clone failed: {exc}", file=sys.stderr)
            sys.exit(exc.returncode)
    else:
        print_instructions()


if __name__ == "__main__":
    main()
