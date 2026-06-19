#!/usr/bin/env python3
"""Helper script with PlantVillage download instructions."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data" / "raw"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    print("PlantVillage dataset setup")
    print("=" * 60)
    print()
    print("1. Download the PlantVillage color dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print()
    print("   Or clone the public GitHub mirror:")
    print("   git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/raw/PlantVillage-Dataset")
    print()
    print("2. Extract so class folders live under:")
    print(f"   {TARGET}/")
    print("   Example:")
    print(f"   {TARGET}/color/Potato___Early_blight/*.jpg")
    print(f"   {TARGET}/PlantVillage-Dataset/raw/color/Pepper__bell___healthy/*.JPG")
    print()
    print("3. Install dependencies and train:")
    print("   python3 -m venv .venv && source .venv/bin/activate")
    print("   pip install -r requirements.txt")
    print("   python3 train.py")
    print()
    print("4. Predict on a leaf image:")
    print("   python3 predict.py path/to/leaf.jpg")
    print()
    print(f"Target directory ready: {TARGET}")


if __name__ == "__main__":
    main()
