# Plant Disease Classification

A Python project that trains a **leaf disease classifier** on the **PlantVillage** dataset using **transfer learning** with MobileNetV2. Given a photo of a crop leaf, the model predicts the plant type and disease (or healthy).

## What it does

| Step | Script | Output |
|------|--------|--------|
| Setup dataset | `download_data.py --clone` | Clones PlantVillage into `data/raw/` |
| Train | `train.py` | Saved model, labels, metrics, plots |
| Predict | `predict.py path/to/leaf.jpg` | Top disease predictions with confidence |

## PlantVillage dataset

PlantVillage contains tens of thousands of labeled leaf images across **14 crop species** and **38 classes** (healthy + diseased). Folder names encode crop and condition, for example:

- `Potato___Early_blight`
- `Tomato___healthy`
- `Pepper__bell___Bacterial_spot`

The loader auto-detects the correct class folder under `data/raw/` — you do not need a fixed directory depth.

## Project structure

```
plant_disease/
├── config.yaml              # Image size, epochs, paths
├── download_data.py         # Dataset download instructions
├── train.py                 # Train + evaluate + save model
├── predict.py               # Predict on a single image
├── requirements.txt
├── src/
│   ├── config.py            # YAML config loader
│   ├── data.py              # PlantVillage loader + tf.data pipelines
│   ├── model.py               # MobileNetV2 transfer-learning model
│   └── evaluate.py            # Metrics, confusion matrix plot
├── data/raw/                # PlantVillage images (not in git)
├── models/                  # Saved model + class labels (generated)
└── assets/                  # Training plots + metrics (generated)
```

## Requirements

- **Python 3.10, 3.11, or 3.12** — the versions [TensorFlow publishes wheels for](https://www.tensorflow.org/install/pip)
- TensorFlow 2.15+
- scikit-learn, matplotlib, numpy, pyyaml, Pillow

Create a virtual environment with a supported Python (for example 3.12), then install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pip install` reports no matching distribution for TensorFlow, check `python --version` and recreate `.venv` with Python 3.10–3.12.

## Quick start

### 1. Download PlantVillage

```bash
python download_data.py --clone
```

This clones the public GitHub mirror (~2 GB) into `data/raw/PlantVillage-Dataset/`. Training uses the full color set under `raw/color/` (38 classes, ~54k images).

For manual options, run `python download_data.py` without flags.

Example layout after clone:

```
data/raw/PlantVillage-Dataset/raw/color/Tomato___Late_blight/*.JPG
```

**Kaggle:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

### 2. Train the classifier

```bash
python3 train.py
```

Training runs in two phases:

1. **Frozen backbone** — train the classification head on top of ImageNet MobileNetV2.
2. **Fine-tuning** — unfreeze upper layers of MobileNetV2 for better leaf-specific features.

Generated artifacts:

| File | Contents |
|------|----------|
| `models/plant_disease_model.keras` | Trained Keras model |
| `models/class_labels.json` | Class names in label order |
| `assets/training_metrics.json` | Validation accuracy and F1 |
| `assets/training_history.png` | Accuracy / loss curves |
| `assets/confusion_matrix.png` | Validation confusion matrix |

### 3. Predict on a leaf image

```bash
python3 predict.py data/raw/color/Potato___Early_blight/some_leaf.jpg
```

Example output:

```
Image: data/raw/color/Potato___Early_blight/some_leaf.jpg
1. Potato___Early_blight | crop=Potato | condition=Early blight | confidence=97.42%
2. Potato___Late_blight | crop=Potato | condition=Late blight | confidence=1.85%
3. Potato___healthy | crop=Potato | condition=healthy | confidence=0.31%
```

## Configuration

Edit `config.yaml`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `data.image_size` | `[224, 224]` | Input size for MobileNetV2 |
| `data.validation_split` | `0.2` | Hold-out validation fraction |
| `data.max_samples_per_class` | `null` | Cap images per class for faster experiments |
| `training.epochs` | `8` | Head training epochs |
| `training.fine_tune_epochs` | `4` | Fine-tuning epochs |
| `training.batch_size` | `32` | Batch size |

For a quick test run on a laptop, set:

```yaml
data:
  max_samples_per_class: 150
training:
  epochs: 3
  fine_tune_epochs: 2
```

## How it works

### Transfer learning

MobileNetV2 is pre-trained on ImageNet. The project:

1. Replaces the top layer with a **38-class softmax** head (or however many classes exist in your folder).
2. Applies **MobileNetV2 preprocessing** and light **horizontal flip** augmentation.
3. Uses **early stopping** on validation accuracy.
4. Fine-tunes the last layers of the backbone for leaf texture and disease patterns.

### Evaluation

After training, the script reports:

- **Accuracy** on the validation split
- **Macro / weighted F1** from scikit-learn
- A **confusion matrix** heatmap saved to `assets/confusion_matrix.png`

## Key concepts

| Term | Meaning |
|------|---------|
| **PlantVillage** | Public dataset of crop leaf images with disease labels |
| **Transfer learning** | Reuse a pre-trained CNN instead of training from scratch |
| **MobileNetV2** | Lightweight CNN suited to laptops and edge devices |
| **Fine-tuning** | Unfreezing upper layers to adapt features to leaf images |
| **Confusion matrix** | Table showing predicted vs actual classes |

## Possible extensions

- Add a FastAPI endpoint for mobile field uploads.
- Export to TensorFlow Lite for on-device inference.
- Use class weights for imbalanced disease categories.
- Compare EfficientNetB0 or ResNet50 backbones.
- Deploy with Grad-CAM heatmaps for explainability.

## Author

**Duncan Mwirigi**  
GitHub: [github.com/duncanmwirigi](https://github.com/duncanmwirigi)  
X: https://x.com/AIStiqDan  
Website: https://bytecityinc.com

## License

MIT — use and modify freely for learning and projects.
