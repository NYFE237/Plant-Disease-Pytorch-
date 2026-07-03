# Plant Disease Detection — PyTorch

Multi-architecture deep learning comparison for plant disease classification using the PlantVillage dataset. Three architectures trained and evaluated: Custom CNN (from scratch), ResNet50, and EfficientNet-B0 (transfer learning).

## Results

| Model           | Val Acc | Test Acc | Macro F1 | Macro Precision | Macro Recall |
|-----------------|---------|----------|----------|-----------------|--------------|
| Custom CNN      | 95.2%   | 95%      | 0.95     | 0.94            | 0.95         |
| ResNet50        | 98.9%   | 99%      | 0.98     | 0.97            | 0.98         |
| EfficientNet-B0 | 99.6%   | ~100%    | ~1.00    | ~0.99           | ~1.00        |

> **Note:** Results obtained on the controlled PlantVillage dataset (single-source, studio conditions). Remaining errors in EfficientNet-B0 are visually plausible confusions between morphologically similar disease classes (e.g. Tomato Early blight vs Late blight). Real-world field performance is expected to be lower.

## Dataset

- **Source**: PlantVillage (Kaggle — emmarex/plantdisease)
- **Classes**: 15 (tomato, potato, pepper — diseases + healthy)
- **Total images**: ~20,000
- **Split**: 70% train / 15% val / 15% test (random_split, seed=42)
- **Class imbalance**: handled via weighted CrossEntropyLoss (sklearn compute_class_weight)

## Models

### Custom CNN (from scratch)
- 3 conv blocks: Conv2d → BatchNorm → ReLU → MaxPool (32 → 64 → 128 filters)
- AdaptiveAvgPool2d(7,7) → Linear(6272, 512) → ReLU → Dropout(0.5) → Linear(512, 15)
- Optimizer: AdamW (lr=3e-4)
- Epochs: 15

### ResNet50 (Transfer Learning — ImageNet)
- Stage 1: backbone frozen, head trained (10 epochs, lr=1e-3)
- Stage 2: full fine-tuning (5 epochs, lr=1e-4)
- Head: Linear(2048, 15)
- Optimizer: AdamW

### EfficientNet-B0 (Transfer Learning — ImageNet)
- Stage 1: backbone frozen, classifier trained (10 epochs, lr=3e-4)
- Stage 2: full fine-tuning (5 epochs, lr=1e-4)
- Head: classifier[1] = Linear(1280, 15)
- Optimizer: AdamW

## Training Details

- **Framework**: PyTorch + torchvision
- **Original training environment**: NVIDIA Tesla V100 (16GB) — UniLu HPC cluster
- **Reproducible on**: Kaggle GPU T4 / Google Colab T4
- **Callbacks**: EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5, patience=2), ModelCheckpoint (best val accuracy)
- **Augmentation**: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(20), ColorJitter

## Key Observations

- **Custom CNN** shows expected train/val gap from epoch 11 (train 99.6% / val 95.2%) — moderate overfitting, addressable with stronger augmentation or dropout tuning.
- **ResNet50** main confusions: Tomato Early blight vs Late blight (visually similar lesion patterns), Tomato Target Spot vs Septoria leaf spot.
- **EfficientNet-B0** near-perfect diagonal in confusion matrix — errors are isolated and visually plausible, not artefacts of data leakage (overlap between train/test indices = 0).

## Repository Structure

```
plant-disease-pytorch/
├── train.py              # Full training script — all 3 models
├── requirements.txt
├── README.md
└── results/
    ├── metrics.json
    ├── comparison.png
    ├── cm_Custom_CNN.png
    ├── cm_ResNet50.png
    └── cm_EfficientNet-B0.png
```

## How to Run

### Kaggle (recommended)
```python
# In a Kaggle notebook with GPU T4 enabled
# Add dataset: emmarex/plantdisease
!python train.py
# Change DATA_DIR in train.py to: /kaggle/input/plantdisease/PlantVillage
```

### Google Colab
```python
from google.colab import drive, files
drive.mount('/content/drive')
files.upload()  # upload train.py + PlantVillage.zip
!unzip PlantVillage.zip -d dataset/
!pip install torch torchvision scikit-learn matplotlib seaborn -q
!python train.py
```

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
Pillow>=9.5.0
```

## Author

**Frank Evin Yami**
Master in Data Science — University of Luxembourg (2023–2025)
Thesis supervised by Dr. Senthil NAGARAJAN

Original thesis models trained on NVIDIA Tesla V100 (UniLu HPC).
This repository reimplements and extends the thesis work in native PyTorch.

GitHub: [github.com/NYFE237](https://github.com/NYFE237)
