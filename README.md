# Plant Disease Detection — PyTorch

Multi-architecture deep learning comparison for plant disease classification using the PlantVillage dataset.

## Models Compared
| Model | Architecture | Strategy |
|---|---|---|
| Custom CNN | 3 conv layers + classifier head | Trained from scratch |
| ResNet50 | Transfer learning (ImageNet) | Two-stage fine-tuning |
| EfficientNet-B0 | Transfer learning (ImageNet) | Two-stage fine-tuning |

## Dataset
- **Source**: PlantVillage (Kaggle — emmarex/plantdisease)
- **Classes**: 15 (tomato, potato, pepper diseases + healthy)
- **Split**: 70% train / 15% val / 15% test
- **Class imbalance**: handled via weighted CrossEntropyLoss

## Training Environment
- Previously trained on **NVIDIA Tesla V100 (16GB)** — UniLu HPC
- Reproducible on **Kaggle GPU T4** or **Google Colab T4**
- Framework: **PyTorch** + torchvision

## Results
See `results/metrics.json` after training.

## How to Run

### On Kaggle
1. Upload `train.py` to a Kaggle notebook
2. Add the PlantVillage dataset
3. Set `DATA_DIR = "/kaggle/input/plantdisease/PlantVillage"`
4. Enable GPU (T4) in notebook settings
5. Run `!python train.py`

### On Google Colab
```python
from google.colab import files
files.upload()  # upload train.py + PlantVillage.zip
!unzip PlantVillage.zip -d dataset/
!pip install torch torchvision scikit-learn matplotlib seaborn
!python train.py
```

## Architecture Details

### Custom CNN
- 3 conv blocks (32 → 64 → 128 filters) with BatchNorm + MaxPool
- AdaptiveAvgPool2d → Dense(512) → Dropout(0.5) → Dense(15)

### ResNet50
- Backbone frozen → train head (10 epochs, lr=1e-3)
- Full fine-tune (5 epochs, lr=1e-4)
- `fc = Linear(2048, 15)`

### EfficientNet-B0
- Backbone frozen → train head (10 epochs, lr=3e-4)
- Full fine-tune (5 epochs, lr=1e-4)
- `classifier[1] = Linear(1280, 15)`

## Key Techniques
- Data augmentation: random flip, rotation, ColorJitter
- Class weights via `sklearn.utils.class_weight.compute_class_weight`
- Early stopping (patience=5)
- ReduceLROnPlateau scheduler
- ModelCheckpoint (best val accuracy)

## Author
**Frank Evin Yami** — Master in Data Science, University of Luxembourg  
Part of thesis work on deep learning for agricultural diagnostics.  
Previous training: NVIDIA Tesla V100, UniLu HPC cluster.
