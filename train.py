import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR    = "dataset/PlantVillage"   # adapte si besoin
IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_CLASSES = 15
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── TRANSFORMS ──────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── DATASET ─────────────────────────────────────────────────────────────────
full_ds     = datasets.ImageFolder(DATA_DIR)
CLASS_NAMES = full_ds.classes
N           = len(full_ds)
n_train     = int(0.70 * N)
n_val       = int(0.15 * N)
n_test      = N - n_train - n_val

generator = torch.Generator().manual_seed(SEED)
train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=generator)

train_ds.dataset.transform = train_tf
val_ds.dataset.transform   = val_tf
test_ds.dataset.transform  = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Dataset : {N} images | {NUM_CLASSES} classes")
print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

# ── CLASS WEIGHTS ────────────────────────────────────────────────────────────
train_labels = [full_ds.targets[i] for i in train_ds.indices]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)
weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# ── MODELS ───────────────────────────────────────────────────────────────────
def build_cnn(num_classes):
    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
        def forward(self, x):
            return self.classifier(self.pool(self.features(x)))
    return CustomCNN()


def build_resnet50(num_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(2048, num_classes)
    return m


def build_efficientnet(num_classes):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    m.classifier[1] = nn.Linear(1280, num_classes)
    return m


# ── TRAINING LOOP ────────────────────────────────────────────────────────────
def train_model(model, name, epochs=15, lr=1e-3, fine_tune_epoch=None, fine_tune_lr=None):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    best_weights  = None
    patience_ctr  = 0
    PATIENCE      = 5

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Fine-tune stage
        if fine_tune_epoch and epoch == fine_tune_epoch:
            print(f"\n  ↳ Fine-tuning all layers (lr={fine_tune_lr})")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr)

        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        tr_acc  = t_correct / t_total
        val_acc = v_correct / v_total
        tr_loss = t_loss / t_total
        vl_loss = v_loss / v_total

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(vl_loss)

        print(f"  [{name}] Epoch {epoch:02d}/{epochs} | "
              f"Train {tr_acc:.4f} / Loss {tr_loss:.4f} | "
              f"Val {val_acc:.4f} / Loss {vl_loss:.4f} | "
              f"{time.time()-t0:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f"results/{name.replace(' ', '_')}.pth")
    return model, history


# ── EVALUATION ───────────────────────────────────────────────────────────────
def evaluate(model, name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    print(f"\n{'='*60}")
    print(f"  {name} — Test Results")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(13, 11))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cmap="Blues", linewidths=0.3)
    plt.title(f"Confusion Matrix — {name}", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results/cm_{name.replace(' ', '_')}.png", dpi=120)
    plt.close()

    return report


# ── COMPARISON PLOT ───────────────────────────────────────────────────────────
def plot_comparison(histories):
    colors = {"CNN": "#e87c3e", "ResNet50": "#4ec4a0", "EfficientNet-B0": "#a07be8"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, hist in histories.items():
        c = colors.get(name, "gray")
        axes[0].plot(hist["val_loss"], label=name, color=c, linewidth=2)
        axes[1].plot(hist["val_acc"],  label=name, color=c, linewidth=2)
    for ax, title, ylabel in zip(axes,
                                  ["Validation Loss", "Validation Accuracy"],
                                  ["Loss", "Accuracy"]):
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=120)
    plt.close()
    print("Comparison plot saved.")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    histories = {}
    reports   = {}

    # 1. Custom CNN
    print("\n" + "="*60)
    print("  MODEL 1: Custom CNN")
    print("="*60)
    cnn, cnn_hist = train_model(build_cnn(NUM_CLASSES), "CNN", epochs=15, lr=3e-4)
    histories["CNN"] = cnn_hist
    reports["CNN"]   = evaluate(cnn, "CNN")

    # 2. ResNet50
    print("\n" + "="*60)
    print("  MODEL 2: ResNet50")
    print("="*60)
    resnet, resnet_hist = train_model(
        build_resnet50(NUM_CLASSES), "ResNet50",
        epochs=15, lr=1e-3,
        fine_tune_epoch=11, fine_tune_lr=1e-4
    )
    histories["ResNet50"] = resnet_hist
    reports["ResNet50"]   = evaluate(resnet, "ResNet50")

    # 3. EfficientNet-B0
    print("\n" + "="*60)
    print("  MODEL 3: EfficientNet-B0")
    print("="*60)
    effnet, effnet_hist = train_model(
        build_efficientnet(NUM_CLASSES), "EfficientNet-B0",
        epochs=15, lr=3e-4,
        fine_tune_epoch=11, fine_tune_lr=1e-4
    )
    histories["EfficientNet-B0"] = effnet_hist
    reports["EfficientNet-B0"]   = evaluate(effnet, "EfficientNet-B0")

    # Comparison plot
    plot_comparison(histories)

    # Final summary
    print("\n" + "="*60)
    print("  FINAL COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'Test Acc':>10} {'Macro F1':>10} {'Macro P':>10} {'Macro R':>10}")
    print("-" * 62)
    summary = {}
    for name, r in reports.items():
        acc = r["accuracy"]
        f1  = r["macro avg"]["f1-score"]
        p   = r["macro avg"]["precision"]
        rec = r["macro avg"]["recall"]
        summary[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                         "macro_precision": round(p, 4), "macro_recall": round(rec, 4)}
        print(f"{name:<20} {acc:>10.4f} {f1:>10.4f} {p:>10.4f} {rec:>10.4f}")

    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAll results saved in results/")
