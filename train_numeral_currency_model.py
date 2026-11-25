"""
Standalone Training Script: Color-Blind, Numeral-Focused Indian Currency Identifier
Architecture: ConvNeXt-Tiny (Pretrained) with Grayscale & High-Contrast Numeral Augmentations
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

# 1. Config
DATASET_ROOT = Path("Indian Currency Dataset")
OUTPUT_DIR = Path("currency_models_balanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["10", "20", "50", "100", "200", "500", "2000"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 35
PATIENCE = 8
SEED = 42

TEST_PER_CLASS = 30
VAL_PER_CLASS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {DEVICE}")

# 2. Audit & Deduplication
records = []
for class_name in CLASS_NAMES:
    folder = DATASET_ROOT / class_name
    for path in folder.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                width, height = im.size
            records.append({
                "path": str(path), "class_name": class_name,
                "width": width, "height": height,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            })
        except Exception:
            continue

df = pd.DataFrame(records).drop_duplicates(subset="sha256", keep="first").reset_index(drop=True)
print(f"Total clean images: {len(df)}")

# 3. Balanced Splits
rng = np.random.default_rng(SEED)
train_rows, val_rows, test_rows = [], [], []

for class_name in CLASS_NAMES:
    class_frame = df[df.class_name == class_name].sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_rows.append(class_frame.iloc[:TEST_PER_CLASS])
    val_rows.append(class_frame.iloc[TEST_PER_CLASS:TEST_PER_CLASS + VAL_PER_CLASS])
    train_rows.append(class_frame.iloc[TEST_PER_CLASS + VAL_PER_CLASS:])

train_df = pd.concat(train_rows, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
val_df = pd.concat(val_rows, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df = pd.concat(test_rows, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

# 4. Transforms
MEAN = [0.45, 0.45, 0.45]
STD = [0.22, 0.22, 0.22]

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Color-blind
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(1.4, 2.4)),
    transforms.RandomAdjustSharpness(sharpness_factor=2.5, p=0.7),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomRotation(12),
    transforms.RandomPerspective(distortion_scale=0.20, p=0.4),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class CurrencyDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB")
        return self.transform(image), CLASS_TO_IDX[row.class_name], row.path

train_dataset = CurrencyDataset(train_df, train_transform)
val_dataset = CurrencyDataset(val_df, eval_transform)
test_dataset = CurrencyDataset(test_df, eval_transform)

train_counts = Counter(train_df.class_name)
weights = [1.0 / train_counts[c] for c in train_df.class_name]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

workers = 0 if os.name == "nt" else 2
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=workers)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=workers)

# 5. Model
def build_model(num_classes=7):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.30),
        nn.Linear(256, num_classes)
    )
    return model

model = build_model(len(CLASS_NAMES)).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

def run_epoch(loader, training=False):
    model.train(training)
    total_loss, y_true, y_pred = 0.0, [], []
    for images, labels, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer); scaler.update()
        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy())
    return total_loss / len(loader.dataset), np.array(y_true), np.array(y_pred)

print("\nStarting ConvNeXt-Tiny Numeral-Focused Training...")
best_f1, stale = -1.0, 0
checkpoint_path = OUTPUT_DIR / "best_currency_model.pth"

for epoch in range(1, EPOCHS + 1):
    train_loss, _, _ = run_epoch(train_loader, training=True)
    val_loss, val_true, val_pred = run_epoch(val_loader, training=False)
    val_macro_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
    val_acc = accuracy_score(val_true, val_pred)
    scheduler.step()
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | Val Acc: {val_acc:.4f}")
    if val_macro_f1 > best_f1:
        best_f1, stale = val_macro_f1, 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "mean": MEAN, "std": STD, "img_size": IMG_SIZE,
            "epoch": epoch, "val_macro_f1": float(val_macro_f1)
        }, checkpoint_path)
    else:
        stale += 1
        if stale >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

# 6. Evaluate
print("\nRunning Test Set Evaluation...")
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_loss, y_true, y_pred = run_epoch(test_loader, training=False)
test_acc = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(f"Test Accuracy: {test_acc:.4f} | Test Macro-F1: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[f"Rs.{x}" for x in CLASS_NAMES], digits=4, zero_division=0))

# 7. ONNX Export
print("\nExporting ONNX model...")
onnx_path = OUTPUT_DIR / "currency_model.onnx"
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
torch.onnx.export(
    model, dummy, str(onnx_path),
    input_names=["input_image"], output_names=["class_logits"],
    dynamic_axes={"input_image": {0: "batch"}, "class_logits": {0: "batch"}},
    opset_version=17
)

labels = {
    "class_names": CLASS_NAMES, "class_to_idx": CLASS_TO_IDX,
    "mean": MEAN, "std": STD, "img_size": IMG_SIZE,
    "confidence_threshold": 0.70,
    "balanced_test_per_class": TEST_PER_CLASS,
    "color_mode": "grayscale_numeral_focused",
    "backbone": "ConvNeXt-Tiny"
}
with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2)

print(f"Saved ONNX model to {onnx_path}")
print(f"Saved config to {OUTPUT_DIR / 'labels.json'}")
print("Training & Export complete!")
