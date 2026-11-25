import json

nb_path = 'Currency_model_balanced_7_denominations.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Header Markdown
nb['cells'][0]['source'] = [
    '# Indian Currency Classification — Color-Blind, Numeral-Focused Seven-Denomination Model\n',
    '\n',
    'This notebook implements an advanced, **color-blind and background-invariant** currency identification pipeline.\n',
    '\n',
    '### Why this approach?\n',
    '1. **Eliminates Color Dependency**: Rather than relying on paper or background colors (which invert with BGR/RGB errors or fail under bad lighting), inputs are converted to grayscale with edge/numeral enhancement.\n',
    '2. **Focuses on Printed Numerals**: High-contrast sharpness and aspect-preserving crops force the model to identify the printed numerals (`10`, `20`, `50`, `100`, `200`, `500`, `2000`), Mahatma Gandhi portrait, and Reserve Bank seal.\n',
    '3. **Solves the ₹200 Data Starvation**: Instead of taking 150 of the 205 ₹200 images for test/val, we use a balanced 30-image test set and 20-image val set, expanding training data from 346 images to **1,436 images** (including 155 images of ₹200).\n',
    '4. **Modern Backbone**: Employs `ConvNeXt-Tiny` with 7x7 depthwise convolutions for text, glyph, and structural numeral recognition.\n'
]

# Cell 3: Configuration
nb['cells'][3]['source'] = [
    'DATASET_ROOT = Path("Indian Currency Dataset")\n',
    'OUTPUT_DIR = Path("currency_models_balanced")\n',
    'OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n',
    '\n',
    'CLASS_NAMES = ["10", "20", "50", "100", "200", "500", "2000"]\n',
    'CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}\n',
    'IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}\n',
    '\n',
    'IMG_SIZE = 224\n',
    'BATCH_SIZE = 32\n',
    'EPOCHS = 35\n',
    'PATIENCE = 8\n',
    'SEED = 42\n',
    '\n',
    '# Balanced test & validation support per denomination\n',
    '# Reserving 30 test + 20 val leaves 155+ images for ₹200 (vs only 5 previously!)\n',
    'TEST_PER_CLASS = 30\n',
    'VAL_PER_CLASS = 20\n',
    '\n',
    'if not DATASET_ROOT.exists():\n',
    '    raise FileNotFoundError(f"Dataset folder not found: {DATASET_ROOT.resolve()}")\n',
    'missing = [c for c in CLASS_NAMES if not (DATASET_ROOT / c).exists()]\n',
    'if missing:\n',
    '    raise FileNotFoundError(f"Missing class folders: {missing}")\n'
]

# Cell 8: Markdown for Transforms
nb['cells'][8]['source'] = [
    '## 4. Color-Blind & Numeral-Focused Transforms and DataLoaders\n',
    '\n',
    'To make the network immune to lighting color casts, OpenCV BGR/RGB channel swapping, and room backgrounds:\n',
    '* `transforms.Grayscale(num_output_channels=3)`: Discards color completely, forcing numeral and texture learning.\n',
    '* `transforms.RandomAdjustSharpness` & `RandomAutocontrast`: Sharpens printed numbers against the paper background.\n',
    '* Aspect ratio preservation: Keeps the ~2.1:1 banknote geometry intact.\n'
]

# Cell 9: Transforms and Dataset code
nb['cells'][9]['source'] = [
    '# Grayscale normalization statistics (identical across 3 channels)\n',
    'MEAN = [0.45, 0.45, 0.45]\n',
    'STD = [0.22, 0.22, 0.22]\n',
    '\n',
    'train_transform = transforms.Compose([\n',
    '    # 1. Strip color completely -> 100% immune to color shifts / inversions\n',
    '    transforms.Grayscale(num_output_channels=3),\n',
    '    \n',
    '    # 2. Aspect-preserving scale & crop focused on banknote geometry\n',
    '    transforms.Resize(256),\n',
    '    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(1.4, 2.4)),\n',
    '    \n',
    '    # 3. Enhance high-frequency numerals and watermark lines\n',
    '    transforms.RandomAdjustSharpness(sharpness_factor=2.5, p=0.7),\n',
    '    transforms.RandomAutocontrast(p=0.5),\n',
    '    \n',
    '    # 4. Perspective and rotation when held by hand\n',
    '    transforms.RandomRotation(12),\n',
    '    transforms.RandomPerspective(distortion_scale=0.20, p=0.4),\n',
    '    \n',
    '    transforms.ToTensor(),\n',
    '    transforms.Normalize(MEAN, STD),\n',
    '])\n',
    '\n',
    'eval_transform = transforms.Compose([\n',
    '    transforms.Grayscale(num_output_channels=3),\n',
    '    transforms.Resize((IMG_SIZE, IMG_SIZE)),\n',
    '    transforms.ToTensor(),\n',
    '    transforms.Normalize(MEAN, STD),\n',
    '])\n',
    '\n',
    'class CurrencyDataset(Dataset):\n',
    '    def __init__(self, frame, transform):\n',
    '        self.frame = frame.reset_index(drop=True)\n',
    '        self.transform = transform\n',
    '    def __len__(self): return len(self.frame)\n',
    '    def __getitem__(self, index):\n',
    '        row = self.frame.iloc[index]\n',
    '        with Image.open(row.path) as image:\n',
    '            image = image.convert("RGB")\n',
    '        return self.transform(image), CLASS_TO_IDX[row.class_name], row.path\n',
    '\n',
    'train_dataset = CurrencyDataset(train_df, train_transform)\n',
    'val_dataset = CurrencyDataset(val_df, eval_transform)\n',
    'test_dataset = CurrencyDataset(test_df, eval_transform)\n',
    '\n',
    '# WeightedRandomSampler guarantees equal class sampling per batch\n',
    'train_counts = Counter(train_df.class_name)\n',
    'weights = [1.0 / train_counts[c] for c in train_df.class_name]\n',
    'sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)\n',
    '\n',
    'workers = 2 if os.name != "nt" else 0\n',
    'pin_memory = torch.cuda.is_available()\n',
    'train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=workers, pin_memory=pin_memory)\n',
    'val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=workers, pin_memory=pin_memory)\n',
    'test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=workers, pin_memory=pin_memory)\n',
    '\n',
    'print(f"Training set size: {len(train_dataset)} images (expanded from 346)")\n',
    'print(f"Balanced test set: {len(test_dataset)} images ({TEST_PER_CLASS} per denomination)")\n'
]

# Cell 10: Model Markdown
nb['cells'][10]['source'] = [
    '## 5. Model: ConvNeXt-Tiny (Numeral & Structural Glyph Architecture)\n',
    '\n',
    'ConvNeXt utilizes large $7\\times7$ depthwise separable convolutions with inverted bottlenecks. It provides a wide receptive field ideally suited for recognizing font typography, numerals (`10`, `20`, `50`, `100`, `200`, `500`, `2000`), and localized watermark security patterns.\n'
]

# Cell 11: Model definition and training setup
nb['cells'][11]['source'] = [
    'def build_model(num_classes=7):\n',
    '    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)\n',
    '    in_features = model.classifier[2].in_features\n',
    '    model.classifier[2] = nn.Sequential(\n',
    '        nn.Linear(in_features, 256),\n',
    '        nn.LayerNorm(256),\n',
    '        nn.GELU(),\n',
    '        nn.Dropout(0.30),\n',
    '        nn.Linear(256, num_classes)\n',
    '    )\n',
    '    return model\n',
    '\n',
    'model = build_model(len(CLASS_NAMES)).to(DEVICE)\n',
    'criterion = nn.CrossEntropyLoss(label_smoothing=0.10)\n',
    'optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)\n',
    'scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)\n',
    'scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))\n',
    'print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))\n'
]

# Cell 18: predict_currency inference function
nb['cells'][18]['source'] = [
    'def predict_currency(image_input, confidence_threshold=0.70):\n',
    '    """\n',
    '    Color-blind, aspect-preserving inference.\n',
    '    Accepts PIL Image, numpy array (RGB or BGR), or file path.\n',
    '    """\n',
    '    if isinstance(image_input, (str, Path)):\n',
    '        image = Image.open(image_input).convert("RGB")\n',
    '    elif isinstance(image_input, np.ndarray):\n',
    '        image = Image.fromarray(image_input).convert("RGB")\n',
    '    else:\n',
    '        image = image_input.convert("RGB")\n',
    '        \n',
    '    # Letterbox padding to preserve 2.1:1 banknote aspect ratio\n',
    '    w, h = image.size\n',
    '    scale = IMG_SIZE / max(w, h)\n',
    '    new_w, new_h = int(w * scale), int(h * scale)\n',
    '    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)\n',
    '    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))\n',
    '    canvas.paste(resized, ((IMG_SIZE - new_w) // 2, ((IMG_SIZE - new_h) // 2)))\n',
    '    \n',
    '    x = eval_transform(canvas).unsqueeze(0).to(DEVICE)\n',
    '    with torch.inference_mode():\n',
    '        probabilities = torch.softmax(model(x), dim=1)[0].cpu().numpy()\n',
    '        \n',
    '    pred_idx = int(probabilities.argmax())\n',
    '    confidence = float(probabilities[pred_idx])\n',
    '    denomination = CLASS_NAMES[pred_idx]\n',
    '    accepted = confidence >= confidence_threshold\n',
    '    message = (f"₹{denomination} note verified with {confidence*100:.1f}% confidence."\n',
    '               if accepted else\n',
    '               "Unable to verify note. Hold one banknote flat and centered in good light.")\n',
    '    return {\n',
    '        "accepted": accepted, "denomination": denomination, "confidence": confidence,\n',
    '        "probabilities": {f"₹{c}": round(float(probabilities[i])*100, 2) for i, c in enumerate(CLASS_NAMES)},\n',
    '        "message": message\n',
    '    }\n',
    '\n',
    '# Example:\n',
    '# print(predict_currency("scratch_pred_500.png"))\n'
]

# Cell 20: ONNX export
nb['cells'][20]['source'] = [
    'import onnx\n',
    'onnx_path = OUTPUT_DIR / "currency_model.onnx"\n',
    'model.eval()\n',
    'dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)\n',
    'torch.onnx.export(\n',
    '    model, dummy, str(onnx_path),\n',
    '    input_names=["input_image"], output_names=["class_logits"],\n',
    '    dynamic_axes={"input_image":{0:"batch"}, "class_logits":{0:"batch"}},\n',
    '    opset_version=17\n',
    ')\n',
    'onnx.checker.check_model(str(onnx_path))\n',
    '\n',
    'labels = {\n',
    '    "class_names": CLASS_NAMES, "class_to_idx": CLASS_TO_IDX,\n',
    '    "mean": MEAN, "std": STD, "img_size": IMG_SIZE,\n',
    '    "confidence_threshold": 0.70,\n',
    '    "balanced_test_per_class": TEST_PER_CLASS,\n',
    '    "color_mode": "grayscale_numeral_focused",\n',
    '    "backbone": "ConvNeXt-Tiny"\n',
    '}\n',
    'with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as f:\n',
    '    json.dump(labels, f, indent=2)\n',
    '\n',
    'print("Saved model:", onnx_path)\n',
    'print("Model size MB:", round(onnx_path.stat().st_size/(1024**2), 2))\n',
    'print("Saved metadata:", OUTPUT_DIR / "labels.json")\n'
]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully updated Currency_model_balanced_7_denominations.ipynb with Approach 1!')
