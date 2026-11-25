"""
VisuNav: Familiar-Person Recognition Pipeline
Assistive Navigation for the Visually Impaired

Tailored for:
- Enrolled contact: Vinay Kasana (friend) from enrollment/Vinay Kasana/
- Test evaluation: test_faces/
- Database template: private_face_db/contacts.json
"""

import json
import time
import queue
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

# ---------------------------------------------------------
# 1. Hardware & Configuration
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[VisuNav] Using hardware device: {DEVICE}")

DB_PATH = Path("private_face_db/contacts.json")
ENROLLMENT_DIR = Path("enrollment")
TEST_FACES_DIR = Path("test_faces")
OUTPUT_DIR = Path("scratch/inference_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Calibrated Hyperparameters for Mobile/Compressed Photos
THRESHOLD = 0.75            # Cosine similarity cutoff for identity verification
MARGIN = 0.08               # Minimum separation between best match and runner-up
MIN_SIZE = 60               # Minimum face width/height in pixels
MIN_BLUR = 20               # Laplacian variance threshold (tolerant of WhatsApp compression)
DETECTION_THRESHOLD = 0.90  # MTCNN detection confidence
MODEL_ID = "inceptionresnetv1-vggface2"

# ---------------------------------------------------------
# 2. Model Initialization
# ---------------------------------------------------------
detector = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=DEVICE)
encoder = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def load_db():
    """Load enrolled contacts database."""
    if not DB_PATH.exists():
        return {"model": MODEL_ID, "contacts": {}}
    db = json.loads(DB_PATH.read_text(encoding="utf-8"))
    if db.get("model") != MODEL_ID:
        raise ValueError("Model mismatch: Please re-enroll contacts.")
    return db


def save_db(db):
    """Save enrolled contacts database safely."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(db, indent=2), encoding="utf-8")
    tmp.replace(DB_PATH)


# ---------------------------------------------------------
# 3. Face Detection & Feature Extraction
# ---------------------------------------------------------
@torch.inference_mode()
def extract_faces(rgb, single_only=False):
    """
    Detects faces, validates image quality, and returns 512-d normalized embeddings.
    Fully compatible with NumPy 2.x and PyTorch.
    """
    image = Image.fromarray(rgb)
    boxes, scores = detector.detect(image)
    if boxes is None or (single_only and len(boxes) != 1):
        return []

    # Explicit cast to float32 to prevent NumPy 2.x ufunc rint errors
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    h, w = rgb.shape[:2]
    accepted = []

    for box, score in zip(boxes, scores):
        if score is None or not np.isfinite(score) or score < DETECTION_THRESHOLD:
            continue
        x1, y1, x2, y2 = np.rint(box.astype(float)).astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if min(x2 - x1, y2 - y1) < MIN_SIZE:
            continue
        crop = rgb[y1:y2, x1:x2]
        blur = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        if blur < MIN_BLUR:
            continue
        accepted.append([x1, y1, x2, y2])

    if not accepted:
        return []

    crops = detector.extract(image, np.asarray(accepted), save_path=None)
    if crops is None:
        return []

    embeddings = F.normalize(encoder(crops.to(DEVICE)), dim=1).cpu().numpy()
    return [{"box": box, "embedding": emb} for box, emb in zip(accepted, embeddings)]


def identify(embedding, db):
    """Matches an embedding against registered templates using cosine similarity."""
    candidates = []
    for key, c in db["contacts"].items():
        if not c.get("consent"):
            continue
        template = np.asarray(c["template"], dtype=np.float32)
        if template.shape != (512,) or not np.isfinite(template).all():
            continue
        template = template / max(np.linalg.norm(template), 1e-12)
        similarity = float(np.dot(embedding, template))
        candidates.append((similarity, key))

    candidates.sort(reverse=True)
    if not candidates:
        return None, None

    best_sim, best_key = candidates[0]
    runner_up = candidates[1][0] if len(candidates) > 1 else -1.0
    is_match = (best_sim >= THRESHOLD and (best_sim - runner_up) >= MARGIN)
    return (best_key if is_match else None), best_sim


# ---------------------------------------------------------
# 4. Enrollment Routine
# ---------------------------------------------------------
def enroll_contact(contact_id, name, relation, folder, consent=False):
    """Enrolls all valid single-face photos from the specified directory."""
    if not consent:
        raise ValueError("Consent required prior to biometric enrollment.")
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    print(f"\n[Enrollment] Scanning {len(paths)} photos in '{folder}' for {name} ({relation})...")

    embeddings = []
    for p in paths:
        try:
            with Image.open(p) as im:
                rgb = np.asarray(im.convert("RGB"))
            faces = extract_faces(rgb, single_only=True)
            if len(faces) == 1:
                embeddings.append(faces[0]["embedding"])
                print(f"  ✓ Accepted: {p.name}")
            else:
                print(f"  ✗ Rejected (single face required or blur/size check failed): {p.name}")
        except Exception as exc:
            print(f"  ✗ Error reading {p.name}: {exc}")

    if len(embeddings) < 3:
        raise ValueError(f"Need at least 3 accepted photos; only {len(embeddings)} passed.")

    master_template = np.mean(embeddings, axis=0)
    master_template /= max(np.linalg.norm(master_template), 1e-12)

    db = load_db()
    db["contacts"][contact_id] = {
        "name": name,
        "relation": relation,
        "consent": True,
        "samples": len(embeddings),
        "template": master_template.tolist(),
    }
    save_db(db)
    print(f"[Enrollment] Successfully enrolled '{name}' with {len(embeddings)} templates in {DB_PATH}!")


# ---------------------------------------------------------
# 5. Inference on Test Faces
# ---------------------------------------------------------
def run_test_inference():
    """Evaluates all images in test_faces/ and saves annotated visualizations."""
    db = load_db()
    if not db.get("contacts"):
        print("[Inference] No contacts registered in database. Enrolling dataset first...")
        enroll_contact("vinay", "Vinay Kasana", "friend", "enrollment/Vinay Kasana", consent=True)
        db = load_db()

    test_files = sorted(list(TEST_FACES_DIR.glob("*.jpeg")) + list(TEST_FACES_DIR.glob("*.jpg")) + list(TEST_FACES_DIR.glob("*.png")))
    if not test_files:
        print(f"[Inference] No test images found in {TEST_FACES_DIR}")
        return

    print(f"\n[Inference] Evaluating {len(test_files)} test images against database:")
    print("=" * 70)

    for img_path in test_files:
        with Image.open(img_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        annotated = rgb.copy()
        faces = extract_faces(rgb)

        print(f"\nImage: {img_path.name} | Detected Faces: {len(faces)}")
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = [int(v) for v in face["box"]]
            key, sim = identify(face["embedding"], db)

            if key:
                c = db["contacts"][key]
                label = f"{c['name']} ({c['relation']}) | {sim:.2f}"
                color = (0, 220, 0) # Green for recognized contact
                status = f"RECOGNIZED -> {c['name']} ({c['relation']})"
            else:
                sim_str = f"{sim:.2f}" if sim is not None else "N/A"
                label = f"Unknown | {sim_str}"
                color = (255, 120, 0) # Orange for unknown person
                status = "UNRECOGNIZED / UNKNOWN"

            print(f"  [Face #{i+1}] {status} | Similarity: {sim:.4f} | BBox: [{x1}, {y1}, {x2}, {y2}]")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
            cv2.putText(annotated, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        save_target = OUTPUT_DIR / img_path.name
        Image.fromarray(annotated).save(save_target)
        print(f"  -> Saved annotated result to: {save_target}")

    print("=" * 70)
    print(f"[Inference Complete] All annotated results stored in '{OUTPUT_DIR}'")


# ---------------------------------------------------------
# 6. ONNX Model Export & ONNX Runtime Inference
# ---------------------------------------------------------
def export_to_onnx(output_path="weights/facenet_inception_resnet_v1.onnx"):
    """Exports the InceptionResnetV1 embedding model to ONNX format."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 160, 160, device=DEVICE)
    print(f"\n[ONNX] Exporting InceptionResnetV1 model to: {output_path} ...")
    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_file),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        dynamo=False,
    )
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"[ONNX] Export Successful! File saved: {output_path} ({size_mb:.2f} MB)")


def run_onnx_inference(onnx_path="weights/facenet_inception_resnet_v1.onnx"):
    """Runs inference on test_faces using ONNX Runtime."""
    import onnxruntime as ort

    db = load_db()
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    test_files = sorted(list(TEST_FACES_DIR.glob("*.jpeg")) + list(TEST_FACES_DIR.glob("*.jpg")) + list(TEST_FACES_DIR.glob("*.png")))
    print(f"\n[ONNX Runtime] Running inference on {len(test_files)} test images:")
    for img_path in test_files:
        with Image.open(img_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        image = Image.fromarray(rgb)
        boxes, _ = detector.detect(image)
        if boxes is None:
            continue
        boxes = np.asarray(boxes, dtype=np.float32)
        accepted = [np.rint(b.astype(float)).astype(int).tolist() for b in boxes if b is not None]
        if not accepted:
            continue
        crops = detector.extract(image, np.asarray(accepted), save_path=None)
        if crops is None:
            continue

        crops_np = crops.numpy().astype(np.float32)
        raw_embs = session.run([output_name], {input_name: crops_np})[0]
        # L2 normalize
        embs = raw_embs / np.linalg.norm(raw_embs, axis=1, keepdims=True)

        print(f"\n  Image: {img_path.name}")
        for i, emb in enumerate(embs):
            key, score = identify(emb, db)
            name = db["contacts"][key]["name"] if key else "Unrecognized / Unknown"
            relation = f"({db['contacts'][key]['relation']})" if key else ""
            print(f"    [Face #{i+1}] {name} {relation} | Similarity: {score:.4f}")


# ---------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Export model to ONNX format
    export_to_onnx("weights/facenet_inception_resnet_v1.onnx")

    # Step 2: Enroll dataset
    enroll_contact(
        contact_id="vinay",
        name="Vinay Kasana",
        relation="friend",
        folder="enrollment/Vinay Kasana",
        consent=True,
    )

    # Step 3: Run inference on test faces (PyTorch)
    run_test_inference()

    # Step 4: Run inference with ONNX Runtime
    run_onnx_inference("weights/facenet_inception_resnet_v1.onnx")

