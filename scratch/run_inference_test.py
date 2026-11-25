import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DB_PATH = Path('private_face_db/contacts.json')
THRESHOLD = 0.75
MARGIN = 0.08
MIN_SIZE = 60
MIN_BLUR = 20
DETECTION_THRESHOLD = 0.90
MODEL_ID = 'inceptionresnetv1-vggface2'

detector = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=DEVICE)
encoder = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

def load_db():
    if not DB_PATH.exists():
        return {'model': MODEL_ID, 'contacts': {}}
    return json.loads(DB_PATH.read_text(encoding='utf-8'))

db = load_db()

@torch.inference_mode()
def extract_faces(rgb):
    image = Image.fromarray(rgb)
    boxes, scores = detector.detect(image)
    if boxes is None:
        return []
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
    emb = F.normalize(encoder(crops.to(DEVICE)), dim=1).cpu().numpy()
    return [{'box': box, 'embedding': e} for box, e in zip(accepted, emb)]

def identify(embedding, db):
    candidates = []
    for key, c in db['contacts'].items():
        if not c.get('consent'):
            continue
        template = np.asarray(c['template'], dtype=np.float32)
        template = template / max(np.linalg.norm(template), 1e-12)
        sim = float(np.dot(embedding, template))
        candidates.append((sim, key))
    candidates.sort(reverse=True)
    if not candidates:
        return None, None
    best, key = candidates[0]
    runner = candidates[1][0] if len(candidates) > 1 else -1
    return (key if best >= THRESHOLD and best - runner >= MARGIN else None), best

output_dir = Path('scratch/inference_results')
output_dir.mkdir(parents=True, exist_ok=True)

test_images = sorted(list(Path('test_faces').glob('*.jpeg')) + list(Path('test_faces').glob('*.jpg')) + list(Path('test_faces').glob('*.png')))
print(f"Found {len(test_images)} test images in test_faces/")

for img_path in test_images:
    with Image.open(img_path) as im:
        rgb = np.asarray(im.convert('RGB'))
    annotated = rgb.copy()
    faces = extract_faces(rgb)
    print(f"\nImage: {img_path.name}")
    print(f"  Faces detected: {len(faces)}")
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = [int(v) for v in face['box']]
        key, score = identify(face['embedding'], db)
        if key:
            c = db['contacts'][key]
            label = f"{c['name']} ({c['relation']}) | sim: {score:.2f}"
            color = (0, 220, 0) # Green for recognized
            print(f"  [Face {i+1}] RECOGNIZED: {c['name']} ({c['relation']}) | Cosine Similarity: {score:.4f}")
        else:
            sim_str = f"{score:.2f}" if score is not None else "N/A"
            label = f"Unrecognized | sim: {sim_str}"
            color = (255, 100, 0) # Orange for unrecognized
            print(f"  [Face {i+1}] UNRECOGNIZED: Cosine Similarity: {score:.4f} (Below threshold {THRESHOLD})")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    save_path = output_dir / img_path.name
    Image.fromarray(annotated).save(save_path)
    print(f"  Saved annotated result -> {save_path}")
