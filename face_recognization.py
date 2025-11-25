"""
face_recognition_realtime_fixed.py
Option C: FaceNet-style embeddings (facenet-pytorch) + SVM.
Press 's' to add a new person (type name). Program captures 30 face images,
re-trains SVM, and continues real-time recognition.
"""

import os
import time
import pickle
import logging
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# -------------------------
# CONFIG
# -------------------------
DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
IMG_PER_PERSON = 30
CAPTURE_INTERVAL = 0.2   # seconds between saved images while capturing
EMBEDDING_BATCH = 32     # process embeddings in batches (if many images)
UNKNOWN_THRESHOLD = 0.7  # lower => stricter unknown detection (adjust experimentally)

SVM_PATH = MODELS_DIR / "svm_classifier.pkl"
LE_PATH = MODELS_DIR / "label_encoder.pkl"

# -------------------------
# Setup logging
# -------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------------------------
# Initialize models (MTCNN + FaceNet)
# -------------------------
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
except ImportError:
    device = "cpu"
    logger.warning("PyTorch not available, defaulting device to CPU")

mtcnn = MTCNN(image_size=160, margin=14, keep_all=True, device=device)  # returns cropped 160x160 faces
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # outputs 512-d embeddings

# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
    except Exception as e:
        logger.error(f"Failed to ensure directory {path}: {e}")
  
def save_pickle(obj, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.debug(f"Saved pickle to {path}")
    except Exception as e:
        logger.error(f"Failed to save pickle to {path}: {e}")

def load_pickle(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.debug(f"Loaded pickle from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load pickle from {path}: {e}")
        return None

def list_images_for_dataset(dataset_dir=DATASET_DIR):
    pairs = []
    try:
        for person_dir in sorted(dataset_dir.iterdir()):
            if not person_dir.is_dir(): 
                continue
            for img_path in person_dir.glob("*.jpg"):
                pairs.append((img_path, person_dir.name))
        logger.debug(f"Collected {len(pairs)} image paths from dataset")
    except Exception as e:
        logger.error(f"Error listing images for dataset: {e}")
    return pairs

# -------------------------
# Capture face images for new person
# -------------------------
def capture_new_person(name, cap):
    person_dir = DATASET_DIR / name
    ensure_dir(person_dir)
    logger.info(f"[+] Capturing images for '{name}' into {person_dir} ...")
    count = 0
    last_saved = 0.0
    start = time.time()
    while count < IMG_PER_PERSON:
        ret, frame = cap.read()
        if not ret:
            logger.warning("[-] Failed to read from camera.")
            break
        display = frame.copy()
        cv2.putText(display, f"Capturing {name}: {count}/{IMG_PER_PERSON}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Capture (Press 'q' to cancel)", display)

        try:
            boxes, probs = mtcnn.detect(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            boxes = None

        if boxes is not None and len(boxes) > 0:
            boxes = np.array(boxes)
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            best_idx = int(np.argmax(areas))
            box_coords = [int(v) for v in boxes[best_idx]]

            now = time.time()
            if now - last_saved >= CAPTURE_INTERVAL:
                try:
                    # Crop the best face from the original frame using the detected box
                    x1, y1, x2, y2 = box_coords
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    face_crop = frame[y1c:y2c, x1c:x2c]
                    if face_crop is None or face_crop.size == 0:
                        # nothing to save for this detection
                        continue
                    # Convert to PIL, resize to model input size and save as JPEG
                    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).resize((160, 160))
                    fname = person_dir / f"{name}_{int(time.time()*1000)}_{count}.jpg"
                    face_pil.save(str(fname))
                    count += 1
                    last_saved = now
                except Exception as e:
                    logger.error(f"Error extracting or saving face image: {e}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("[!] Capture cancelled by user.")
            break
    elapsed = time.time() - start
    logger.info(f"[+] Done capturing {count} images in {elapsed:.1f}s.")
    cv2.destroyWindow("Capture (Press 'q' to cancel)")

# -------------------------
# Build embeddings from dataset
# -------------------------
def build_embeddings():
    pairs = list_images_for_dataset()
    if not pairs:
        logger.warning("[!] No images in dataset. Add someone first by pressing 's'.")
        return None, None
    X = []
    y = []
    paths = [p for p, _ in pairs]
    labels = [lbl for _, lbl in pairs]

    batch_faces = []
    batch_labels = []
    for p, lbl in zip(paths, labels):
        try:
            img = Image.open(p).convert('RGB')
            img = img.resize((160,160))
            batch_faces.append(np.asarray(img))
            batch_labels.append(lbl)
            if len(batch_faces) >= EMBEDDING_BATCH:
                emb = faces_to_embeddings(batch_faces)
                X.append(emb)
                y.extend(batch_labels)
                batch_faces, batch_labels = [], []
        except Exception as e:
            logger.error(f"Failed processing image {p}: {e}")
    if batch_faces:
        emb = faces_to_embeddings(batch_faces)
        X.append(emb)
        y.extend(batch_labels)

    try:
        X = np.vstack(X)  # shape N x 512
        y = np.array(y)
        logger.info(f"[+] Built embeddings for {len(y)} images (classes: {len(set(y))}).")
        return X, y
    except Exception as e:
        logger.error(f"Failed to stack embeddings: {e}")
        return None, None

def faces_to_embeddings(faces_list):
    import torch
    try:
        faces = np.stack([np.asarray(f).astype(np.float32) for f in faces_list], axis=0)
        faces = (faces / 255.0 - 0.5) / 0.5
        faces_t = torch.tensor(faces).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            embs = resnet(faces_t).cpu().numpy()
        return embs
    except Exception as e:
        logger.error(f"Failed to compute embeddings: {e}")
        return np.empty((0,512))

# -------------------------
# Train SVM classifier
# -------------------------
def train_and_save_classifier():
    X, y = build_embeddings()
    if X is None or len(set(y)) < 2:
        logger.warning("[!] Need at least 2 classes to train SVM. Add more people first.")
        return False
    try:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X, y_enc)
        save_pickle(clf, SVM_PATH)
        save_pickle(le, LE_PATH)
        logger.info(f"[+] Trained SVM and saved to {SVM_PATH}, labels to {LE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to train and save classifier: {e}")
        return False

# -------------------------
# Real-time recognition loop
# -------------------------
def recognize_realtime():
    clf = None
    le = None
    if SVM_PATH.exists() and LE_PATH.exists():
        clf = load_pickle(SVM_PATH)
        le = load_pickle(LE_PATH)
        if clf is None or le is None:
            logger.warning("[!] Failed to load existing classifier or label encoder. Starting fresh.")
            clf, le = None, None
        else:
            logger.info("[+] Loaded existing classifier.")
    else:
        logger.info("[!] No classifier found. Add a person (press 's') to create one.")

    # Attempt opening camera indices 0 and 1 for flexibility
    cap = None
    for cam_idx in [0, 1]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            logger.info(f"Camera opened successfully at index {cam_idx}")
            break
        else:
            logger.warning(f"Failed to open camera at index {cam_idx}")
            cap.release()
            cap = None
    if cap is None:
        logger.error("[-] Cannot open any camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera. Exiting.")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            boxes, probs = mtcnn.detect(Image.fromarray(rgb))
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            boxes = None

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
                face_img = frame[y1c:y2c, x1c:x2c]
                if face_img.size == 0:
                    continue
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).resize((160,160))
                emb = faces_to_embeddings([np.asarray(face_pil)])
                name = "Unknown"

                if clf is not None and le is not None:
                    try:
                        probs_arr = clf.predict_proba(emb)[0]
                        best_idx = np.argmax(probs_arr)
                        best_prob = probs_arr[best_idx]
                        if best_prob >= UNKNOWN_THRESHOLD:
                            name = le.inverse_transform([best_idx])[0]
                        else:
                            name = "Unknown"
                    except Exception as e:
                        logger.error(f"Error in SVM prediction: {e}")
                        name = "Unknown"

                color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)
                cv2.putText(frame, f"{name}", (x1c, y1c - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition (press 's' to add person, 'q' to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Quit signal received. Exiting.")
            break
        elif key == ord('s'):
            # To avoid blocking input freezing the GUI window, use OpenCV window input dialog replacement
            cv2.putText(frame, "Enter name then press Enter in console", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Face Recognition (press 's' to add person, 'q' to quit)", frame)
            cv2.waitKey(1)
            try:
                name = input("Enter name for new person (no spaces): ").strip()
                if name == "":
                    logger.warning("[-] Invalid name, skipping.")
                    continue
                capture_new_person(name, cap)
                logger.info("[*] Retraining classifier (this may take a moment)...")
                ok = train_and_save_classifier()
                if ok:
                    clf = load_pickle(SVM_PATH)
                    le = load_pickle(LE_PATH)
                    if clf is None or le is None:
                        logger.error("Failed to reload classifier after training")
            except (EOFError, KeyboardInterrupt):
                logger.warning("Input cancelled during name entry. Skipping adding person.")
                continue
            except Exception as e:
                logger.error(f"Unexpected error during adding person: {e}")

    cap.release()
    cv2.destroyAllWindows()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    ensure_dir(DATASET_DIR)
    try:
        recognize_realtime()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
