# Assistive-Navigation-for-the-Visually-Impaired
This project proposes an AI-powered assistive navigation system that uses computer vision, 
artificial intelligence, and spatial audio to provide real-time environmental awareness and 
guidance to visually impaired users. 

The system will use a camera-based vision module mounted on a wearable device or 
connected to a smartphone/edge-computing device. The captured visual information will be 
processed using AI models to detect objects, estimate their relative distance and direction, 
recognize important visual information, and identify potential hazards. 

### Key Objectives
1.  Real-Time Obstacle and Distance Estimation
2.  Intelligent Path and Hazard Assessment
3.  Text and Currency Reading
4.  Facial Recognition
5.  Dynamic Obstacle Awareness
6.  Prioritized Spatial Audio Feedback 


## AI Models Developed

The current project repository contains multiple AI-based models/modules that form the core perception capabilities of the system.

### 1. Object Detection Model

The Object Detection module is responsible for identifying objects present in the user's surrounding environment.

**Technology:** YOLOv8

**Files:**
- `Object_detection model.ipynb`
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolo11n.pt`
- `pi_deploy/models/yolo_obstacles.onnx`

**Purpose:**
- Detect objects from camera/image input.
- Identify the class of detected objects.
- Determine the location of objects within the image.
- Provide the primary environmental perception capability of the system.

The YOLO-based object detection model forms the foundation of the navigation system by helping the system understand what objects are present around the user.

---

### 2. OCR Model

The OCR module is responsible for extracting and recognizing text from images.

**Implementation:**
- `OCR_model.ipynb`

**Purpose:**
- Detect text from captured images.
- Extract readable information from signs, labels, documents, and other visual text.
- Provide text information that can later be converted into audio feedback.

This module helps make printed and displayed textual information more accessible to visually impaired users.

---

### 3. Currency Recognition Model

A dedicated two-stage hybrid currency recognition module is developed to recognize Indian currency notes across 7 denominations (Rs. 10, 20, 50, 100, 200, 500, 2000) using orientation-aware OCR numeral spotting and a ConvNeXt-Tiny vision fallback.

**Files:**
- `Currency_model_balanced_7_denominations.ipynb`
- `train_numeral_currency_model.py`
- `test_hybrid_currency.py` (supports `--image <path>` and `--dataset-dir <dir>` CLI options)
- `currency_models_balanced/`
- `pi_deploy/models/currency_model.onnx`

**Purpose:**
- Detect Indian currency notes.
- Identify the denomination of the detected note.
- Provide the recognized currency information to the user.

The currency recognition capability is intended to assist users in identifying currency independently.

---

### 4. Familiar Person Recognition Model

The Familiar Person Recognition module detects faces and matches them against locally enrolled contacts with consent-gated embeddings.

**Technology:** MTCNN + InceptionResnetV1 (VGGFace2)

**Files:**
- `VisuNav_Familiar_Person_Recognition.ipynb`
- `run_familiar_person_recognition.py`
- `private_face_db/contacts.json`
- `pi_deploy/models/facenet_inception_resnet_v1.onnx`

**Purpose:**
- Detect faces and validate image quality (size and blur checks).
- Match faces against enrolled contacts using cosine similarity.
- Support ONNX Runtime inference for edge deployment.

---

### 5. Edge Deployment & Evaluation

- `pi_deploy/pi_assistive_nav.py` — Unified edge deployment entrypoint with ONNX models in `pi_deploy/models/`.
- `MODEL_EVALUATION_RESULTS.md` — Model evaluation metrics and test summaries.

---

## 🔄 Project Flow

The overall system follows a **Perception → Processing → Decision → Feedback** pipeline.

```text
                    ┌─────────────────────┐
                    │    Camera Input     │
                    │  / Video Stream     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Image Capture &   │
                    │  Pre-processing     │
                    │      (OpenCV)       │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │       AI Perception        │
                 └──────────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │    Object    │  │     OCR      │  │   Currency   │
      │   Detection  │  │    Model     │  │ Recognition  │
      │   YOLOv8     │  │              │  │    Model     │
      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Information &       │
                    │ Situation Analysis  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Decision &          │
                    │ Prioritization      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Audio / Voice       │
                    │ Feedback            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       User          │
                    │  Environmental      │
                    │     Awareness       │
                    └─────────────────────┘
```