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
- `yolov8n.pt`
- `best.onnx`
- `best (1).onnx`

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

A dedicated currency recognition module is developed to recognize Indian currency notes and identify their denominations.

**Files:**
- `Currency_model.ipynb`
- `best_currency_model.pth`
- `currency_model.onnx`
- `currency_models/`

**Purpose:**
- Detect Indian currency notes.
- Identify the denomination of the detected note.
- Provide the recognized currency information to the user.

The currency recognition capability is intended to assist users in identifying currency independently.

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