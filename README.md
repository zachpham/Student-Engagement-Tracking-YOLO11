# 🧠 Student Engagement Detection using Deep Learning

## 📌 Project Overview
This project focuses on building a **Deep Learning-based system** to detect and classify **student engagement levels** in a classroom environment. The system analyzes visual data (images/video frames) to determine whether a student is focused or distracted.

---

## 📊 Dataset Specifications
- Source: Student Concentration Image Dataset (Kaggle)
- Scale: ~2,120 images
- Classes: Focused, Confused, Frustrated, Bored, Drowsy, Looking Away
- Split: 70% Train / 20% Val / 10% Test

---

## 🏗️ Model Architecture
- YOLO11-cls (lightweight classification model)
- Optimized for real-time performance

---

## ⚙️ Pipeline
1. Data preprocessing
2. Model training
3. Evaluation
4. Real-time deployment

---

## 🎯 Two-Stage System
- Stage 1: Person Detection (YOLO)
- Stage 2: Behavior Classification

---

## 🚀 Training Config
- Epochs: 50
- Batch size: 32
- Image size: 224x224

---

## 📈 Results
- Top-1 Accuracy: ~97–98%
- Top-5 Accuracy: ~100%

---

## ⚠️ Limitations
- No temporal tracking
- Sensitive to lighting & occlusion

---

## 🔮 Future Work
- Add LSTM/Transformer
- Improve dataset diversity
