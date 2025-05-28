# 🧠 Fabric Defect Detection – AI Model Training

This repository contains all training experiments and model development pipelines used for the **AI-powered fabric defect detection system**. We explore various CNN architectures and transformer models to classify fabric defects such as **holes, stains, horizental, vertical, Thread, defect-free**. The trained models are later deployed in the [backend API repo](https://github.com/skukreja123/FYP-defect-dection-backend) of the real-time defect detection system.

> 📍 GitHub Link: [skukreja123/FYP-defect-detection-AI](https://github.com/skukreja123/FYP-defect-detection-AI)

---

## 📂 Datasets Used

1. **[TILDA Fabric Dataset – TILDA Image Patches](https://www.kaggle.com/datasets/angelolmg/tilda-400-64x64-patches)**
2. **[Fabric Defect Dataset – Kaggle](https://www.kaggle.com/datasets/k213080/fabric-defect-dataset)**
3. **Mixed Dataset** combining multiple real-world and structured datasets for improved generalization.



## 🧪 Initial Experiments – Classical Models & Binary/Ternary Patterns

Before moving to advanced deep learning architectures like ResNet, EfficientNet, YOLO, and Vision Transformers, we began our model development phase with foundational CNN architectures and basic pattern-based classification techniques. This helped us understand the dataset distribution, feature behavior, and model learning limitations.

## ✅ Models & Techniques Explored:
  Simple CNN (Convolutional Neural Network):
  
  Built from scratch with 2–4 convolutional layers.
  
  Trained on grayscale and RGB fabric samples.

## Binary Pattern Classification:
  
  Applied thresholding and edge detection to extract binary textures.
  
  Evaluated handcrafted features for defect detection.

## Ternary Pattern Analysis:

  Explored local ternary patterns (LTP) for enhanced texture robustness.
  
  Used them with shallow CNNs and traditional classifiers (SVM, KNN).

## 🔍 Observations:
  These models performed reasonably well on simple defect categories but struggled with:
  
  Complex textures
  
  Lighting variations
  
  Small or overlapping defects
  
  Accuracy plateaued after a certain epoch even with hyperparameter tuning.
  
  Helped define our baseline and justify the need for deeper, pretrained models.


---

## 🎯 Use Cases & Training Strategies

We experimented with **three different use case scenarios** to identify the best-performing training pipeline.

---

### ✅ Use Case 1 – Baseline Training

- No **data augmentation**, no **image generation**, no **oversampling**
- Loss Function: **Weighted Categorical Cross Entropy**
- Models Used:
  - ✅ EfficientNet
  - ✅ ResNet
  - ✅ VGG
  - ✅ YOLO
  - ✅ Vision Transformer (ViT)
- Evaluation:
  - Confusion Matrix
  - Classification Report
  - Training & Testing Accuracy/Loss Graphs
  - ROC/AUC Curve

---

### ✅ Use Case 2 – Oversampling Strategy

- Oversampling of minority defect classes using data replication or SMOTE
- Suitable class-balanced **loss functions** (e.g., Focal Loss or Balanced CE)
- Models Used:
  - ✅ EfficientNet
  - ✅ ResNet
  - ✅ VGG
  - ✅ YOLO
  - ✅ Vision Transformer (ViT)
- Evaluation:
  - Confusion Matrix
  - Classification Report
  - Training & Testing Accuracy/Loss Graphs
  - ROC/AUC Curve

---

### ✅ Use Case 3 – Full Augmentation Pipeline

- **Oversampling** + **Data Augmentation** (rotation, scaling, flipping, brightness adjustment)
- Robust **loss functions** tuned for noisy, imbalanced datasets
- Models Used:
  - ✅ EfficientNet
  - ✅ ResNet
  - ✅ VGG
  - ✅ YOLO
  - ✅ Vision Transformer (ViT)
- Evaluation:
  - Confusion Matrix
  - Classification Report
  - Training & Testing Accuracy/Loss Graphs
  - ROC/AUC Curve

---

## 📊 Visual Evaluation & Metrics

Each model and use case was evaluated using:

- 📌 **Confusion Matrix**  
- 📌 **Classification Report (Precision, Recall, F1-score)**
- 📈 **Accuracy & Loss Curves** for both training and validation
- 🧪 **ROC/AUC Curves** for multi-class evaluation

---

## 🛠 Technologies & Libraries

| Area              | Tools / Frameworks                              |
|-------------------|--------------------------------------------------|
| Deep Learning     | PyTorch, Torchvision, TensorFlow (ViT)           |
| Image Processing  | OpenCV, Albumentations, PIL                      |
| Models Used       | EfficientNet, ResNet, VGG, YOLO, Vision Transformer |
| Evaluation        | scikit-learn, matplotlib, seaborn                |
| Data Handling     | Pandas, NumPy, Imbalanced-learn                  |
| Dev Tools         | Jupyter Notebook, Google Colab, Git              |




---

## 🤝 Contributors

  Sahil Kukreja – Developer, Model Trainer, Backend Engineer
   [GitHub Profile](https://github.com/skukreja123)

Areeb – Developer, Model Trainer, Backend Engineer
 [GitHub Profile](https://github.com/areebbinnadeem)

Mustafa – Developer, Model Trainer, Backend Engineer
 

---

## 🔗 Related Projects

- 🔙 **[Backend API for Prediction](https://github.com/skukreja123/FYP-defect-dection-backend)**
- ⚛️ **[Frontend React App](https://github.com/skukreja123/FYP-defect-detection-frontend)*




