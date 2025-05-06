# DiabPredict+ : DeepLearningEnhancedDiagnosis
DiabPredict+ is a smart healthcare project that uses deep learning techniques to predict diabetes with high accuracy. Combining autoencoders for feature extraction and an artificial neural network (ANN) for classification, this project showcases how modern machine learning can empower early-stage medical diagnostics.

## 🚀 Overview
This project was built using the Pima Indians Diabetes Dataset, which includes key medical indicators such as glucose levels, BMI, insulin, and age. It demonstrates how dimensionality reduction via autoencoders can improve model performance by highlighting only the most relevant patterns.

### 🔧 Technologies Used: Python, TensorFlow, Keras, Scikit-learn, Matplotlib, Seaborn

## 🎯 Objective
To build an accurate and efficient deep learning model that can:

Detect whether a patient is likely to have diabetes based on diagnostic features.

Showcase use of unsupervised deep learning (autoencoders) to improve feature quality.

Provide an end-to-end ML pipeline, from data exploration to prediction and evaluation.

## 📊 Dataset
🗂 Source: Pima Indians Diabetes Database (Kaggle)
📌 Features:

Glucose, BMI, Blood Pressure, Insulin, Skin Thickness, Pregnancies, Age

Target: Outcome (1 = Diabetic, 0 = Non-Diabetic)

## 🧠 Architecture
Raw Data
   ↓
Preprocessing & Scaling
   ↓
Autoencoder (for feature extraction)
   ↓
Encoded Features
   ↓
Artificial Neural Network (ANN)
   ↓
Prediction + Evaluation

## 📌 Key Highlights
🔍 Feature Extraction with Autoencoder: Dimensionality reduction to focus on the most meaningful signals.

🧠 Deep ANN Classifier: A two-layer ANN with dropout and batch normalization.

📈 Evaluation Metrics: Accuracy, Classification Report, Confusion Matrix.

🧪 Data Visualization: Heatmaps and countplots for correlation and class distribution.

## 📷 Sample Visualizations
### 🔥 Correlation Matrix
Helps understand feature relationships before training.

### 🌀 Confusion Matrix
Shows model performance visually.

## 📦 How to Run
### 1. Clone the repository
git clone https://github.com/your-username/DiabPredict-Deep-Learning-Diagnosis.git
cd DiabPredict-Deep-Learning-Diagnosis

### 2. Install dependencies
pip install -r requirements.txt

### 3. Launch notebook
jupyter notebook DiabPredict_Pipeline.ipynb

## 📈 Performance
| Metric     | Value                        |
| ---------- | ---------------------------- |
| Accuracy   | \~85%                        |
| Precision  | High for diabetic class      |
| Recall     | Balanced across both classes |
| Model Size | Lightweight and fast         |

## 👨‍💻 About Me
Savee Gupta
🎓 M.Tech in Computer Science (AI) — NSUT
💼 Passionate about solving real-world problems using AI
📬 https://www.linkedin.com/in/savee-gupta-9b85991ab/ | https://github.com/saveegupta

## 💬 HR / Recruiter Note
This project highlights my practical experience in:

Applying deep learning to real-world health problems

Building complete ML pipelines from scratch

Working with unsupervised (Autoencoders) and supervised (ANN) models

Visualizing, analyzing, and validating ML model performance

I’m open to internship/full-time roles in Data Science, AI, or ML Engineering.

## ⭐ Support This Project
If you found this helpful:

🌟 Star this repo

🍴 Fork it for your version

📢 Share with peers












