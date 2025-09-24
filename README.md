# Federated_Learning_HeartDiseasePrediction
Hybrid Federated Learning vs Centralized Learning (With and Without SMOTE)
📌 Overview

This project compares Hybrid Federated Learning (FL) and Centralized Deep Learning (DL) models for Heart Disease Prediction.
To address class imbalance, we evaluate models both with and without SMOTE (Synthetic Minority Over-sampling Technique).

The project also visualizes results and compares performance across accuracy, precision, recall, F1-score, and loss.

📊 Dataset

Dataset Used: Framingham Heart Study Dataset
Target Variable: TenYearCHD (0 → No CHD, 1 → CHD within 10 years).

Preprocessing Steps:

Missing values filled with column mean.

Standard scaling applied to features.

Train-test split (80-20).

SMOTE applied (in second experiment) to handle class imbalance.
Methodology

Hybrid Federated Learning (FL)

3 clients created by splitting dataset.

Federated averaging (FedAvg) algorithm used.

Clients sampled per round (partial participation).

Centralized Deep Learning (DL)

Traditional training using combined dataset.

SMOTE (Class Imbalance Handling)

Applied to oversample minority class before training.

Both FL and DL evaluated with and without SMOTE.

🏗 Model Architecture

Dense Neural Network:

Dense(64, ReLU) → BatchNorm → Dropout(0.2)

Dense(32, ReLU) → BatchNorm → Dropout(0.2)

Dense(16, ReLU) → BatchNorm

Dense(1, Sigmoid)

📈 Results
Metrics Compared:

Accuracy

Precision

Recall

F1-Score

Loss

📊 Bar Chart: Hybrid FL vs Centralized (With & Without SMOTE)

Installation & Usage
1️⃣ Clone Repository
git clone https://github.com/yoursai-2005-ai/federated-learning-comparison.git
cd federated-learning-comparison
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Script
python scripts/fl_vs_centralized_with_smote.py
4️⃣ Run in Jupyter/Colab
notebooks/hybrid_federated_learning_smote.ipynb
Requirements
Example requirements.txt:
tensorflow==2.17.0
tensorflow-federated==0.71.0
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
scipy==1.10.1

Future Work
Compare with traditional ML models (SVM, Random Forest, Logistic Regression).
Test with different FL algorithms (FedProx, FedNova, etc.).
Extend to larger medical datasets.

Authors
B. Sai Sailu and others
B.Tech CSE (AI & ML)
