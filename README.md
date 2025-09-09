# ❤️ Heart Disease Prediction Project

## 📌 Overview
This project predicts the likelihood of heart disease using machine learning models trained on the **Cleveland Heart Disease dataset**.  
It covers data preprocessing, dimensionality reduction (PCA), feature selection, model training, evaluation, and deployment via a **Streamlit web app**.  
A public link can be created using **Ngrok** for easy sharing.

---

## 🚀 Features
✅ Data Cleaning & Preprocessing (handling missing values, encoding, scaling)  
✅ Exploratory Data Analysis (EDA) with histograms, boxplots, and heatmaps  
✅ Dimensionality Reduction using PCA  
✅ Supervised Learning Models: Logistic Regression, Random Forest, SVM, Naive Bayes, KNN  
✅ Unsupervised Learning Models: K-Means & Hierarchical Clustering  
✅ Model Evaluation: Accuracy, Precision, Recall, F1-Score, ROC & AUC  
✅ Hyperparameter Tuning (GridSearchCV & RandomizedSearchCV)  
✅ Model Export using `joblib` (.pkl file)  
✅ Streamlit Web App for real-time prediction  
✅ Ngrok Integration for public access  

---

## 📊 Dataset
- **Source:** Cleveland Heart Disease Dataset (UCI ML Repository)  
- **Rows:** 303  
- **Features:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`  
- **Target:** `num` → converted to **binary target**  
  - `1` → Heart Disease Present  
  - `0` → No Heart Disease  

---

## 🏗 Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib  
- **UI:** Streamlit  
- **Deployment:** Pyngrok (for tunneling), GitHub  

---

## 📂 Project Structure
Heart_Disease_Project/
├── data/
│ └── heart_disease.csv
├── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ └── 06_hyperparameter_tuning.ipynb
├── models/
│ └── final_model.pkl
├── ui/
│ └── app.py
├── deployment/
│ └── ngrok_tunnel.py
├── results/
│ └── evaluation_metrics.txt
├── requirements.txt
├── README.md
└── .gitignore



---

## ▶ How to Run the App

1️⃣ **Clone this repository**
```bash
git clone <your-repo-url>
cd Heart_Disease_Project

pip install -r requirements.txt
python src/full_pipeline.py
streamlit run ui/app.py
ngrok authtoken <YOUR_NGROK_AUTHTOKEN>
python deployment/ngrok_tunnel.py

📊 Evaluation Metrics
The file results/evaluation_metrics.txt contains:
Accuracy, Precision, Recall, F1-score
ROC & AUC values
Confusion Matrix
