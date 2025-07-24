# 🧠 Credit Risk Analysis using Machine Learning

This project presents a complete end-to-end Credit Risk Prediction pipeline. It handles data preprocessing, EDA, feature engineering using WOE/IV, SMOTE for imbalance correction, multiple machine learning models, and a production-ready **Streamlit dashboard** for real-time user input and scoring.

---

## 🚀 Features

- 📊 Exploratory Data Analysis (EDA) & Outlier Detection
- 🔍 IV/WOE-based Feature Engineering
- ⚖️ SMOTE Oversampling for Imbalanced Dataset
- 🧪 Trained ML Models: 
  - Logistic Regression (with and without WOE)
  - Decision Tree
  - Random Forest
  - XGBoost, LightGBM, CatBoost
  - K-Nearest Neighbors (KNN)
- 📈 Model Evaluation Metrics: ROC AUC, Accuracy, F1 Score, Confusion Matrix
- 🧾 Scorecard Generation with `scorecardpy`
- 🧪 Saved Models using `joblib`
- 🖥️ Real-Time Prediction via **Streamlit UI**

---

## 📂 Project Structure

credit-risk-analysis-using-ML/
├── dataset/
│ └── train.csv
│
├── models/
│ ├── logistic_model.pkl
│ ├── xgb_model.pkl
│ ├── catboost_model.pkl
│ ├── decision_tree_model.pkl
│ ├── random_forest_model.pkl
│ ├── knn_model.pkl
│ ├── lgbm_model.pkl
│ ├── logistic_model_woe.pkl
│ ├── scaler.pkl
│ ├── scaler_features.pkl
│ └── scorecard_bins.pkl
│
├── app.py # Streamlit Dashboard
├── model_training.py # Main ML pipeline and EDA
├── requirements.txt
└── README.md


---

## 🧠 How the ML Pipeline Works

1. **EDA & Cleaning**  
   Handling missing values, treating anomalies, and visualizing default ratios.

2. **Feature Selection via Information Value (IV)**  
   Low IV (< 0.02) features dropped, and monotonic binning (WOE) applied.

3. **WOE Binning & Scorecard Generation**  
   Used `scorecardpy` to calculate WOE/IV and derive a credit scorecard.

4. **Train/Test Split + SMOTE**  
   Handled class imbalance using SMOTE and standardized features with `MinMaxScaler`.

5. **Model Training**  
   Trained multiple classifiers and evaluated using `accuracy`, `f1_score`, and `roc_auc_score`.

6. **Streamlit App Deployment**  
   Takes user input and sends it to the saved models for real-time prediction & score display.

---

## 🧪 To Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/credit-risk.git
cd credit-risk-analysis-using-ML
