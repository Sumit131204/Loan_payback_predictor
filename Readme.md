# 🏦 Loan Payback Prediction using CatBoost + Streamlit

This project predicts the **probability that a borrower will pay back a loan**, based on various financial and demographic factors.  
It is built for the **Kaggle Playground Series - 2025** and demonstrates a complete **Machine Learning workflow** — from model training and preprocessing to deployment through a **Streamlit web interface**.

---

## 🚀 Project Overview

### 🎯 Goal

Predict whether a borrower will **repay their loan** (or not) using machine learning techniques on structured tabular data.

### 🧠 Model Used

- **Algorithm:** [CatBoost Classifier](https://catboost.ai/)
- **Evaluation Metric:** ROC-AUC score
- **Pipeline Components:**
  - Label Encoding (`loan_purpose`, `grade_subgrade`)
  - Standard Scaler + OneHotEncoder via ColumnTransformer
  - CatBoost model trained and exported as `.cbm`
  - Web deployment using **Streamlit**

---

## 🧩 Features

- **Interactive Web App** built with Streamlit
- **Pre-trained Model Integration:** CatBoost model (`.cbm`) loaded dynamically
- **Automatic Preprocessing:** Uses saved `preprocessor.joblib` and `label_encoders.joblib`
- **Real-Time Predictions:** Input borrower details to get repayment probability instantly
- **Downloadable Results:** Save your prediction as a `.csv` file

---

## 📁 Folder Structure

📦 Loan-Payback-Prediction
├── app.py # Streamlit web app
├── catboost_model.cbm # Trained CatBoost model
├── preprocessor.joblib # Fitted ColumnTransformer (scaler + one-hot encoder)
├── label_encoders.joblib # Dictionary of fitted LabelEncoders
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── notebooks/
└── kaggle_predicting_loans.ipynb # Training notebook

## ✔️ Download dataset from here :

link: https://drive.google.com/drive/folders/1a_Tk33c0CKSmncWMPWjjr2lVlj2e_d68?usp=sharing

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Loan-Payback-Prediction.git
cd Loan-Payback-Prediction
python -m venv venv
venv\Scripts\activate      # (Windows)
source venv/bin/activate   # (Mac/Linux)
pip install -r requirements.txt
streamlit run app.py


## Technologies Used
Python 3.9+
Libraries:
    pandas
    numpy
    catboost
    scikit-learn
    joblib
    streamlit


🧑‍💻 Author

Sumit Kumar
📘 Electronics and Telecommunication Engineering
💡 Machine Learning, AI, and Data Science

If you find this useful, give it a ⭐ on GitHub!
```
