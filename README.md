# 🩺 DiabeSense — Pima Diabetes Prediction App

A production-grade Streamlit dashboard for diabetes risk prediction using the Pima Indian Diabetes Dataset.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Upload your dataset
- In the sidebar, upload your `pima.csv` file
- The model will train automatically (takes ~30 seconds)
- Navigate using the sidebar tabs

---

## 📁 Project Structure

```
diabetes_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🔬 Pipeline Architecture

| Step | Method |
|------|--------|
| Missing value handling | Replace 0→NaN for Glucose, BP, etc. → IQR clipping → Mean imputation |
| Feature selection | LassoCV (L1 regularization) |
| Class balancing | SMOTE (Synthetic Minority Over-sampling) |
| Hyperparameter tuning | RandomizedSearchCV (optimising F1-Score) |
| Primary model | Decision Tree with `ccp_alpha` pruning |
| Comparison models | Logistic Regression, Gradient Boosting |

---

## 📊 App Pages

- **🔮 Predict** — Enter patient clinical values and get an instant risk score, gauge, and clinical notes
- **📊 Model Analytics** — KPI cards, model comparison bar chart, ROC curves, confusion matrix, feature importance
- **📈 Data Insights** — Dataset overview, feature distributions by outcome, correlation heatmap
- **ℹ️ About** — Full pipeline explanation and improvements over baseline

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** → select your repo and `app.py`
4. Click **Deploy** — done!

> Note: The app requires the user to upload `pima.csv` at runtime. The dataset is NOT bundled.

---

## 📦 Dataset

**Pima Indian Diabetes Database**  
Source: [UCI ML Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
768 samples · 8 features · Binary classification (Outcome: 0 / 1)

---

## ⚠️ Disclaimer

For **educational and research purposes only**. Not a substitute for professional medical advice.
