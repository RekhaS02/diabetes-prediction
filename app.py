import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, precision_recall_curve)
from sklearn.feature_selection import SelectKBest, chi2, RFE
from imblearn.over_sampling import SMOTE
import joblib
import io
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabeSense · Pima Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background: #0a0f1e;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1427 !important;
    border-right: 1px solid #1e2d50;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a2a5e 50%, #0d1b3e 100%);
    border: 1px solid #2a3f7a;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(64,120,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4078ff, #7aa3ff, #a8c4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin: 0;
}
.hero-subtitle {
    color: #8899bb;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #0d1427;
    border: 1px solid #1e2d50;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4078ff; }
.metric-label {
    font-size: 0.75rem;
    color: #6677aa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #7aa3ff;
    margin-top: 0.3rem;
}

/* Prediction result */
.result-positive {
    background: linear-gradient(135deg, #2d1010, #3d1515);
    border: 1px solid #7a2020;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #0d2d1a, #0d3d22);
    border: 1px solid #1a7a40;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
}
.result-subtitle { color: #aab0c0; font-size: 0.9rem; margin-top: 0.3rem; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #a0b4e0;
    border-left: 3px solid #4078ff;
    padding-left: 0.8rem;
    margin-bottom: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1427;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #6677aa !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e2d50 !important;
    color: #7aa3ff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2a4db5, #4078ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Sliders & inputs */
[data-baseweb="slider"] { color: #4078ff; }
.stSlider [data-baseweb="slider"] > div > div > div {
    background: #4078ff !important;
}

/* Info boxes */
.info-box {
    background: #0d1e3a;
    border: 1px solid #1e3a6a;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.875rem;
    color: #8899bb;
    line-height: 1.6;
}

/* Progress bar accent */
.stProgress > div > div > div { background: #4078ff !important; }

/* Divider */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e2d50, transparent);
    margin: 1.5rem 0;
}

/* Risk gauge */
.gauge-container { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
COLS_WITH_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_bundle.pkl")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def preprocess(df):
    data = df.copy()
    for col in COLS_WITH_MISSING:
        data[col] = data[col].replace(0, np.nan)
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        data[col] = data[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    imputer = SimpleImputer(strategy="mean")
    features = data[FEATURE_NAMES]
    imputed = pd.DataFrame(imputer.fit_transform(features), columns=FEATURE_NAMES)
    data[FEATURE_NAMES] = imputed.values
    return data, imputer

def train_pipeline(df):
    df_clean, imputer = preprocess(df)
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Lasso feature selection
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(X_scaled, y)
    selected_mask = np.abs(lasso.coef_) > 0
    if selected_mask.sum() < 3:
        selected_mask = np.argsort(np.abs(lasso.coef_))[-5:] 
        mask_bool = np.zeros(len(lasso.coef_), dtype=bool)
        mask_bool[selected_mask] = True
        selected_mask = mask_bool

    selected_features = np.array(FEATURE_NAMES)[selected_mask]
    X_sel = X_scaled[:, selected_mask]

    # SMOTE
    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_sel, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

    # Hyperparameter tuning
    param_dist = {
        'max_depth': [3, 4, 5],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 15],
        'criterion': ['gini', 'entropy'],
        'ccp_alpha': [0.005, 0.01, 0.02]
    }
    rscv = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_distributions=param_dist, n_iter=30,
        cv=5, scoring='f1', random_state=42, n_jobs=-1)
    rscv.fit(X_train, y_train)
    best_dt = rscv.best_estimator_

    # Also train LR & GB for comparison
    lr = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)

    gb_param = {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1],
                'max_depth': [3, 4], 'subsample': [0.7, 0.8]}
    gb_rscv = RandomizedSearchCV(GradientBoostingClassifier(random_state=42),
                                  param_distributions=gb_param, n_iter=12,
                                  cv=5, scoring='f1', random_state=42, n_jobs=-1)
    gb_rscv.fit(X_train, y_train)
    best_gb = gb_rscv.best_estimator_

    models_dict = {'Decision Tree': best_dt, 'Logistic Regression': lr, 'Gradient Boosting': best_gb}

    # Metrics
    metrics_list = []
    roc_data = {}
    for name, model in models_dict.items():
        yp = model.predict(X_test)
        yprob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yprob)
        roc_data[name] = (fpr, tpr)
        metrics_list.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, yp), 4),
            'Precision': round(precision_score(y_test, yp), 4),
            'Recall': round(recall_score(y_test, yp), 4),
            'F1-Score': round(f1_score(y_test, yp), 4),
            'AUC-ROC': round(roc_auc_score(y_test, yprob), 4),
        })

    bundle = {
        'model': best_dt,
        'scaler': scaler,
        'imputer': imputer,
        'selected_mask': selected_mask,
        'selected_features': selected_features,
        'X_test': X_test,
        'y_test': y_test,
        'metrics': pd.DataFrame(metrics_list),
        'roc_data': roc_data,
        'best_params': rscv.best_params_,
        'models_dict': models_dict,
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle

def predict_single(bundle, input_values):
    x = np.array(input_values).reshape(1, -1)
    # Impute
    x_df = pd.DataFrame(x, columns=FEATURE_NAMES)
    for col in COLS_WITH_MISSING:
        x_df[col] = x_df[col].replace(0, np.nan)
    x_imp = bundle['imputer'].transform(x_df)
    x_scaled = bundle['scaler'].transform(x_imp)
    x_sel = x_scaled[:, bundle['selected_mask']]
    prob = bundle['model'].predict_proba(x_sel)[0, 1]
    pred = int(prob >= 0.5)
    return pred, prob

def make_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={'suffix': '%', 'font': {'size': 36, 'color': '#e8eaf0', 'family': 'Syne'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#444', 'tickfont': {'color': '#666'}},
            'bar': {'color': '#ff4d4d' if prob >= 0.5 else '#2ecc71', 'thickness': 0.3},
            'bgcolor': '#0d1427',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': '#0d2d1a'},
                {'range': [33, 66], 'color': '#2d2a0d'},
                {'range': [66, 100], 'color': '#2d0d0d'}
            ],
            'threshold': {'line': {'color': '#7aa3ff', 'width': 2}, 'thickness': 0.75, 'value': 50}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        font={'color': '#e8eaf0'}
    )
    return fig

def make_roc_chart(roc_data):
    colors = {'Decision Tree': '#4078ff', 'Logistic Regression': '#ff9f40', 'Gradient Boosting': '#2ecc71'}
    fig = go.Figure()
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='#444', dash='dash', width=1))
    for name, (fpr, tpr) in roc_data.items():
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', name=name,
            line=dict(color=colors.get(name, '#aaa'), width=2.5)
        ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1427',
        xaxis=dict(title='False Positive Rate', color='#6677aa',
                   gridcolor='#1e2d50', zerolinecolor='#1e2d50'),
        yaxis=dict(title='True Positive Rate', color='#6677aa',
                   gridcolor='#1e2d50', zerolinecolor='#1e2d50'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#aab0c0')),
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color='#8899bb')
    )
    return fig

def make_metrics_chart(metrics_df):
    cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    colors = ['#4078ff', '#ff9f40', '#2ecc71']
    fig = go.Figure()
    for i, row in metrics_df.iterrows():
        fig.add_trace(go.Bar(
            name=row['Model'],
            x=cols,
            y=[row[c] for c in cols],
            marker_color=colors[i % len(colors)],
            text=[f"{row[c]:.3f}" for c in cols],
            textposition='outside',
            textfont=dict(color='#aab0c0', size=11)
        ))
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1427',
        xaxis=dict(color='#6677aa', gridcolor='#1e2d50'),
        yaxis=dict(range=[0, 1.12], color='#6677aa', gridcolor='#1e2d50'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#aab0c0')),
        height=380, margin=dict(l=10, r=10, t=20, b=10),
        font=dict(color='#8899bb')
    )
    return fig

def make_confusion_matrix(bundle):
    y_pred = bundle['model'].predict(bundle['X_test'])
    cm = confusion_matrix(bundle['y_test'], y_pred)
    labels = ['No Diabetes', 'Diabetes']
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, '#0d1427'], [0.5, '#1a2d5a'], [1, '#4078ff']],
        text=cm, texttemplate='<b>%{text}</b>',
        textfont=dict(size=22, color='white'),
        showscale=False
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1427',
        xaxis=dict(title='Predicted', color='#8899bb'),
        yaxis=dict(title='Actual', color='#8899bb', autorange='reversed'),
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color='#8899bb')
    )
    return fig

def make_feature_importance(bundle):
    model = bundle['model']
    features = bundle['selected_features']
    importances = model.feature_importances_
    idx = np.argsort(importances)
    fig = go.Figure(go.Bar(
        x=importances[idx], y=features[idx],
        orientation='h',
        marker=dict(
            color=importances[idx],
            colorscale=[[0, '#1a2d5a'], [1, '#4078ff']],
            showscale=False
        ),
        text=[f"{v:.3f}" for v in importances[idx]],
        textposition='outside',
        textfont=dict(color='#8899bb')
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1427',
        xaxis=dict(color='#6677aa', gridcolor='#1e2d50'),
        yaxis=dict(color='#aab0c0'),
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color='#8899bb')
    )
    return fig

def risk_badge(prob):
    if prob < 0.33:
        return "🟢 Low Risk", "#2ecc71"
    elif prob < 0.66:
        return "🟡 Moderate Risk", "#f0a500"
    else:
        return "🔴 High Risk", "#e74c3c"

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem'>
        <span style='font-size:2.5rem'>🩺</span>
        <p style='font-family:Syne; font-weight:700; color:#7aa3ff; font-size:1.1rem; margin:0.3rem 0 0'>DiabeSense</p>
        <p style='color:#4a5a80; font-size:0.78rem; margin:0'>Pima Indian Diabetes Predictor</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Upload Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload pima.csv", type=["csv"], label_visibility="collapsed")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)

    page = st.radio("", ["🔮 Predict", "📊 Model Analytics", "📈 Data Insights", "ℹ️ About"],
                    label_visibility="collapsed")

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <b style='color:#7aa3ff'>Dataset:</b> Pima Indian Diabetes Database<br>
        <b style='color:#7aa3ff'>Source:</b> UCI ML Repository<br>
        <b style='color:#7aa3ff'>Samples:</b> 768 patients<br>
        <b style='color:#7aa3ff'>Features:</b> 8 clinical variables
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-title">DiabeSense</p>
    <p class="hero-subtitle">Clinical Diabetes Risk Prediction · Pima Indian Dataset · Decision Tree Ensemble</p>
</div>
""", unsafe_allow_html=True)

# Load or train model
bundle = None
df = None

if uploaded:
    df = pd.read_csv(uploaded)
    if 'Outcome' not in df.columns:
        st.error("❌ CSV must contain an 'Outcome' column.")
    else:
        with st.spinner("⚙️ Training model pipeline..."):
            bundle = train_pipeline(df)
        st.success(f"✅ Model trained on {len(df)} samples. Best DT params: `{bundle['best_params']}`")
elif os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
else:
    st.info("👈 Upload your **pima.csv** in the sidebar to get started.")

# ─── PAGE: PREDICT ───────────────────────────
if "Predict" in page:
    st.markdown('<div class="section-title">Patient Clinical Input</div>', unsafe_allow_html=True)

    if bundle is None:
        st.warning("Please upload the dataset first to train the model.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pregnancies = st.slider("Pregnancies", 0, 17, 1, help="Number of times pregnant")
            glucose = st.slider("Glucose (mg/dL)", 0, 200, 120, help="Plasma glucose concentration")
        with c2:
            blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 122, 72, help="Diastolic blood pressure")
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 23, help="Triceps skin fold thickness")
        with c3:
            insulin = st.slider("Insulin (μU/ml)", 0, 846, 80, help="2-Hour serum insulin")
            bmi = st.slider("BMI (kg/m²)", 0.0, 67.1, 32.0, step=0.1, help="Body mass index")
        with c4:
            dpf = st.slider("Diabetes Pedigree", 0.078, 2.42, 0.47, step=0.001,
                            help="Diabetes pedigree function — family history score")
            age = st.slider("Age (years)", 21, 81, 33)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        if st.button("⚡  Run Prediction", use_container_width=True):
            inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            pred, prob = predict_single(bundle, inputs)
            badge, badge_color = risk_badge(prob)

            col_g, col_r = st.columns([1, 1.6])
            with col_g:
                st.markdown('<div class="section-title">Risk Score</div>', unsafe_allow_html=True)
                st.plotly_chart(make_gauge(prob), use_container_width=True, config={'displayModeBar': False})

            with col_r:
                st.markdown('<div class="section-title">Diagnosis</div>', unsafe_allow_html=True)
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-positive">
                        <p class="result-title" style="color:#ff6b6b">⚠️ Diabetes Detected</p>
                        <p class="result-subtitle">Probability: <b style="color:#ff9999">{prob:.1%}</b></p>
                        <p class="result-subtitle">{badge}</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <p class="result-title" style="color:#52e090">✅ No Diabetes Detected</p>
                        <p class="result-subtitle">Probability of diabetes: <b style="color:#88ffb8">{prob:.1%}</b></p>
                        <p class="result-subtitle">{badge}</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
                # Feature contribution breakdown
                x = np.array(inputs).reshape(1, -1)
                x_df = pd.DataFrame(x, columns=FEATURE_NAMES)
                for col in COLS_WITH_MISSING:
                    x_df[col] = x_df[col].replace(0, np.nan)
                x_imp = bundle['imputer'].transform(x_df)
                x_scaled = bundle['scaler'].transform(x_imp)
                x_sel = x_scaled[:, bundle['selected_mask']]

                features_used = bundle['selected_features']
                importances = bundle['model'].feature_importances_

                importance_df = pd.DataFrame({
                    'Feature': features_used,
                    'Importance': importances,
                    'Value': x_sel[0]
                }).sort_values('Importance', ascending=True)  # ascending so highest bar is on top

                n_features = len(importance_df)
                chart_height = max(280, n_features * 52)

                fig_imp = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color=importance_df['Importance'],
                        colorscale=[[0, '#1a2d5a'], [1, '#4078ff']],
                        showscale=False
                    ),
                    text=[f"{v:.3f}" for v in importance_df['Importance']],
                    textposition='outside',
                    textfont=dict(color='#aab0c0', size=12),
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
                ))
                fig_imp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#0d1427',
                    height=chart_height,
                    showlegend=False,
                    xaxis=dict(
                        title=dict(text='Importance Score', font=dict(color='#8899bb', size=12)),
                        color='#6677aa',
                        gridcolor='#1e2d50',
                        tickfont=dict(color='#8899bb', size=11),
                        range=[0, importance_df['Importance'].max() * 1.25]
                    ),
                    yaxis=dict(
                        title=dict(text='Feature', font=dict(color='#8899bb', size=12)),
                        color='#aab0c0',
                        tickfont=dict(color='#c0cce8', size=12),
                        automargin=True
                    ),
                    margin=dict(l=10, r=60, t=10, b=40),
                )
                st.markdown('<div class="section-title" style="margin-top:0.5rem">Feature Weights Used</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})

            # Clinical summary
            st.markdown('<div class="section-title">Clinical Summary</div>', unsafe_allow_html=True)
            notes = []
            if glucose > 140: notes.append(("🔴", "Glucose", f"{glucose} mg/dL is above the normal threshold (< 140 mg/dL)"))
            if bmi > 30: notes.append(("🟡", "BMI", f"{bmi:.1f} kg/m² indicates obesity (≥ 30)"))
            if blood_pressure > 90: notes.append(("🟡", "Blood Pressure", f"{blood_pressure} mmHg is elevated"))
            if insulin > 200: notes.append(("🟡", "Insulin", f"{insulin} μU/ml is high (normal < 200)"))
            if age > 45: notes.append(("🟡", "Age", f"Age {age} is a risk factor for Type-2 diabetes"))
            if dpf > 1.0: notes.append(("🔴", "Pedigree", f"High family history score ({dpf:.3f})"))

            if notes:
                ncols = st.columns(min(len(notes), 3))
                for i, (icon, label, msg) in enumerate(notes):
                    with ncols[i % 3]:
                        st.markdown(f"""
                        <div class="info-box">
                            <b style="color:#7aa3ff">{icon} {label}</b><br>{msg}
                        </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">✅ All clinical markers appear within normal ranges.</div>',
                            unsafe_allow_html=True)

# ─── PAGE: MODEL ANALYTICS ────────────────────
elif "Analytics" in page:
    if bundle is None:
        st.warning("Please upload the dataset first.")
    else:
        metrics_df = bundle['metrics']

        # Top KPI row
        best_row = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
        kpi_cols = st.columns(5)
        for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']):
            val = metrics_df.loc[metrics_df['Model'] == 'Decision Tree', metric].values[0]
            with kpi_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric}</div>
                    <div class="metric-value">{val:.3f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Model Comparison", "📉 ROC Curves", "🔲 Confusion Matrix", "🌿 Feature Importance"])

        with tab1:
            st.markdown('<div class="section-title">All Models — Performance Metrics</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(make_metrics_chart(metrics_df), use_container_width=True,
                            config={'displayModeBar': False})
            st.dataframe(
                metrics_df.set_index('Model').style
                .format("{:.4f}")
                .background_gradient(cmap='Blues', axis=None),
                use_container_width=True
            )

        with tab2:
            st.markdown('<div class="section-title">ROC Curves — All Models</div>', unsafe_allow_html=True)
            st.plotly_chart(make_roc_chart(bundle['roc_data']), use_container_width=True,
                            config={'displayModeBar': False})

        with tab3:
            st.markdown('<div class="section-title">Confusion Matrix — Best Decision Tree</div>',
                        unsafe_allow_html=True)
            col_cm, col_stat = st.columns([1.5, 1])
            with col_cm:
                st.plotly_chart(make_confusion_matrix(bundle), use_container_width=True,
                                config={'displayModeBar': False})
            with col_stat:
                y_pred = bundle['model'].predict(bundle['X_test'])
                cm = confusion_matrix(bundle['y_test'], y_pred)
                tn, fp, fn, tp = cm.ravel()
                st.markdown(f"""
                <div class="info-box" style="line-height:2.2">
                    <b style="color:#2ecc71">True Negative (TN):</b> {tn}<br>
                    <b style="color:#4078ff">False Positive (FP):</b> {fp}<br>
                    <b style="color:#ff9f40">False Negative (FN):</b> {fn}<br>
                    <b style="color:#e74c3c">True Positive (TP):</b> {tp}<br>
                    <hr style="border-color:#1e2d50; margin:0.5rem 0">
                    <b style="color:#aab0c0">Specificity:</b> {tn/(tn+fp):.3f}<br>
                    <b style="color:#aab0c0">Sensitivity:</b> {tp/(tp+fn):.3f}
                </div>""", unsafe_allow_html=True)

        with tab4:
            st.markdown('<div class="section-title">Feature Importance — Decision Tree</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(make_feature_importance(bundle), use_container_width=True,
                            config={'displayModeBar': False})
            st.markdown(f"""
            <div class="info-box">
                <b style="color:#7aa3ff">Selected Features:</b> {', '.join(bundle['selected_features'])}<br>
                <b style="color:#7aa3ff">Selection Method:</b> Lasso Regression (L1 penalty) + threshold on |coefficient| > 0<br>
                <b style="color:#7aa3ff">Best Parameters:</b> {bundle['best_params']}
            </div>""", unsafe_allow_html=True)

# ─── PAGE: DATA INSIGHTS ─────────────────────
elif "Insights" in page:
    if df is None:
        if os.path.exists(MODEL_PATH):
            st.info("Dataset not in memory. Re-upload CSV to view data insights.")
        else:
            st.warning("Please upload the dataset first.")
    else:
        st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Samples</div>'
                        f'<div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
        with m2:
            pos = df['Outcome'].sum()
            st.markdown(f'<div class="metric-card"><div class="metric-label">Diabetic</div>'
                        f'<div class="metric-value">{pos}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Non-Diabetic</div>'
                        f'<div class="metric-value">{len(df)-pos}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Prevalence</div>'
                        f'<div class="metric-value">{pos/len(df):.1%}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        tab_a, tab_b, tab_c = st.tabs(["📋 Raw Data", "📊 Distributions", "🔗 Correlation"])

        with tab_a:
            st.dataframe(df.describe().T.style.format("{:.2f}").background_gradient(cmap='Blues', axis=1),
                         use_container_width=True)
            st.dataframe(df.head(20), use_container_width=True)

        with tab_b:
            feat = st.selectbox("Select feature", FEATURE_NAMES)
            fig_hist = px.histogram(df, x=feat, color='Outcome',
                                    color_discrete_map={0: '#4078ff', 1: '#e74c3c'},
                                    barmode='overlay', nbins=30, opacity=0.75,
                                    template='none')
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1427',
                xaxis=dict(color='#6677aa', gridcolor='#1e2d50'),
                yaxis=dict(color='#6677aa', gridcolor='#1e2d50'),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#aab0c0'),
                            title=dict(text='Outcome', font=dict(color='#aab0c0'))),
                height=340, margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

        with tab_c:
            corr = df.corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale=[[0, '#0d1427'], [0.5, '#1e2d50'], [1, '#4078ff']],
                text=np.round(corr.values, 2), texttemplate='%{text}',
                textfont=dict(size=11, color='#c8d8ff'), showscale=True,
                zmin=-1, zmax=1
            ))
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1427',
                xaxis=dict(color='#8899bb'), yaxis=dict(color='#8899bb'),
                height=420, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})

# ─── PAGE: ABOUT ─────────────────────────────
elif "About" in page:
    st.markdown("""
    <div style="max-width: 760px; margin: 0 auto; line-height: 1.9; color: #aab0c0;">
    <div class="section-title">Project Overview</div>
    <div class="info-box">
    This application predicts Type-2 diabetes risk using the <b style="color:#7aa3ff">Pima Indian Diabetes Dataset</b>
    from the UCI Machine Learning Repository. The pipeline applies state-of-the-art preprocessing,
    class balancing, and hyperparameter-tuned Decision Tree classification.
    </div>

    <div class="section-title" style="margin-top:1.5rem">Pipeline Architecture</div>
    <div class="info-box">
    <b style="color:#7aa3ff">1. Preprocessing</b><br>
    Replace physiologically impossible zeros (Glucose, BP, etc.) with NaN → IQR-based outlier
    clipping → mean imputation.<br><br>
    <b style="color:#7aa3ff">2. Feature Selection</b><br>
    LassoCV (L1 regularization) to zero-out uninformative features, keeping only those with
    non-zero coefficients.<br><br>
    <b style="color:#7aa3ff">3. Class Balancing</b><br>
    SMOTE (Synthetic Minority Over-sampling Technique) to address the 65/35 class imbalance.<br><br>
    <b style="color:#7aa3ff">4. Hyperparameter Tuning</b><br>
    RandomizedSearchCV over depth, split, leaf, criterion, and ccp_alpha — optimising F1-Score
    to balance precision and recall.<br><br>
    <b style="color:#7aa3ff">5. Comparison Models</b><br>
    Logistic Regression (L2, balanced) and Gradient Boosting Classifier trained on the same
    split for benchmarking.
    </div>

    <div class="section-title" style="margin-top:1.5rem">Improvements Over Baseline</div>
    <div class="info-box">
    ✅ <b style="color:#7aa3ff">Lasso feature selection</b> replaces chi-square + RFE cascade.<br>
    ✅ <b style="color:#7aa3ff">ccp_alpha pruning</b> added to prevent Decision Tree overfitting.<br>
    ✅ <b style="color:#7aa3ff">F1-Score</b> used as tuning objective (better for imbalanced classes).<br>
    ✅ <b style="color:#7aa3ff">Risk gauge + clinical notes</b> for interpretable output.<br>
    ✅ <b style="color:#7aa3ff">ROC curves, confusion matrix, feature importance</b> all in one dashboard.<br>
    ✅ <b style="color:#7aa3ff">Future risk projection</b> replaced with reliable probabilistic output.
    </div>

    <div class="section-title" style="margin-top:1.5rem">Disclaimer</div>
    <div class="info-box">
    ⚠️ This tool is for <b>educational and research purposes only</b>. It is not a substitute for
    professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare
    provider.
    </div>
    </div>
    """, unsafe_allow_html=True)
