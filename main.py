# -*- coding: utf-8 -*-
"""Sprint 3_ FINAL.ipynb
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, fbeta_score   # ✅ ADDED
)

from xgboost import XGBClassifier
import shap
import joblib
import os

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

RANDOM_STATE = 42
TEST_SIZE = 0.2

DATA_PATH = "processed.cleveland.data"
TARGET_COL = "target"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", TARGET_COL
]

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns,
    na_values=["?"]
)

print("Raw shape:", df.shape)
display(df.head())

print("\nMissing values before cleaning:")
print(df.isna().sum())

if df[TARGET_COL].nunique() > 2:
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

df = df.drop_duplicates()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= Logistic Regression =================

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

log_reg.fit(X_train_scaled, y_train)

y_pred_log = log_reg.predict(X_test_scaled)
y_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

log_acc = accuracy_score(y_test, y_pred_log)
log_prec = precision_score(y_test, y_pred_log, zero_division=0)
log_rec = recall_score(y_test, y_pred_log, zero_division=0)
log_f1 = f1_score(y_test, y_pred_log, zero_division=0)
log_f2 = fbeta_score(y_test, y_pred_log, beta=2, zero_division=0)  # ✅ ADDED
log_auc = roc_auc_score(y_test, y_proba_log)

print("\n================ Logistic Regression ================")
print("Accuracy :", round(log_acc, 3))
print("Precision:", round(log_prec, 3))
print("Recall   :", round(log_rec, 3))
print("F1-score :", round(log_f1, 3))
print("F2-score :", round(log_f2, 3))  # ✅ ADDED
print("ROC-AUC  :", round(log_auc, 3))

# ================= Random Forest =================

rf_clf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)

rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)
y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
rf_rec = recall_score(y_test, y_pred_rf, zero_division=0)
rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
rf_f2 = fbeta_score(y_test, y_pred_rf, beta=2, zero_division=0)  # ✅ ADDED
rf_auc = roc_auc_score(y_test, y_proba_rf)

print("\n================ Random Forest ================")
print("Accuracy :", round(rf_acc, 3))
print("Precision:", round(rf_prec, 3))
print("Recall   :", round(rf_rec, 3))
print("F1-score :", round(rf_f1, 3))
print("F2-score :", round(rf_f2, 3))  # ✅ ADDED
print("ROC-AUC  :", round(rf_auc, 3))

# ================= XGBoost =================

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    gamma=0.1,
    reg_lambda=2.0,
    scale_pos_weight=pos_weight
)

xgb_base.fit(X_train_scaled, y_train)

y_proba_xgb = xgb_base.predict_proba(X_test_scaled)[:, 1]
y_pred_xgb = (y_proba_xgb >= 0.5).astype(int)

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_prec = precision_score(y_test, y_pred_xgb, zero_division=0)
xgb_rec = recall_score(y_test, y_pred_xgb, zero_division=0)
xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
xgb_f2 = fbeta_score(y_test, y_pred_xgb, beta=2, zero_division=0)
xgb_auc = roc_auc_score(y_test, y_proba_xgb)

# ================= Model Comparison =================

results_df = pd.DataFrame({
    "model_name": ["LogReg", "RandomForest", "XGBoost"],
    "accuracy":   [log_acc, rf_acc, xgb_acc],
    "precision":  [log_prec, rf_prec, xgb_prec],
    "recall":     [log_rec, rf_rec, xgb_rec],
    "f1":         [log_f1, rf_f1, xgb_f1],
    "f2":         [log_f2, rf_f2, xgb_f2],  # ✅ ADDED
    "roc_auc":    [log_auc, rf_auc, xgb_auc]
})

print("\nModel Comparison:")
display(results_df)
