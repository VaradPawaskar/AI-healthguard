#%%
#Sprint 2
from IPython.display import display
from EDA import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

# 3. TRAIN–TEST SPLIT + SCALING

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. BASELINE MODEL 1 – LOGISTIC REGRESSION

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
log_auc = roc_auc_score(y_test, y_proba_log)

print("\n================ Logistic Regression (balanced) ================")
print("Accuracy :", round(log_acc, 3))
print("Precision:", round(log_prec, 3))
print("Recall   :", round(log_rec, 3))
print("F1-score :", round(log_f1, 3))
print("ROC-AUC  :", round(log_auc, 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log, zero_division=0))

cm_log = confusion_matrix(y_test, y_pred_log)
ConfusionMatrixDisplay(cm_log).plot()
plt.title("Logistic Regression – Confusion Matrix")
plt.show()

fpr_log, tpr_log, thr_log = roc_curve(y_test, y_proba_log)
plt.figure()
plt.plot(fpr_log, tpr_log, label=f"LogReg (AUC={log_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression – ROC Curve")
plt.legend()
plt.show()

# 5. BASELINE MODEL 2 – RANDOM FOREST

rf_clf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)

rf_clf.fit(X_train, y_train)   # trees don't need scaling

y_pred_rf = rf_clf.predict(X_test)
y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
rf_rec = recall_score(y_test, y_pred_rf, zero_division=0)
rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
rf_auc = roc_auc_score(y_test, y_proba_rf)

print("\n================ Random Forest (baseline) ================")
print("Accuracy :", round(rf_acc, 3))
print("Precision:", round(rf_prec, 3))
print("Recall   :", round(rf_rec, 3))
print("F1-score :", round(rf_f1, 3))
print("ROC-AUC  :", round(rf_auc, 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))

cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(cm_rf).plot()
plt.title("Random Forest – Confusion Matrix")
plt.show()

fpr_rf, tpr_rf, thr_rf = roc_curve(y_test, y_proba_rf)
plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest – ROC Curve")
plt.legend()
plt.show()

# %%
