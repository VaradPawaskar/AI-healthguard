#%%
#Sprint 3
import shap
from xgboost import XGBClassifier

from train import *

# Class imbalance weight: negatives / positives
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    n_jobs=-1,

    # sensible starting point
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,

    subsample=0.9,
    colsample_bytree=0.9,

    min_child_weight=2,
    gamma=0.1,
    reg_lambda=2.0,

    scale_pos_weight=pos_weight  # make it care more about positive (sick) class
)

xgb_base.fit(X_train_scaled, y_train)
y_proba_xgb = xgb_base.predict_proba(X_test_scaled)[:, 1]
y_pred_xgb  = (y_proba_xgb >= 0.5).astype(int)

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "n_estimators":    [300, 400, 500],
    "max_depth":       [3, 4],
    "learning_rate":   [0.01, 0.05, 0.10],
    "subsample":       [0.8, 0.9, 1.0],
    "colsample_bytree":[0.8, 0.9, 1.0],
    "min_child_weight":[1, 2, 4],
    "gamma":           [0, 0.1, 0.3],
    "reg_lambda":      [1.0, 2.0, 4.0],
}

xgb_for_search = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    n_jobs=-1,
    scale_pos_weight=pos_weight
)

xgb_rand = RandomizedSearchCV(
    estimator=xgb_for_search,
    param_distributions=param_dist,
    n_iter=25,          # small but decent
    scoring="recall",   # or "f1" / "fbeta"
    cv=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

xgb_rand.fit(X_train_scaled, y_train)

print("\nBest params:", xgb_rand.best_params_)

best_xgb = xgb_rand.best_estimator_
y_proba_xgb = best_xgb.predict_proba(X_test_scaled)[:, 1]
y_pred_xgb  = (y_proba_xgb >= 0.5).astype(int)

from sklearn.metrics import fbeta_score

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_prec = precision_score(y_test, y_pred_xgb, zero_division=0)
xgb_rec = recall_score(y_test, y_pred_xgb, zero_division=0)
xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
xgb_f2 = fbeta_score(y_test, y_pred_xgb, beta=2, zero_division=0)
xgb_auc = roc_auc_score(y_test, y_proba_xgb)

print("XGB – Accuracy:", xgb_acc)
print("XGB – Recall  :", xgb_rec)
print("XGB – F2      :", xgb_f2)


# 7. MODEL COMPARISON

results_df = pd.DataFrame({
    "model_name": ["LogReg", "RandomForest", "XGBoost"],
    "accuracy":   [log_acc, rf_acc, xgb_acc],
    "precision":  [log_prec, rf_prec, xgb_prec],
    "recall":     [log_rec, rf_rec, xgb_rec],
    "f1":         [log_f1, rf_f1, xgb_f1],
    "roc_auc":    [log_auc, rf_auc, xgb_auc]
})

print("\nModel Comparison:")
display(results_df)

plt.figure(figsize=(10, 5))
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
for metric in metrics:
    plt.plot(results_df["model_name"], results_df[metric], marker="o", label=metric)

plt.xticks(rotation=15)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# 7. MODEL COMPARISON

results_df = pd.DataFrame({
    "model_name": ["LogReg", "RandomForest", "XGBoost"],
    "accuracy":   [log_acc, rf_acc, xgb_acc],
    "precision":  [log_prec, rf_prec, xgb_prec],
    "recall":     [log_rec, rf_rec, xgb_rec],
    "f1":         [log_f1, rf_f1, xgb_f1],
    "roc_auc":    [log_auc, rf_auc, xgb_auc]
})

print("\nModel Comparison:")
display(results_df)

plt.figure(figsize=(10, 5))
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
for metric in metrics:
    plt.plot(results_df["model_name"], results_df[metric], marker="o", label=metric)

plt.xticks(rotation=15)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# 8. SHAP EXPLAINABILITY (FOR XGBOOST)

print("\nInitializing SHAP...")
shap.initjs()

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test_scaled)

print("\nSHAP Summary Plot – Global Feature Importance")
shap.summary_plot(shap_values, X_test, plot_type="bar")

print("\nSHAP Summary Plot – Feature impact")
shap.summary_plot(shap_values, X_test)

sample_index = 1
sample = X_test_scaled[sample_index:sample_index+1]
shap_values_sample = explainer.shap_values(sample)

print(f"\nSHAP force plot for test sample index {sample_index}")
shap.force_plot(explainer.expected_value, shap_values_sample, X_test.iloc[sample_index:sample_index+1])
# %%
