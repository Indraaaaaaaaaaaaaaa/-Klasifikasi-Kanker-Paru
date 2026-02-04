# ===============================
# TRAINING FINAL — LUNG CANCER
# ===============================

import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)

from xgboost import XGBClassifier

# ===============================
# CONFIG
# ===============================
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50

FEATURES = [
    "GENDER","AGE","SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE",
    "CHRONIC_DISEASE","FATIGUE","ALLERGY","WHEEZING","ALCOHOL_CONSUMING",
    "COUGHING","SHORTNESS_OF_BREATH","SWALLOWING_DIFFICULTY","CHEST_PAIN"
]
TARGET = "LUNG_CANCER"

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("lcs_clean.csv")
X = df[FEATURES]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ===============================
# CLASS WEIGHT
# ===============================
classes = np.array([0, 1])
cw_vals = compute_class_weight("balanced", classes=classes, y=y_train)
cw = {int(c): float(w) for c, w in zip(classes, cw_vals)}

# ===============================
# MODELS
# ===============================
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight=cw))
])

xgb = XGBClassifier(**XGB_PARAMS)

# Train
lr.fit(X_train, y_train)
xgb.fit(X_train, y_train, sample_weight=[cw[i] for i in y_train])

# ===============================
# EVALUATION
# ===============================
def evaluate(name, model):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "proba": proba
    }

res_lr = evaluate("LogisticRegression", lr)
res_xgb = evaluate("XGBoost", xgb)

best = max([res_lr, res_xgb], key=lambda x: x["f1"])
best_model = lr if best["model"] == "LogisticRegression" else xgb
best_proba = best["proba"]

# ===============================
# SAVE MODEL
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_lung_cancer_model.joblib")

# ===============================
# SAVE METRICS
# ===============================
metrics = best.copy()
metrics.pop("proba")

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# ===============================
# VISUALIZATION
# ===============================
os.makedirs("outputs", exist_ok=True)

# ROC
fpr, tpr, _ = roc_curve(y_test, best_proba)
plt.plot(fpr, tpr)
plt.savefig("outputs/roc_curve.png")
plt.close()

# PR
prec, rec, _ = precision_recall_curve(y_test, best_proba)
plt.plot(rec, prec)
plt.savefig("outputs/pr_curve.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, (best_proba >= THRESHOLD).astype(int))
plt.imshow(cm)
plt.colorbar()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("TRAINING SELESAI ✔")
