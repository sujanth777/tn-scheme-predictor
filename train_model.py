"""
train_model.py
----------------
Baseline multi-label Random Forest model.
Reads citizens_dataset.csv -> encodes features -> trains -> prints accuracy/F1.
Saves the trained model with joblib so the (future) Streamlit app can load it.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

df = pd.read_csv("citizens_dataset.csv")
scheme_cols = [c for c in df.columns if c.startswith("scheme_")]
feature_cols = [c for c in df.columns if c not in scheme_cols]

# ---- Encode categorical (Yes/No/Gender etc.) columns into numbers ----
X = pd.get_dummies(df[feature_cols])
Y = df[scheme_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ---- Train baseline multi-label Random Forest ----
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# ---- Evaluate ----
preds = model.predict(X_test)

exact_match_accuracy = accuracy_score(y_test, preds)  # ALL labels must match per row
f1_micro = f1_score(y_test, preds, average="micro")
f1_macro = f1_score(y_test, preds, average="macro")

print(f"Exact-match accuracy (all schemes correct per citizen): {exact_match_accuracy:.3f}")
print(f"F1 score (micro-average): {f1_micro:.3f}")
print(f"F1 score (macro-average): {f1_macro:.3f}")

# ---- Per-scheme F1 breakdown ----
print("\nPer-scheme F1 scores:")
per_scheme_f1 = f1_score(y_test, preds, average=None)
for name, score in zip(scheme_cols, per_scheme_f1):
    print(f"  {name.replace('scheme_', '').replace('_', ' '):45s} F1={score:.3f}")

# ---- Save model + feature column list (so the app can rebuild the same encoding) ----
joblib.dump(model, "scheme_model.pkl")
joblib.dump(list(X.columns), "model_feature_columns.pkl")
print("\nSaved trained model -> scheme_model.pkl")
