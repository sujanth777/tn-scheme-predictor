"""
train_model_v2.py
-------------------
Upgraded training script for the ML subject review:
  1. Trains Random Forest WITH class_weight='balanced' (handles rare schemes better)
  2. Compares 3 models: Random Forest, Logistic Regression, XGBoost
  3. Saves confusion matrices for the 3 weakest schemes as PNGs
  4. Saves the best-performing model as scheme_model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("citizens_dataset.csv")
scheme_cols = [c for c in df.columns if c.startswith("scheme_")]
feature_cols = [c for c in df.columns if c not in scheme_cols]

X = pd.get_dummies(df[feature_cols])
Y = df[scheme_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------------
# Define the 3 models to compare
# ----------------------------------------------------------------------
models = {
    "Random Forest (balanced)": MultiOutputClassifier(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    ),
    "Logistic Regression": MultiOutputClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced")
    ),
    "XGBoost": MultiOutputClassifier(
        XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42)
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    exact_match = accuracy_score(y_test, preds)
    f1_micro = f1_score(y_test, preds, average="micro")
    f1_macro = f1_score(y_test, preds, average="macro")

    results.append({
        "Model": name,
        "Exact Match Accuracy": round(exact_match, 3),
        "F1 (micro)": round(f1_micro, 3),
        "F1 (macro)": round(f1_macro, 3),
    })
    trained_models[name] = (model, preds)
    print(f"  Exact-match accuracy: {exact_match:.3f} | F1 micro: {f1_micro:.3f} | F1 macro: {f1_macro:.3f}")

# ----------------------------------------------------------------------
# Model comparison table
# ----------------------------------------------------------------------
comparison_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(comparison_df.to_string(index=False))
comparison_df.to_csv("model_comparison.csv", index=False)

# Pick best model by F1 macro (fairer to rare schemes than micro/accuracy)
best_model_name = comparison_df.loc[comparison_df["F1 (macro)"].idxmax(), "Model"]
best_model, best_preds = trained_models[best_model_name]
print(f"\nBest model (by F1 macro): {best_model_name}")

# ----------------------------------------------------------------------
# Per-scheme F1 for the best model
# ----------------------------------------------------------------------
print(f"\nPer-scheme F1 scores ({best_model_name}):")
per_scheme_f1 = f1_score(y_test, best_preds, average=None)
f1_summary = []
for name, score in zip(scheme_cols, per_scheme_f1):
    clean_name = name.replace("scheme_", "").replace("_", " ")
    print(f"  {clean_name:45s} F1={score:.3f}")
    f1_summary.append({"Scheme": clean_name, "F1": round(score, 3)})

pd.DataFrame(f1_summary).to_csv("per_scheme_f1.csv", index=False)

# ----------------------------------------------------------------------
# Confusion matrices for the 3 weakest schemes (lowest F1)
# ----------------------------------------------------------------------
weakest = pd.DataFrame(f1_summary).nsmallest(3, "F1")
print(f"\nGenerating confusion matrices for weakest schemes: {list(weakest['Scheme'])}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
y_test_arr = y_test.values
best_preds_arr = np.array(best_preds)

for i, (_, row) in enumerate(weakest.iterrows()):
    scheme_idx = list(scheme_cols).index("scheme_" + row["Scheme"].replace(" ", "_"))
    cm = confusion_matrix(y_test_arr[:, scheme_idx], best_preds_arr[:, scheme_idx])
    axes[i].imshow(cm, cmap="Blues")
    axes[i].set_title(f"{row['Scheme']}\n(F1={row['F1']:.2f})", fontsize=10)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
    axes[i].set_xticks([0, 1]); axes[i].set_xticklabels(["No", "Yes"])
    axes[i].set_yticks([0, 1]); axes[i].set_yticklabels(["No", "Yes"])
    for r in range(2):
        for c in range(2):
            axes[i].text(c, r, cm[r, c], ha="center", va="center", fontsize=12)

plt.tight_layout()
plt.savefig("confusion_matrices_weakest_schemes.png", dpi=150)
plt.close()
print("Saved confusion_matrices_weakest_schemes.png")

# ----------------------------------------------------------------------
# Save best model
# ----------------------------------------------------------------------
joblib.dump(best_model, "scheme_model.pkl")
joblib.dump(list(X.columns), "model_feature_columns.pkl")
print(f"\nSaved best model ({best_model_name}) -> scheme_model.pkl")
