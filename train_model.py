"""
train_model.py
---------------
End-to-end training pipeline for EduPredict AI.

Run with:  python train_model.py

Steps:
1. Load raw data
2. Clean it
3. Engineer a couple of simple features
4. Encode categorical columns + scale numeric ones
5. Train Logistic Regression, Decision Tree, Random Forest
6. Evaluate all three and pick the best by F1 score (macro)
7. Save the best model + all preprocessing objects + comparison plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from utils import (
    load_data, clean_data, engineer_features, encode_features, scale_features,
    save_artifacts, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL
)

PLOTS_DIR = "plots"


# ---------------------------------------------------------------------------
# 1. EXPLORATORY PLOTS (built from the cleaned data, saved for the Analytics
#    page and the README). Kept separate from model-evaluation plots below.
# ---------------------------------------------------------------------------
def make_eda_plots(df: pd.DataFrame):
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Grade distribution
    plt.figure(figsize=(6, 4))
    df[TARGET_COL].value_counts().sort_index().plot(kind="bar", color="#4C72B0")
    plt.title("Grade Distribution")
    plt.xlabel("Grade")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/grade_distribution.png", dpi=120)
    plt.close()

    # Attendance distribution
    plt.figure(figsize=(6, 4))
    plt.hist(df["attendance_percentage"], bins=20, color="#55A868", edgecolor="black")
    plt.title("Attendance Distribution")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/attendance_distribution.png", dpi=120)
    plt.close()

    # Correlation heatmap (numeric columns only) — built manually with
    # matplotlib's imshow since Seaborn is not allowed in this project.
    numeric_df = df[NUMERIC_COLS + ["study_efficiency", "academic_risk"]]
    corr = numeric_df.corr()
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 2. MODEL TRAINING + EVALUATION
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names):
    """
    Train Logistic Regression, Decision Tree, and Random Forest, then
    evaluate each on the held-out test set.

    WHY THESE THREE MODELS:
    - Logistic Regression: a simple, fast linear baseline. Easy to explain
      ("each feature gets a weight; weighted sum decides the class").
    - Decision Tree: captures non-linear rules ("if attendance > 80 AND
      failures == 0 -> likely A"), and is very easy to visualize/explain.
    - Random Forest: an ensemble of many decision trees; usually the most
      accurate of the three because averaging many trees reduces
      overfitting compared to a single tree.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # METRIC EXPLANATIONS (also shown in the notebook):
        # Accuracy  - % of predictions that were exactly correct. Can be
        #             misleading on imbalanced classes (e.g. few "A" grades).
        # Precision - of all students predicted as grade X, how many really
        #             were X. High precision = few false alarms.
        # Recall    - of all students who really were grade X, how many the
        #             model correctly found. High recall = few missed cases.
        # F1 Score  - harmonic mean of precision & recall; a single balanced
        #             score, especially useful with imbalanced classes.
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        trained_models[name] = model

        print(f"\n{name}")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall   : {rec:.3f}")
        print(f"  F1 Score : {f1:.3f}")

    # Pick the best model by macro F1 — the fairest single number to
    # compare on an imbalanced multi-class problem like this one.
    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} (F1 = {results[best_name]['f1']:.3f})")

    # --- Plot: model accuracy comparison ---
    plt.figure(figsize=(6, 4))
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    bars = plt.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_accuracy_comparison.png", dpi=120)
    plt.close()

    # --- Plot: confusion matrix for the best model ---
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=120)
    plt.close()

    # --- Plot: feature importance ---
    # Tree-based models expose feature_importances_ directly (how much each
    # feature reduces impurity across all splits). Logistic Regression has
    # no such attribute, but the absolute value of its learned coefficients
    # serves the same purpose: bigger |coefficient| = bigger influence on
    # the prediction (features were scaled, so coefficients are comparable).
    feature_names = X_train.columns
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_).mean(axis=0)
    else:
        importances = None

    if importances is not None:
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(7, 4))
        plt.bar(range(len(importances)), importances[order], color="#8172B2")
        plt.xticks(range(len(importances)), feature_names[order], rotation=45, ha="right")
        plt.title(f"Feature Importance - {best_name}")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=120)
        plt.close()

    return best_name, best_model, results


def main():
    print("Loading data...")
    df_raw = load_data()

    print("Cleaning data...")
    df_clean = clean_data(df_raw)

    print("Engineering features...")
    df_feat = engineer_features(df_clean)

    print("Generating EDA plots...")
    make_eda_plots(df_feat)

    print("Encoding categorical features...")
    df_encoded, encoders = encode_features(df_feat, fit=True)

    # Encode the target labels (A, B, C, D, F) to integers 0-4
    target_encoder = LabelEncoder()
    df_encoded[TARGET_COL] = target_encoder.fit_transform(df_encoded[TARGET_COL])

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS + ["study_efficiency", "academic_risk"]
    X = df_encoded[feature_cols]
    y = df_encoded[TARGET_COL]

    print("Scaling numeric features...")
    # Only the numeric columns need scaling — see utils.scale_features
    # docstring for why Random Forest doesn't need this step.
    X_scaled, scaler = scale_features(X, NUMERIC_COLS + ["study_efficiency"], fit=True)

    print("Splitting into train/test sets (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models...")
    best_name, best_model, results = train_and_evaluate(
        X_train, X_test, y_train, y_test, class_names=target_encoder.classes_
    )

    print("Saving best model + preprocessing artifacts...")
    save_artifacts(best_model, encoders, scaler, feature_cols, target_encoder)

    print("\nDone. Best model saved to models/best_model.pkl")
    return results


if __name__ == "__main__":
    main()
