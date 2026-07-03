"""Train and evaluate diabetes risk models on the Pima Indians dataset.

Two kinds of evaluation are reported:
  1. Stratified 5-fold cross-validation  -> a trustworthy performance estimate.
  2. A single held-out test split        -> a detailed per-class report and
                                             confusion matrix.
The deployed model (Logistic Regression) and its scaler are saved separately
so the Streamlit app can load them independently.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/diabetes.csv"
# Columns where a value of 0 is biologically impossible -> treat 0 as missing.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    """Load the CSV and replace impossible zeros with each column's median."""
    df = pd.read_csv(path)
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, df[col].median())
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y


def cross_validate(name, estimator, X, y):
    """Stratified 5-fold cross-validation.

    The scaler lives inside the pipeline, so it is re-fit on each fold's
    training rows only -> no data leakage into the test fold.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"\n[{name}] fold AUCs: {scores.round(3)}")
    print(f"[{name}] mean AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
    return scores


def evaluate_on_holdout(name, estimator, X_train, X_test, y_train, y_test):
    """Fit on the training split and print a full report on the test split."""
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)[:, 1]

    print(f"\n===== {name}: held-out test set =====")
    print(classification_report(y_test, preds, digits=3))

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print("Confusion matrix:")
    print(f"  True Negatives : {tn:3d}    False Positives: {fp:3d}")
    print(f"  False Negatives: {fn:3d}    True Positives : {tp:3d}")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    return estimator


def main():
    X, y = load_data()
    print("Class balance (0 = healthy, 1 = diabetic):")
    print(y.value_counts())

    # 1) Cross-validation: the trustworthy, averaged estimate.
    cross_validate("LogisticRegression", LogisticRegression(max_iter=1000), X, y)
    cross_validate(
        "RandomForest",
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        X, y,
    )

    # 2) Single held-out split: detailed per-class report + confusion matrix.
    # stratify=y keeps the 65/35 class ratio in both train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = evaluate_on_holdout(
        "LogisticRegression", LogisticRegression(max_iter=1000),
        X_train_scaled, X_test_scaled, y_train, y_test,
    )
    evaluate_on_holdout(
        "RandomForest",
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        X_train_scaled, X_test_scaled, y_train, y_test,
    )

    # Save the deployed model + scaler (separate files, so app.py is unchanged).
    joblib.dump(logreg, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nSaved model.pkl and scaler.pkl")


if __name__ == "__main__":
    main()
