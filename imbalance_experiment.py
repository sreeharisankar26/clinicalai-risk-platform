"""Compare strategies for handling class imbalance on the diabetes dataset.

Baseline vs class_weight='balanced' vs SMOTE, all measured with the SAME
stratified 5-fold cross-validation so the comparison is fair. We report
recall (catching diabetics) alongside AUC and precision, because recall is
the metric this whole exercise is trying to improve.

Requires: pip install imbalanced-learn
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train import load_data, RANDOM_STATE

# One CV object reused everywhere -> identical folds for every strategy.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# recall/precision default to the positive class (label 1 = diabetic).
SCORING = ["roc_auc", "recall", "precision"]


def report(name, pipe, X, y):
    """Cross-validate one strategy and print its averaged metrics."""
    results = cross_validate(pipe, X, y, cv=CV, scoring=SCORING)
    auc = results["test_roc_auc"].mean()
    recall = results["test_recall"].mean()
    precision = results["test_precision"].mean()
    print(f"{name:<26} AUC={auc:.3f}   recall={recall:.3f}   precision={precision:.3f}")


def main():
    X, y = load_data()

    # 1) Baseline: no imbalance handling at all.
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])

    # 2) class_weight: penalise mistakes on the rare (diabetic) class more.
    weighted = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced")),
    ])

    # 3) SMOTE: synthesise new minority examples INSIDE each training fold only.
    # Using imblearn's Pipeline guarantees SMOTE never touches the test fold.
    smote = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ])

    print("Strategy comparison (5-fold stratified CV, positive class = diabetic):\n")
    report("Baseline (no handling)", baseline, X, y)
    report("class_weight=balanced", weighted, X, y)
    report("SMOTE", smote, X, y)


if __name__ == "__main__":
    main()
