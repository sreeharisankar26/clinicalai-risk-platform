"""Explain model predictions with SHAP.

SHAP gives each feature a contribution to each individual prediction, showing
how much it pushed the diabetes risk up or down. Two views are produced:
  1. summary (global): which features matter most across ALL patients
  2. per-patient (local): why the model scored one specific patient

Requires: pip install shap
"""

import matplotlib
matplotlib.use("Agg")  # render plots to files without needing a display window

import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from train import load_data, RANDOM_STATE


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Tree models pair with SHAP's fast, exact TreeExplainer. No scaling needed:
    # trees split on raw thresholds, so scaling would not change the tree at all.
    model = RandomForestClassifier(n_estimators=200, max_depth=5,
                                   random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)          # contributions for every test patient
    sv = shap_values[..., 1]                 # keep the "diabetic" class (index 1)

    # 1) GLOBAL view: feature importance across all patients.
    shap.plots.beeswarm(sv, show=False)
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_summary.png")

    # 2) LOCAL view: explain one specific patient in plain numbers.
    i = 0
    print(f"\nExplaining patient #{i}:")
    print(X_test.iloc[i].to_string())
    contributions = dict(zip(X_test.columns, sv.values[i]))
    print("\nFeature contributions (sorted by impact):")
    for feat, val in sorted(contributions.items(), key=lambda kv: -abs(kv[1])):
        direction = "raises risk" if val > 0 else "lowers risk"
        print(f"  {feat:<26} {val:+.3f}  ({direction})")

    shap.plots.waterfall(sv[i], show=False)
    plt.savefig("shap_patient0.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved shap_patient0.png")


if __name__ == "__main__":
    main()
