"""Train the production model on ALL data and save it for serving.

We serve a Random Forest: its performance is within noise of the alternatives,
and it enables exact SHAP explanations via TreeExplainer. Trees split on raw
feature thresholds, so no scaler is needed.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier

from train import load_data, RANDOM_STATE


def main():
    X, y = load_data()
    model = RandomForestClassifier(n_estimators=200, max_depth=5,
                                   random_state=RANDOM_STATE)
    model.fit(X, y)  # fit on the full dataset for the final deployed model
    joblib.dump(model, "model.pkl")
    joblib.dump(list(X.columns), "features.pkl")  # remember exact column order
    print("Saved model.pkl and features.pkl")


if __name__ == "__main__":
    main()
