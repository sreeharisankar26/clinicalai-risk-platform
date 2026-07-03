"""Data drift detection with Evidently.

A deployed model is trained on one snapshot of the world. In production the
incoming data can drift - different patients, new equipment, seasonal shifts -
and the model silently degrades. This script compares a REFERENCE batch (the
training distribution) against a CURRENT batch (simulated incoming data) and
generates an HTML drift report.

Requires: pip install evidently
"""

from evidently import Report
from evidently.presets import DataDriftPreset

from train import load_data


def main():
    X, _ = load_data()

    # Reference = the distribution the model was trained on.
    reference = X.iloc[:500].reset_index(drop=True)

    # Current = a simulated incoming production batch. We deliberately shift
    # Glucose and Age upward to mimic an older, higher-risk population arriving,
    # so the report clearly flags drift on those features.
    current = X.iloc[500:].reset_index(drop=True).copy()
    current["Glucose"] = current["Glucose"] + 25
    current["Age"] = current["Age"] + 10

    report = Report([DataDriftPreset()])
    result = report.run(reference, current)  # newer Evidently returns a snapshot

    # save_html lives on the run result in newer versions, on the report in older.
    target = result if hasattr(result, "save_html") else report
    target.save_html("drift_report.html")
    print("Saved drift_report.html - open it in your browser.")


if __name__ == "__main__":
    main()
