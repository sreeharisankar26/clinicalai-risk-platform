"""Hyperparameter tuning with Optuna, with every trial logged to MLflow.

Optuna proposes hyperparameter values and uses the history of past trials to
pick smarter values next time (Bayesian optimisation). Each trial is scored
with the SAME stratified 5-fold CV (mean AUC) and recorded as an MLflow run,
so all experiments live in a browsable dashboard instead of scrolling away.

Requires: pip install optuna xgboost mlflow
View results afterwards with:  mlflow ui   (then open http://localhost:5000)
"""

import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from train import load_data, RANDOM_STATE

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# Modern MLflow needs a database backend; log to a local SQLite file.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes-tuning")  # groups all runs under one project


def score_model(model, X, y):
    """Mean 5-fold AUC for a model, scaling kept inside the pipeline."""
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    return cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc").mean()


def rf_objective(trial, X, y):
    """One Random Forest trial, logged as an MLflow run."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    model = RandomForestClassifier(**params, random_state=RANDOM_STATE)
    auc = score_model(model, X, y)
    with mlflow.start_run(run_name="RandomForest-trial"):
        mlflow.set_tag("model", "RandomForest")
        mlflow.log_params(params)
        mlflow.log_metric("cv_auc", auc)
    return auc


def xgb_objective(trial, X, y):
    """One XGBoost trial, logged as an MLflow run."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    model = XGBClassifier(**params, eval_metric="logloss", random_state=RANDOM_STATE)
    auc = score_model(model, X, y)
    with mlflow.start_run(run_name="XGBoost-trial"):
        mlflow.set_tag("model", "XGBoost")
        mlflow.log_params(params)
        mlflow.log_metric("cv_auc", auc)
    return auc


def tune(name, objective, X, y, n_trials=40):
    study = optuna.create_study(direction="maximize")  # we want the HIGHEST AUC
    study.optimize(lambda t: objective(t, X, y), n_trials=n_trials)
    print(f"\n{name}: best AUC = {study.best_value:.3f}")
    print(f"{name}: best params = {study.best_params}")
    return study


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # hide per-trial spam
    X, y = load_data()
    tune("RandomForest", rf_objective, X, y)
    tune("XGBoost", xgb_objective, X, y)
    print("\nDone. Run 'mlflow ui' and open http://localhost:5000 to compare runs.")


if __name__ == "__main__":
    main()
