"""Shared experiment harness.

Every experiment uses the same protocol so results are comparable:
- the same 80/20 train/test split as notebooks 03 and 04 (train_test_split, random_state=42);
- 5-fold KFold(shuffle=True, random_state=42) cross-validation on the training portion only;
- the test portion is not used for any experiment; it is reserved for one final check;
- metrics on two scales:
    * log scale:      log1p(owners)            -> R2_log, MAE_log
    * original scale: owners in millions       -> R2_orig, MAE_orig, RMSE_orig
  Models fitted on the raw target are converted to the log scale with log1p(max(prediction, 0));
  models fitted on log1p(owners) are converted back with expm1(prediction).

Populations (separate research questions, never pooled in comparisons):
- "A_all_games": every labeled game (the population a real forecast must serve);
- "B_below_1M":  games with fewer than 1M owners in both training and test data (diagnostic only,
                 because a game's popularity is unknown before release).

Results are appended to results/experiment_log.csv and never overwritten.
"""
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data", "processed"))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_CSV = os.path.join(RESULTS_DIR, "experiment_log.csv")

RANDOM_STATE = 42
N_FOLDS = 5
POPULATIONS = {"A_all_games": None, "B_below_1M": 1_000_000}


def load_data(engineered_file=None):
    """Load the modeling table from notebook 01 (row order preserved) and optional engineered features."""
    data = pd.read_csv(os.path.join(DATA_DIR, "steam_model_table.csv.gz"), parse_dates=["release_dt"])
    with open(os.path.join(DATA_DIR, "features.json")) as f:
        spec = json.load(f)
    base_features = spec["numeric"] + spec["genres"] + spec["categories"]
    data["owners"] = (data["owners_millions"] * 1e6).round()
    if engineered_file:
        eng = pd.read_csv(os.path.join(DATA_DIR, engineered_file))
        n = len(data)
        data = data.merge(eng, on="appid", how="left", validate="one_to_one")  # left merge keeps row order
        assert len(data) == n
    return data, base_features


def get_split(data, population):
    """Same split as the notebooks; a population filter is applied after splitting (as in notebook 04)."""
    train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    threshold = POPULATIONS[population]
    if threshold is not None:
        train, test = train[train["owners"] < threshold], test[test["owners"] < threshold]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def metrics(y_owners, pred_owners, pred_log=None):
    y_owners = np.asarray(y_owners, dtype=float)
    pred_owners = np.asarray(pred_owners, dtype=float)
    y_log = np.log1p(y_owners)
    if pred_log is None:
        pred_log = np.log1p(np.clip(pred_owners, 0, None))
    y_m, p_m = y_owners / 1e6, pred_owners / 1e6
    return {
        "R2_log": r2_score(y_log, pred_log),
        "MAE_log": mean_absolute_error(y_log, pred_log),
        "R2_orig": r2_score(y_m, p_m),
        "MAE_orig": mean_absolute_error(y_m, p_m),
        "RMSE_orig": mean_squared_error(y_m, p_m) ** 0.5,
    }


def fit_model(model, X, y_owners, target):
    if target == "raw":        # owners in millions, exactly as in the baseline notebooks
        return clone(model).fit(X, np.asarray(y_owners) / 1e6)
    if target == "log1p":
        return clone(model).fit(X, np.log1p(np.asarray(y_owners)))
    raise ValueError(target)


def predict(fitted, X, target, smearing=None):
    """Return (predictions in owners, predictions on the log scale)."""
    p = fitted.predict(X)
    if target == "raw":
        owners = p * 1e6
        return owners, np.log1p(np.clip(owners, 0, None))
    log_pred = p
    owners = np.expm1(log_pred)
    if smearing is not None:
        owners = owners * smearing
    return owners, log_pred


def cross_validate(model, data, features, target, population, smearing=False):
    """5-fold CV on the training portion. Returns per-fold validation metrics, per-fold training
    metrics (overfitting diagnostic only) and out-of-fold predictions."""
    train, _ = get_split(data, population)
    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(train))
    val_rows, trn_rows = [], []
    oof_owners, oof_log = np.zeros(len(train)), np.zeros(len(train))
    for tr, va in folds:
        X_tr, y_tr = train.loc[tr, features], train.loc[tr, "owners"]
        fitted = fit_model(model, X_tr, y_tr, target)
        factor = None
        if smearing and target == "log1p":
            # Duan's smearing estimator, computed from training-fold residuals only
            resid = np.log1p(y_tr.values) - fitted.predict(X_tr)
            factor = float(np.mean(np.exp(resid)))
        p_tr, pl_tr = predict(fitted, X_tr, target, factor)
        p_va, pl_va = predict(fitted, train.loc[va, features], target, factor)
        oof_owners[va], oof_log[va] = p_va, pl_va
        val_rows.append(metrics(train.loc[va, "owners"], p_va, pl_va))
        trn_rows.append(metrics(y_tr, p_tr, pl_tr))
    return pd.DataFrame(val_rows), pd.DataFrame(trn_rows), oof_owners, oof_log, train


def run(exp_id, label, model, data, features, target, population, feature_set, params="", smearing=False,
        save_oof=False):
    t0 = time.time()
    val, trn, oof_owners, oof_log, train = cross_validate(model, data, features, target, population, smearing)
    row = {
        "exp_id": exp_id, "label": label, "population": population, "n_train": len(train),
        "target": target + ("+smearing" if smearing else ""), "feature_set": feature_set,
        "n_features": len(features), "model": type(model).__name__, "params": params,
        "cv_R2_log_mean": val["R2_log"].mean(), "cv_R2_log_sd": val["R2_log"].std(),
        "cv_R2_orig_mean": val["R2_orig"].mean(), "cv_R2_orig_sd": val["R2_orig"].std(),
        "cv_MAE_orig": val["MAE_orig"].mean(), "cv_RMSE_orig": val["RMSE_orig"].mean(),
        "cv_MAE_log": val["MAE_log"].mean(),
        "train_R2_log": trn["R2_log"].mean(), "train_R2_orig": trn["R2_orig"].mean(),
        "seconds": round(time.time() - t0, 1), "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    header = not os.path.exists(LOG_CSV)
    pd.DataFrame([row]).to_csv(LOG_CSV, mode="a", header=header, index=False)
    detail = {"row": row, "folds_validation": val.to_dict(orient="list"), "folds_training": trn.to_dict(orient="list"),
              "features": features}
    stem = f"{exp_id}__{label}__{population}".replace(" ", "_")
    path = os.path.join(RESULTS_DIR, stem + ".json")
    k = 2
    while os.path.exists(path):  # never overwrite an earlier run
        path = os.path.join(RESULTS_DIR, f"{stem}__run{k}.json")
        k += 1
    with open(path, "w") as f:
        json.dump(detail, f, indent=1, default=float)
    if save_oof:
        out = train[["appid", "name", "release_dt", "estimated_owners", "owners"]].copy()
        out["pred_owners"], out["pred_log"] = oof_owners, oof_log
        out.to_csv(os.path.join(RESULTS_DIR, stem + "__oof.csv.gz"), index=False)
    return row


def show(rows):
    cols = ["exp_id", "label", "population", "target", "n_features", "cv_R2_log_mean", "cv_R2_log_sd",
            "cv_R2_orig_mean", "cv_R2_orig_sd", "cv_MAE_orig", "cv_RMSE_orig", "train_R2_log", "seconds"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(pd.DataFrame(rows)[cols].round(4).to_string(index=False))
