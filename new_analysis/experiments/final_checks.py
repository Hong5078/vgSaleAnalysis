"""Validity checks and one-time test evaluation for the selected model (E02/E03: HistGradientBoosting on log1p owners).

1. Grouped cross-validation: same protocol as the experiments but with folds grouped by normalised title,
   so re-releases or editions sharing a title cannot sit in both training and validation folds.
2. Test set: the selected model is fitted once on the full training portion and evaluated once on the
   held-out 20% test set, next to the notebook-03 baseline (Random Forest on the raw target).
3. Blockbuster analysis on the out-of-fold predictions of E02.
Outputs are written to results/final_checks_*.csv; nothing earlier is overwritten.
"""
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

import harness as H

RS = H.RANDOM_STATE
OUT = H.RESULTS_DIR
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

data, base = H.load_data()
hgb = HistGradientBoostingRegressor(random_state=RS)


def title_key(names):
    k = names.fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return k.where(k != "", "row" + pd.Series(range(len(k)), index=k.index).astype(str))


# ---------------------------------------------------------------- 1. grouped cross-validation
rows = []
for pop in ["A_all_games", "B_below_1M"]:
    train, _ = H.get_split(data, pop)
    groups = title_key(train["name"])
    for scheme, splitter in [("KFold (protocol)", KFold(5, shuffle=True, random_state=RS)),
                             ("GroupKFold by title", GroupKFold(5, shuffle=True, random_state=RS))]:
        fold_m = []
        for tr, va in splitter.split(train, groups=groups):
            m = H.fit_model(hgb, train.loc[tr, base], train.loc[tr, "owners"], "log1p")
            resid = np.log1p(train.loc[tr, "owners"].values) - m.predict(train.loc[tr, base])
            p, pl = H.predict(m, train.loc[va, base], "log1p", float(np.mean(np.exp(resid))))
            fold_m.append(H.metrics(train.loc[va, "owners"], p, pl))
        fm = pd.DataFrame(fold_m)
        rows.append({"population": pop, "folds": scheme, "R2_log": fm["R2_log"].mean(), "R2_log_sd": fm["R2_log"].std(),
                     "R2_orig (smearing)": fm["R2_orig"].mean(), "R2_orig_sd": fm["R2_orig"].std()})
grouped = pd.DataFrame(rows)
grouped.to_csv(os.path.join(OUT, "final_checks_grouped_cv.csv"), index=False)
print("1. Grouped cross-validation\n", grouped.round(4).to_string(index=False), "\n")

# ---------------------------------------------------------------- 2. one-time test-set evaluation
rows = []
for pop in ["A_all_games", "B_below_1M"]:
    train, test = H.get_split(data, pop)
    # selected model
    m = H.fit_model(hgb, train[base], train["owners"], "log1p")
    resid = np.log1p(train["owners"].values) - m.predict(train[base])
    smear = float(np.mean(np.exp(resid)))
    for label, factor in [("HistGB, log1p target (E02)", None), ("HistGB, log1p target + smearing (E03)", smear)]:
        p, pl = H.predict(m, test[base], "log1p", factor)
        rows.append({"population": pop, "model": label, "n_test": len(test), **H.metrics(test["owners"], p, pl)})
    # notebook baseline for reference
    rf = H.fit_model(RandomForestRegressor(random_state=RS, n_jobs=-1), train[base], train["owners"], "raw")
    p, pl = H.predict(rf, test[base], "raw")
    rows.append({"population": pop, "model": "Random Forest, raw target (notebook baseline)", "n_test": len(test),
                 **H.metrics(test["owners"], p, pl)})
    print(f"{pop}: smearing factor from full training portion = {smear:.3f}")
test_res = pd.DataFrame(rows)
test_res.to_csv(os.path.join(OUT, "final_checks_test_set.csv"), index=False)
print("2. Test set (used once)\n", test_res.round(4).to_string(index=False), "\n")

# ---------------------------------------------------------------- 3. blockbuster analysis (E02 out-of-fold)
oof = pd.read_csv(os.path.join(OUT, "E02__HistGB__A_all_games__oof.csv.gz"))
oof["actual_M"], oof["pred_M"] = oof["owners"] / 1e6, oof["pred_owners"] / 1e6
oof["sq_err"] = (oof["pred_M"] - oof["actual_M"]) ** 2
oof["log_err"] = oof["pred_log"] - np.log1p(oof["owners"])
order = oof.drop_duplicates("estimated_owners").sort_values("owners")["estimated_owners"]
by_range = oof.groupby("estimated_owners").agg(
    games=("owners", "size"), mean_actual_M=("actual_M", "mean"), median_pred_M=("pred_M", "median"),
    mean_log_error=("log_err", "mean"), share_sq_error=("sq_err", "sum")).reindex(order)
by_range["share_games"] = by_range["games"] / by_range["games"].sum()
by_range["share_sq_error"] = by_range["share_sq_error"] / by_range["share_sq_error"].sum()
by_range.to_csv(os.path.join(OUT, "final_checks_error_by_range.csv"))
print("3a. E02 out-of-fold errors by owner range\n", by_range.round(4).to_string(), "\n")

big = oof["owners"] >= 10_000_000
print(f"3b. Games >= 10M owners: {big.sum()} ({big.mean():.2%} of training games), "
      f"{oof.loc[big, 'sq_err'].sum() / oof['sq_err'].sum():.1%} of original-scale squared error")
worst = oof.assign(abs_err_M=(oof["pred_M"] - oof["actual_M"]).abs()).nlargest(12, "abs_err_M")
worst = worst[["name", "release_dt", "actual_M", "pred_M"]]
worst.to_csv(os.path.join(OUT, "final_checks_largest_errors.csv"), index=False)
print("3c. Largest absolute errors (out-of-fold)\n", worst.round(3).to_string(index=False))
