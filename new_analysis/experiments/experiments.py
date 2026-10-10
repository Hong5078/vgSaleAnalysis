"""Experiment definitions. Run one experiment with:  python experiments.py E01

Each experiment changes one major factor relative to the experiment it builds on. Hypotheses,
results and interpretations are recorded in EXPERIMENT_LOG.md; raw numbers in results/.
"""
import sys

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import harness as H

RS = H.RANDOM_STATE


def E00():
    """Reproduction check (not counted as an experiment): the harness must reproduce the notebook results."""
    data, base = H.load_data()
    models = {"Mean": DummyRegressor(), "LinReg": LinearRegression(),
              "DecTree": DecisionTreeRegressor(random_state=RS),
              "RandForest": RandomForestRegressor(random_state=RS, n_jobs=-1)}
    rows = []
    for pop in ["A_all_games", "B_below_1M"]:
        for name, m in models.items():
            rows.append(H.run("E00", name, m, data, base, "raw", pop, "base42", "sklearn defaults"))
    H.show(rows)


def E01():
    """Target transformation: log1p(owners) instead of owners in millions. Everything else as E00."""
    data, base = H.load_data()
    models = {"LinReg": LinearRegression(), "DecTree": DecisionTreeRegressor(random_state=RS),
              "RandForest": RandomForestRegressor(random_state=RS, n_jobs=-1)}
    rows = []
    for pop in ["A_all_games", "B_below_1M"]:
        for name, m in models.items():
            rows.append(H.run("E01", name, m, data, base, "log1p", pop, "base42", "sklearn defaults",
                              save_oof=(name == "RandForest" and pop == "A_all_games")))
    H.show(rows)


def E02():
    """Model family: HistGradientBoosting (defaults) on the E01 setting (log1p target, base features)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    data, base = H.load_data()
    rows = []
    for pop in ["A_all_games", "B_below_1M"]:
        rows.append(H.run("E02", "HistGB", HistGradientBoostingRegressor(random_state=RS), data, base, "log1p", pop,
                          "base42", "defaults: lr=0.1, max_iter=100, max_leaf_nodes=31, min_samples_leaf=20, "
                                    "early_stopping=auto", save_oof=(pop == "A_all_games")))
    H.show(rows)


def E03():
    """Back-transformation: Duan smearing correction on the E02 model (log-scale predictions unchanged)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    data, base = H.load_data()
    rows = []
    for pop in ["A_all_games", "B_below_1M"]:
        rows.append(H.run("E03", "HistGB_smearing", HistGradientBoostingRegressor(random_state=RS), data, base,
                          "log1p", pop, "base42", "E02 model; owners = expm1(pred) * mean(exp(training residuals))",
                          smearing=True))
    H.show(rows)


ENGINEERED = "engineered_features.csv.gz"   # written by notebook 07 (Section 12.1)


def _new_variable_step(exp_id, added, kept=()):
    """Add one group of new pre-release variables to the E02 setting (HistGB, log1p target) plus the
    groups already adopted. Requires notebook 07 to have written the engineered feature file."""
    import json
    import os
    from sklearn.ensemble import HistGradientBoostingRegressor
    data, base = H.load_data(ENGINEERED)
    with open(os.path.join(H.DATA_DIR, "engineered_features.json")) as f:
        groups = json.load(f)
    features = base + [c for g in kept for c in groups[g]] + groups[added]
    feature_set = "base42+" + "+".join(g[0] for g in (*kept, added))
    rows = [H.run(exp_id, "HistGB", HistGradientBoostingRegressor(random_state=RS), data, features, "log1p", pop,
                  feature_set, "defaults") for pop in ["A_all_games", "B_below_1M"]]
    rows.append(H.run(exp_id, "LinReg", LinearRegression(), data, features, "log1p", "A_all_games", feature_set,
                      "sklearn defaults"))
    H.show(rows)


A, B, C, D = "A. Studio experience", "B. Franchise and title", "C. Store-page lead time", "D. Store-page extras"


def E06():
    """New variables, group A (studio experience), on the E02 setting."""
    _new_variable_step("E06", A)


def E07():
    """New variables, group B (franchise and title), on top of E06."""
    _new_variable_step("E07", B, kept=(A,))


def E08():
    """New variables, group C (store-page lead time), on top of E06 (E07 was not adopted)."""
    _new_variable_step("E08", C, kept=(A,))


def E09():
    """New variables, group D (store-page extras), on top of E08."""
    _new_variable_step("E09", D, kept=(A, C))


def E10():
    """Sensitivity: past sales of the developer's and publisher's earlier games, on a chronological split of
    the training portion (fit on releases up to 2021, score on 2022 to June 2024). Not part of the main model:
    the past owner counts are a 2025 snapshot and include sales made after the predicted game's release."""
    import json
    import os
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    data, base = H.load_data(ENGINEERED)
    with open(os.path.join(H.DATA_DIR, "engineered_features.json")) as f:
        groups = json.load(f)
    adopted = base + groups[A] + groups[C]
    train, _ = H.get_split(data, "A_all_games")
    early, late = train[train["release_dt"] <= "2021-12-31"], train[train["release_dt"] > "2021-12-31"]
    rows = []
    for label, features in [("base42", base), ("base42+A+C", adopted),
                            ("base42+A+C+past_sales", adopted + groups["Past sales (sensitivity only)"])]:
        m = H.fit_model(HistGradientBoostingRegressor(random_state=RS), early[features], early["owners"], "log1p")
        p, pl = H.predict(m, late[features], "log1p")
        rows.append({"exp_id": "E10", "feature_set": label, "n_features": len(features), "n_fit": len(early),
                     "n_scored": len(late), **H.metrics(late["owners"], p, pl)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(H.RESULTS_DIR, "E10_chronological_past_sales.csv"), index=False)
    print(out.round(4).to_string(index=False))


def E11():
    """Forecasting later releases (E11-E14, run together). Chronological split of the training portion: fit on
    releases up to 2021, score on 2022 to June 2024; then each year 2016-2024 predicted from earlier years.
    E11 release year removed; E12 year-relative target; E13 fitted on 2017-2021 only; E14 year level from a
    five-year linear trend. Requires notebook 07's engineered feature file."""
    import json
    import os
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    data, base = H.load_data(ENGINEERED)
    with open(os.path.join(H.DATA_DIR, "engineered_features.json")) as f:
        groups = json.load(f)
    adopted = base + groups[A] + groups[C]
    no_year = [c for c in adopted if c != "release_year"]
    data["log_owners"] = np.log1p(data["owners"])
    train, _ = H.get_split(data, "A_all_games")
    early, late = train[train["release_dt"] <= "2021-12-31"], train[train["release_dt"] > "2021-12-31"]

    def forecast(features, fit, score, relative=False, level="last", seed=RS):
        model = HistGradientBoostingRegressor(random_state=seed)
        if not relative:
            return model.fit(fit[features], fit["log_owners"]).predict(score[features])
        year_mean = fit["log_owners"].groupby(fit["release_year"]).mean()
        pred = model.fit(fit[features], fit["log_owners"] - fit["release_year"].map(year_mean)).predict(score[features])
        if level == "last":
            return pred + year_mean.iloc[-1]
        slope, intercept = np.polyfit(year_mean.index[-5:], year_mean.iloc[-5:], 1)
        return pred + intercept + slope * score["release_year"].to_numpy()

    def evaluate(frame, pred):
        y, year = frame["log_owners"].reset_index(drop=True), frame["release_year"].to_numpy()
        pred = pd.Series(pred)
        within = r2_score(y - y.groupby(year).transform("mean"), pred - pred.groupby(year).transform("mean"))
        return {"R2_log": r2_score(y, pred), "within_year_R2_log": within, "bias_log": float(np.mean(pred - y))}

    rows = []
    for exp_id, features, fit, relative, level in [
            ("E08", adopted, early, False, None), ("E11", no_year, early, False, None),
            ("E12", adopted, early, True, "last"), ("E12+E11", no_year, early, True, "last"),
            ("E13", adopted, early[early["release_year"] >= 2017], False, None), ("E14", adopted, early, True, "trend")]:
        by_seed = [r2_score(late["log_owners"], forecast(features, fit, late, relative, level, s)) for s in (42, 1, 2, 3, 4)]
        rows.append({"exp_id": exp_id, "n_fit": len(fit), "n_scored": len(late),
                     **evaluate(late, forecast(features, fit, late, relative, level)),
                     "R2_log_mean_5_seeds": np.mean(by_seed), "R2_log_min": min(by_seed), "R2_log_max": max(by_seed)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(H.RESULTS_DIR, "E11_E14_chronological.csv"), index=False)
    print(out.round(4).to_string(index=False))

    rows = []
    for year in range(2016, 2025):
        earlier, target = train[train["release_year"] < year], train[train["release_year"] == year]
        a = evaluate(target, forecast(adopted, earlier, target))
        t = evaluate(target, forecast(adopted, earlier, target, True, "trend"))
        rows.append({"release_year": year, "n_fit": len(earlier), "n_scored": len(target), "ahead_R2_log": a["R2_log"],
                     "ahead_bias_log": a["bias_log"], "E14_R2_log": t["R2_log"], "E14_bias_log": t["bias_log"]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(H.RESULTS_DIR, "E11_E14_one_year_ahead.csv"), index=False)
    print(out.round(4).to_string(index=False))


E12 = E13 = E14 = E11


def E15():
    """Low group and breakout group (E15-E16, run together). E15: classify whether a game rises above the
    0-20,000 owner range. E16: two-stage prediction (the group first, then log owners among breakout games).
    Random folds and one year ahead (2022-2024 from earlier years). Requires notebook 07's feature file."""
    import json
    import os
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
    from sklearn.model_selection import KFold
    data, base = H.load_data(ENGINEERED)
    with open(os.path.join(H.DATA_DIR, "engineered_features.json")) as f:
        groups = json.load(f)
    features = base + groups[A] + groups[C]
    data["log_owners"] = np.log1p(data["owners"])
    data["breakout"] = (data["owners"] > 10_000).astype(int)
    low_value = np.log1p(10_000)
    train, _ = H.get_split(data, "A_all_games")
    folds = list(KFold(n_splits=H.N_FOLDS, shuffle=True, random_state=RS).split(train))

    def stages(fit, score, seed=RS):
        p = HistGradientBoostingClassifier(random_state=seed).fit(fit[features], fit["breakout"]) \
            .predict_proba(score[features])[:, 1]
        above = fit[fit["breakout"] == 1]
        s2 = HistGradientBoostingRegressor(random_state=seed).fit(above[features], above["log_owners"]).predict(score[features])
        single = HistGradientBoostingRegressor(random_state=seed).fit(fit[features], fit["log_owners"]).predict(score[features])
        return p, s2, single

    def ranking(y, score, year):
        aucs, sizes, hits, picked = [], [], 0, 0
        for v in np.unique(year):
            idx = np.where(year == v)[0]
            if len(idx) >= 200 and 0 < y[idx].mean() < 1:
                aucs.append(roc_auc_score(y[idx], score[idx]))
                sizes.append(len(idx))
                k = int(round(0.1 * len(idx)))
                hits += y[idx[np.argsort(-score[idx])[:k]]].sum()
                picked += k
        return {"base_rate": y.mean(), "roc_auc": roc_auc_score(y, score),
                "within_year_roc_auc": float(np.average(aucs, weights=sizes)), "precision_top10": hits / picked}

    def regression(frame, pred):
        return {"R2_log": r2_score(frame["log_owners"], pred), "MAE_log": mean_absolute_error(frame["log_owners"], pred)}

    p, s2, single = np.zeros(len(train)), np.zeros(len(train)), np.zeros(len(train))
    for tr, va in folds:
        p[va], s2[va], single[va] = stages(train.iloc[tr], train.iloc[va])
    y, year = train["breakout"].to_numpy(), train["release_year"].to_numpy()
    e15 = [{"scored": "random folds", "model": "HistGB classifier", **ranking(y, p, year)},
           {"scored": "random folds", "model": "E08 regression as ranker", **ranking(y, single, year)}]
    e16 = [{"scored": "random folds", "model": "single regression (E08)", **regression(train, single)},
           {"scored": "random folds", "model": "two-stage, expected value", **regression(train, p * s2 + (1 - p) * low_value)},
           {"scored": "random folds", "model": "two-stage, cut at 50%", **regression(train, np.where(p >= 0.5, s2, low_value))}]
    ahead = {"single regression (E08)": [], "two-stage, expected value": [], "two-stage, cut at 50%": []}
    for target_year in (2022, 2023, 2024):
        earlier, target = train[train["release_year"] < target_year], train[train["release_year"] == target_year]
        for seed in (42, 1, 2, 3, 4):
            p, s2, single = stages(earlier, target, seed)
            if seed == RS:
                e15.append({"scored": f"one year ahead, {target_year}", "model": "HistGB classifier",
                            **ranking(target["breakout"].to_numpy(), p, target["release_year"].to_numpy())})
            ahead["single regression (E08)"].append(regression(target, single))
            ahead["two-stage, expected value"].append(regression(target, p * s2 + (1 - p) * low_value))
            ahead["two-stage, cut at 50%"].append(regression(target, np.where(p >= 0.5, s2, low_value)))
    for model, rows in ahead.items():
        e16.append({"scored": "one year ahead, mean of 2022-2024 and 5 seeds", "model": model, **pd.DataFrame(rows).mean()})
    for name, rows in [("E15_breakout_classifier.csv", e15), ("E16_two_stage.csv", e16)]:
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(H.RESULTS_DIR, name), index=False)
        print(out.round(4).to_string(index=False))


E16 = E15


def E17():
    """Forecasting from the predecessor (E17-E19, run together), after Schaer, Kourentzes & Fildes (EJOR 2026).
    E17: the predecessor's owners alone (the paper's benchmark). E18: six predecessor variables added to the
    E08 model. E19: plus the four studio averages of E10. Random folds and one year ahead (2022-2024 from
    earlier years, five seeds). Requires the feature files written by notebooks 07 and 10."""
    import json
    import os
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold
    data, base = H.load_data(ENGINEERED)
    data = data.merge(pd.read_csv(os.path.join(H.DATA_DIR, "predecessor_features.csv.gz")), on="appid", how="left",
                      validate="one_to_one")
    with open(os.path.join(H.DATA_DIR, "engineered_features.json")) as f:
        groups = json.load(f)
    store_page = base + groups[A] + groups[C]
    predecessor = ["series_prev_log_owners", "series_gap_years", "dev_prev_log_owners", "dev_gap_years",
                   "pub_prev_log_owners", "pub_gap_years"]
    sets = {"E08": store_page, "E18": store_page + predecessor,
            "E19": store_page + predecessor + groups["Past sales (sensitivity only)"]}
    data["log_owners"] = np.log1p(data["owners"])
    train, _ = H.get_split(data, "A_all_games")
    folds = list(KFold(n_splits=H.N_FOLDS, shuffle=True, random_state=RS).split(train))
    rows = []
    for exp_id, features in sets.items():
        oof = np.zeros(len(train))
        for tr, va in folds:
            oof[va] = HistGradientBoostingRegressor(random_state=RS).fit(
                train.iloc[tr][features], train["log_owners"].to_numpy()[tr]).predict(train.iloc[va][features])
        ahead = []
        for year in (2022, 2023, 2024):
            earlier, target = train[train["release_year"] < year], train[train["release_year"] == year]
            for seed in (42, 1, 2, 3, 4):
                pred = HistGradientBoostingRegressor(random_state=seed).fit(earlier[features], earlier["log_owners"]) \
                    .predict(target[features])
                row = {"all": r2_score(target["log_owners"], pred), "MAE_log": mean_absolute_error(target["log_owners"], pred)}
                for kind, part in target.assign(pred=pred).groupby("analogy_type"):
                    row[kind] = r2_score(part["log_owners"], part["pred"])
                if exp_id == "E08" and seed == RS:
                    has = target["analogy_type"] != "No predecessor"
                    row["naive_MAE_log"] = mean_absolute_error(target.loc[has, "log_owners"], target.loc[has, "analogy_log_owners"])
                    row["E08_MAE_log_same_games"] = mean_absolute_error(target.loc[has, "log_owners"], pred[has.to_numpy()])
                ahead.append(row)
        a = pd.DataFrame(ahead).mean()
        rows.append({"exp_id": exp_id, "n_features": len(features), "random_folds_R2_log": r2_score(train["log_owners"], oof),
                     "ahead_R2_log": a["all"], "ahead_MAE_log": a["MAE_log"],
                     **{(k if k.endswith("MAE_log") or k.endswith("same_games") else f"ahead_R2_log | {k}"): a[k]
                        for k in a.index if k not in ("all", "MAE_log")}})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(H.RESULTS_DIR, "E17_E19_predecessor.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(out.round(4).T.to_string())


E18 = E19 = E17


if __name__ == "__main__":
    globals()[sys.argv[1]]()
