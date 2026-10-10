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


if __name__ == "__main__":
    globals()[sys.argv[1]]()
