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


if __name__ == "__main__":
    globals()[sys.argv[1]]()
