"""Interpretation and error analysis of the best model (E02). No model changes, no new model inputs.

Model:   HistGradientBoostingRegressor(random_state=42), scikit-learn defaults
Target:  log1p(owners)            Features: the 42 release-time features of notebook 01
Split:   the harness split (80/20, seed 42); 5-fold KFold (seed 42) on the training portion

Part 1  Permutation importance (R² on log1p owners) on held-out data:
        primary  = the 5 cross-validation validation folds (each scored by the model that did not see it)
        check    = the test set, scored by the model fitted on the full training portion
Part 2  Per-game test-set predictions and the largest errors
Part 3  Error patterns by owner tier, release year, price, business model, genre/category, play mode,
        blockbuster status, and by diagnostic-only variables (reviews, rating, developer/publisher) that
        are NOT model inputs
Part 4  Diagnostic plots
Outputs go to interpretation/ (a new run folder is created if it already exists).
"""
import json
import os

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold

import harness as H

RS = H.RANDOM_STATE
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

OUT = os.path.join(H.HERE, "interpretation")
k = 2
while os.path.exists(OUT) and os.listdir(OUT):  # never overwrite an earlier run
    OUT = os.path.join(H.HERE, f"interpretation_run{k}")
    k += 1
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
BLUE, GREEN, ORANGE, RED, GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8C8C8C"

data, FEATURES = H.load_data()
train, test = H.get_split(data, "A_all_games")
model = HistGradientBoostingRegressor(random_state=RS)
y_tr_log, y_te_log = np.log1p(train["owners"].values), np.log1p(test["owners"].values)

FEATURE_GROUPS = {
    "Price / business model": ["price", "genre_Free_To_Play"],
    "Release year": ["release_year"],
    "Localisation": ["n_languages", "n_audio_languages"],
    "Platform support (Mac/Linux)": ["mac", "linux"],
    "Store-page presentation": ["n_screenshots", "description_length"],
    "Age rating & content descriptors": ["required_age", "genre_Violent", "genre_Gore", "genre_Nudity"],
    "Game genres": ["genre_Indie", "genre_Casual", "genre_Action", "genre_Adventure", "genre_Simulation",
                    "genre_Strategy", "genre_RPG", "genre_Sports", "genre_Racing", "genre_Massively_Multiplayer",
                    "genre_Education"],
    "Play mode (single/multiplayer)": ["cat_Single_player", "cat_Multi_player", "cat_PvP", "cat_Co_op",
                                       "cat_Online_PvP", "cat_Online_Co_op", "cat_Shared_Split_Screen",
                                       "cat_Shared_Split_Screen_PvP", "cat_Shared_Split_Screen_Co_op"],
    "Controller & VR support": ["cat_Full_controller_support", "cat_Partial_Controller_Support",
                                "cat_Tracked_Controller_Support", "cat_VR_Only"],
    "Steam platform features": ["cat_Steam_Cloud", "cat_Steam_Leaderboards", "cat_Stats",
                                "cat_Includes_level_editor", "cat_Captions_available"],
}
assert sorted(c for g in FEATURE_GROUPS.values() for c in g) == sorted(FEATURES)
GROUP_OF = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}


def grouped_permutation(fitted, X, y, n_repeats, seed):
    rng = np.random.default_rng(seed)
    base_score = r2_score(y, fitted.predict(X))
    out = {}
    for g, cols in FEATURE_GROUPS.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]
            drops.append(base_score - r2_score(y, fitted.predict(Xp)))
        out[g] = drops
    return out


# =============================================================== Part 1: permutation importance
folds = list(KFold(n_splits=H.N_FOLDS, shuffle=True, random_state=RS).split(train))
per_feature = {c: [] for c in FEATURES}
per_group = {g: [] for g in FEATURE_GROUPS}
oof_log = np.zeros(len(train))
fold_r2 = []
for i, (tr, va) in enumerate(folds):
    m = clone(model).fit(train.loc[tr, FEATURES], y_tr_log[tr])
    X_va, y_va = train.loc[va, FEATURES], y_tr_log[va]
    oof_log[va] = m.predict(X_va)
    fold_r2.append(r2_score(y_va, oof_log[va]))
    pi = permutation_importance(m, X_va, y_va, scoring="r2", n_repeats=5, random_state=RS + i, n_jobs=1)
    for j, c in enumerate(FEATURES):
        per_feature[c].extend(pi.importances[j])
    for g, drops in grouped_permutation(m, X_va, y_va, 5, RS + i).items():
        per_group[g].extend(drops)
print(f"CV R2_log per fold: {np.round(fold_r2, 4)} mean {np.mean(fold_r2):.4f} (E02 reported 0.4367)")

final = clone(model).fit(train[FEATURES], y_tr_log)
pred_te_log = final.predict(test[FEATURES])
print(f"Test R2_log: {r2_score(y_te_log, pred_te_log):.4f} (final_checks reported 0.4255)")
pi_test = permutation_importance(final, test[FEATURES], y_te_log, scoring="r2", n_repeats=10, random_state=RS, n_jobs=1)
group_test = grouped_permutation(final, test[FEATURES], y_te_log, 10, RS)

imp = pd.DataFrame({
    "feature": FEATURES,
    "group": [GROUP_OF[c] for c in FEATURES],
    "importance_mean": [np.mean(per_feature[c]) for c in FEATURES],
    "importance_sd": [np.std(per_feature[c], ddof=1) for c in FEATURES],
    "test_importance_mean": pi_test.importances_mean,
}).sort_values("importance_mean", ascending=False)
imp["rank"] = np.arange(1, len(imp) + 1)
imp["test_rank"] = imp["test_importance_mean"].rank(ascending=False).astype(int)
imp = imp[["rank", "feature", "group", "importance_mean", "importance_sd", "test_importance_mean", "test_rank"]]
imp.to_csv(os.path.join(TAB, "permutation_importance_features.csv"), index=False)

gimp = pd.DataFrame({
    "group": list(FEATURE_GROUPS),
    "n_features": [len(v) for v in FEATURE_GROUPS.values()],
    "importance_mean": [np.mean(per_group[g]) for g in FEATURE_GROUPS],
    "importance_sd": [np.std(per_group[g], ddof=1) for g in FEATURE_GROUPS],
    "test_importance_mean": [np.mean(group_test[g]) for g in FEATURE_GROUPS],
}).sort_values("importance_mean", ascending=False)
gimp["rank"] = np.arange(1, len(gimp) + 1)
gimp.to_csv(os.path.join(TAB, "permutation_importance_groups.csv"), index=False)
print("\nTop 15 features (CV folds; importance = drop in R2_log)\n", imp.head(15).round(4).to_string(index=False))
print("\nFeature groups\n", gimp.round(4).to_string(index=False))
rank_corr = imp[["importance_mean", "test_importance_mean"]].corr(method="spearman").iloc[0, 1]
print(f"\nSpearman rank correlation, CV vs test importance: {rank_corr:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
top = imp.head(15).iloc[::-1]
axes[0].barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], color=GREEN, capsize=3)
axes[0].set_xlabel("drop in R² (log1p owners) when the feature is shuffled")
axes[0].set_title("(a) Top 15 features (5 CV folds × 5 repeats, mean ± SD)")
g = gimp.iloc[::-1]
axes[1].barh(g["group"], g["importance_mean"], xerr=g["importance_sd"], color=BLUE, capsize=3)
axes[1].set_xlabel("drop in R² (log1p owners) when the whole group is shuffled")
axes[1].set_title("(b) Feature groups shuffled jointly")
fig.suptitle("Permutation importance of the best model (HistGradientBoosting, log1p owners)", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_a_permutation_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

# =============================================================== Part 2: test-set predictions
raw_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
with open(os.path.join(raw_path, "games.json"), encoding="utf-8") as f:
    raw = json.load(f)
diag = pd.DataFrame([{
    "appid": int(a), "positive": v.get("positive") or 0, "negative": v.get("negative") or 0,
    "developer": (v.get("developers") or [None])[0], "publisher": (v.get("publishers") or [None])[0],
    "self_published": int(bool(v.get("developers")) and set(v.get("developers") or []) == set(v.get("publishers") or [])),
    "release_dt_raw": v.get("release_date"),
} for a, v in raw.items()])
del raw
diag["release_dt_raw"] = pd.to_datetime(diag["release_dt_raw"], errors="coerce", format="mixed")


def prior_releases(key):
    # releases by the same developer/publisher strictly before each game's release date (all apps, labeled or not)
    per_day = diag.dropna(subset=[key, "release_dt_raw"]).groupby([key, "release_dt_raw"]).size().rename("n").reset_index()
    per_day = per_day.sort_values([key, "release_dt_raw"])
    per_day["prior"] = per_day.groupby(key)["n"].cumsum() - per_day["n"]
    out = diag[["appid", key, "release_dt_raw"]].merge(per_day[[key, "release_dt_raw", "prior"]],
                                                       on=[key, "release_dt_raw"], how="left")
    return out.set_index("appid")["prior"].fillna(0)


diag["dev_prior_releases"] = diag["appid"].map(prior_releases("developer"))
diag["pub_prior_releases"] = diag["appid"].map(prior_releases("publisher"))
diag["reviews"] = diag["positive"] + diag["negative"]
diag["rating"] = np.where(diag["reviews"] >= 10, diag["positive"] / diag["reviews"].replace(0, np.nan), np.nan)
DIAG_COLS = ["reviews", "rating", "developer", "publisher", "self_published", "dev_prior_releases", "pub_prior_releases"]


def build_frame(frame, pred_log):
    df = frame[["appid", "name", "release_dt", "estimated_owners", "owners"] + FEATURES].copy()
    df["true_log"] = np.log1p(df["owners"])
    df["pred_log"] = pred_log
    df["pred_owners"] = np.expm1(df["pred_log"])
    df["resid_log"] = df["pred_log"] - df["true_log"]            # > 0: over-prediction
    df["abs_log_err"] = df["resid_log"].abs()
    df["abs_err_owners"] = (df["pred_owners"] - df["owners"]).abs()
    df["abs_err_M"] = df["abs_err_owners"] / 1e6
    df["pct_error"] = 100 * (df["pred_owners"] - df["owners"]) / df["owners"]
    df["fold_error"] = np.maximum(df["pred_owners"], df["owners"]) / np.minimum(df["pred_owners"], df["owners"])
    return df.merge(diag[["appid"] + DIAG_COLS], on="appid", how="left")


te = build_frame(test, pred_te_log)
oof = build_frame(train, oof_log)
pred_cols = ["appid", "name", "release_dt", "estimated_owners", "owners", "pred_owners", "true_log", "pred_log",
             "resid_log", "abs_log_err", "abs_err_owners", "pct_error", "fold_error"]
te[pred_cols + ["price", "release_year", "n_languages", "genre_Free_To_Play"] + DIAG_COLS] \
    .to_csv(os.path.join(TAB, "test_predictions.csv.gz"), index=False)

lists = {
    "largest_log_errors": te.nlargest(20, "abs_log_err"),
    "largest_owner_errors": te.nlargest(20, "abs_err_owners"),
    "largest_overpredictions": te[te["resid_log"] > 0].nlargest(20, "resid_log"),
    "largest_underpredictions": te[te["resid_log"] < 0].nsmallest(20, "resid_log"),
}
for name, frame in lists.items():
    frame[pred_cols + ["price", "n_languages", "genre_Free_To_Play", "reviews", "rating", "publisher"]] \
        .to_csv(os.path.join(TAB, f"test_{name}.csv"), index=False)
show = ["name", "release_dt", "estimated_owners", "pred_owners", "resid_log", "fold_error", "price", "n_languages", "reviews", "rating"]
for name, frame in lists.items():
    print(f"\n{name}\n", frame[show].head(10).round(2).to_string(index=False))

# =============================================================== Part 3: error patterns
TIER_BINS = [0, 20_000, 100_000, 1_000_000, 10_000_000, np.inf]
TIER_LABELS = ["<20k", "20k–100k", "100k–1M", "1M–10M", "10M+"]


def add_groups(df):
    df["owner_tier"] = pd.cut(df["owners"], TIER_BINS, labels=TIER_LABELS, right=False)
    df["year_band"] = pd.cut(df["release_year"], [0, 2013, 2016, 2019, 2022, 2030],
                             labels=["≤2013", "2014–2016", "2017–2019", "2020–2022", "2023–2024"])
    df["price_band"] = pd.cut(df["price"], [-0.01, 0, 5, 10, 20, 40, 10_000],
                              labels=["free", "$0.01–5", "$5–10", "$10–20", "$20–40", ">$40"])
    df["business_model"] = np.select([df["genre_Free_To_Play"] == 1, df["price"] == 0],
                                     ["free-to-play (genre)", "free, no F2P genre"], "paid")
    mp_cols = ["cat_Multi_player", "cat_PvP", "cat_Co_op", "cat_Online_PvP", "cat_Online_Co_op",
               "cat_Shared_Split_Screen", "cat_Shared_Split_Screen_PvP", "cat_Shared_Split_Screen_Co_op"]
    mp = df[mp_cols].max(axis=1) == 1
    online = df[["cat_Online_PvP", "cat_Online_Co_op", "genre_Massively_Multiplayer"]].max(axis=1) == 1
    df["play_mode"] = np.select([online, mp, df["cat_Single_player"] == 1],
                                ["online multiplayer / MMO", "local or other multiplayer", "single-player only"], "unspecified")
    df["blockbuster"] = np.where(df["owners"] >= 10_000_000, "10M+ owners", "below 10M")
    df["review_band"] = pd.cut(df["reviews"], [-1, 0, 9, 99, 999, 9_999, np.inf],
                               labels=["0", "1–9", "10–99", "100–999", "1k–9.9k", "10k+"])
    df["rating_band"] = pd.cut(df["rating"], [0, 0.5, 0.7, 0.8, 0.9, 1.0001],
                               labels=["<50%", "50–70%", "70–80%", "80–90%", "≥90%"], right=False)
    df["publisher_type"] = np.where(df["self_published"] == 1, "self-published", "external publisher")
    df["pub_experience"] = pd.cut(df["pub_prior_releases"], [-1, 0, 4, 19, np.inf],
                                  labels=["first release", "1–4 prior", "5–19 prior", "20+ prior"])
    return df


te, oof = add_groups(te), add_groups(oof)


def group_stats(df, key):
    g = df.groupby(key, observed=True)
    return pd.DataFrame({
        "games": g.size(),
        "MAE_log": g["abs_log_err"].mean(),
        "RMSE_log": g["resid_log"].apply(lambda s: np.sqrt(np.mean(s ** 2))),
        "mean_bias_log": g["resid_log"].mean(),
        "median_fold_error": g["fold_error"].median(),
        "MAE_orig_M": g["abs_err_M"].mean(),
        "share_of_sq_error_orig": g["abs_err_M"].apply(lambda s: np.sum(s ** 2)) / np.sum(df["abs_err_M"] ** 2),
    })


groupings = {
    "owner_tier": "Owner tier (actual)", "estimated_owners": "Owner range (actual)", "year_band": "Release year",
    "price_band": "Price", "business_model": "Business model", "play_mode": "Play mode",
    "blockbuster": "Blockbuster status", "review_band": "Review count (diagnostic only, post-release)",
    "rating_band": "User rating, games with ≥10 reviews (diagnostic only, post-release)",
    "publisher_type": "Publisher (diagnostic only)", "pub_experience": "Publisher prior releases (diagnostic only)",
}
all_stats = []
for key, title in groupings.items():
    s = group_stats(te, key)
    if key == "estimated_owners":
        s = s.reindex(te.drop_duplicates("estimated_owners").sort_values("owners")["estimated_owners"])
    s.insert(0, "level", s.index.astype(str))
    s.insert(0, "grouping", title)
    all_stats.append(s.reset_index(drop=True))
    print(f"\n{title}\n", s.drop(columns=["grouping"]).round(3).to_string(index=False))

flag_rows = []
for c in FEATURES:
    if c.startswith(("genre_", "cat_")):
        sub = te[te[c] == 1]
        flag_rows.append({"grouping": "Genre/category flag = 1", "level": c, "games": len(sub),
                          "MAE_log": sub["abs_log_err"].mean(), "RMSE_log": np.sqrt(np.mean(sub["resid_log"] ** 2)),
                          "mean_bias_log": sub["resid_log"].mean(), "median_fold_error": sub["fold_error"].median(),
                          "MAE_orig_M": sub["abs_err_M"].mean(),
                          "share_of_sq_error_orig": np.sum(sub["abs_err_M"] ** 2) / np.sum(te["abs_err_M"] ** 2)})
flags = pd.DataFrame(flag_rows).sort_values("MAE_log", ascending=False)
print("\nGenre/category flags (test)\n", flags.drop(columns=["grouping"]).round(3).to_string(index=False))
pd.concat(all_stats + [flags], ignore_index=True).to_csv(os.path.join(TAB, "error_by_group_test.csv"), index=False)

# replication of the main patterns on the out-of-fold predictions (training portion, 61k games)
rep = []
for key in ["owner_tier", "year_band", "business_model", "play_mode", "price_band"]:
    a, b = group_stats(te, key), group_stats(oof, key)
    rep.append(pd.DataFrame({"grouping": key, "level": a.index.astype(str),
                             "test_games": a["games"].values, "test_bias": a["mean_bias_log"].values,
                             "test_MAE_log": a["MAE_log"].values,
                             "oof_games": b.reindex(a.index)["games"].values,
                             "oof_bias": b.reindex(a.index)["mean_bias_log"].values,
                             "oof_MAE_log": b.reindex(a.index)["MAE_log"].values}))
rep = pd.concat(rep, ignore_index=True)
rep.to_csv(os.path.join(TAB, "pattern_replication_test_vs_oof.csv"), index=False)
print("\nPattern replication (test vs out-of-fold)\n", rep.round(3).to_string(index=False))

# =============================================================== blockbusters (all 86: 66 out-of-fold + 20 test)
both = pd.concat([oof.assign(source="out-of-fold"), te.assign(source="test")], ignore_index=True)
both["pred_percentile"] = both["pred_log"].rank(pct=True)
mp_share = lambda s: (s != "single-player only") & (s != "unspecified")
profile = both.groupby("owner_tier", observed=True).agg(
    games=("owners", "size"),
    median_pred_owners=("pred_owners", "median"),
    median_pred_percentile=("pred_percentile", "median"),
    share_free_to_play=("genre_Free_To_Play", "mean"),
    share_MMO=("genre_Massively_Multiplayer", "mean"),
    share_online_multiplayer=("play_mode", lambda s: (s == "online multiplayer / MMO").mean()),
    share_price_20_plus=("price", lambda s: (s >= 20).mean()),
    median_languages=("n_languages", "median"),
    median_audio_languages=("n_audio_languages", "median"),
    median_screenshots=("n_screenshots", "median"),
    median_release_year=("release_year", "median"),
    share_controller_full=("cat_Full_controller_support", "mean"),
    share_self_published=("self_published", "mean"),
    median_pub_prior_releases=("pub_prior_releases", "median"),
)
profile.to_csv(os.path.join(TAB, "blockbuster_profile_by_tier.csv"))
print("\nProfile by owner tier (out-of-fold + test predictions; last two columns diagnostic only)\n",
      profile.round(3).T.to_string())

is_bb = (both["owners"] >= 10_000_000).astype(int)
auc = roc_auc_score(is_bb, both["pred_log"])
top_rows = []
for q in (0.001, 0.01, 0.05):
    cut = both["pred_log"].quantile(1 - q)
    sel = both[both["pred_log"] >= cut]
    top_rows.append({"top share of predictions": q, "games": len(sel), "10M+ games in selection": int((sel["owners"] >= 1e7).sum()),
                     "share 10M+": (sel["owners"] >= 1e7).mean(), "share of all 10M+ captured": (sel["owners"] >= 1e7).sum() / is_bb.sum(),
                     "median actual owners": sel["owners"].median(), "share below 1M": (sel["owners"] < 1e6).mean()})
top_tbl = pd.DataFrame(top_rows)
top_tbl.to_csv(os.path.join(TAB, "blockbuster_ranking.csv"), index=False)
print(f"\nROC-AUC of predicted log owners for identifying 10M+ games: {auc:.3f} (base rate {is_bb.mean():.4f})")
print(top_tbl.round(4).to_string(index=False))
bb = both[both["owners"] >= 1e7]
print(f"10M+ games: median predicted owners {bb['pred_owners'].median():,.0f}; median predicted percentile "
      f"{bb['pred_percentile'].median():.3f}; mean log residual {bb['resid_log'].mean():.2f}")
bb.sort_values("owners", ascending=False)[
    ["source", "name", "release_dt", "estimated_owners", "pred_owners", "pred_percentile", "resid_log", "price",
     "genre_Free_To_Play", "play_mode", "n_languages", "publisher", "self_published"]] \
    .to_csv(os.path.join(TAB, "blockbusters_all_86.csv"), index=False)

# =============================================================== label consistency (diagnostic only)
# A Steam user review requires owning the game, so a review count above the upper bound of the owner
# range is logically impossible and indicates an erroneous owner label.
for df in (te, oof):
    df["owners_upper"] = df["estimated_owners"].str.split(" - ").str[1].astype(float)
    df["label_inconsistent"] = df["reviews"] > df["owners_upper"]
n_bad_te, n_bad_oof = int(te["label_inconsistent"].sum()), int(oof["label_inconsistent"].sum())
print(f"\nGames whose review count exceeds the upper bound of their owner range: test {n_bad_te} "
      f"({n_bad_te / len(te):.2%}), out-of-fold {n_bad_oof} ({n_bad_oof / len(oof):.2%})")
bad = te[te["label_inconsistent"]]
print(f"  test: mean log residual {bad['resid_log'].mean():.2f} (all test games {te['resid_log'].mean():.2f}); "
      f"{bad['name'].isin(lists['largest_overpredictions']['name']).sum()} of the 20 largest over-predictions")
consistency = pd.DataFrame([
    {"subset": "all test games", "games": len(te), "R2_log": r2_score(te["true_log"], te["pred_log"]),
     "MAE_log": te["abs_log_err"].mean()},
    {"subset": "test games with consistent labels", "games": int((~te["label_inconsistent"]).sum()),
     "R2_log": r2_score(te.loc[~te["label_inconsistent"], "true_log"], te.loc[~te["label_inconsistent"], "pred_log"]),
     "MAE_log": te.loc[~te["label_inconsistent"], "abs_log_err"].mean()},
])
consistency.to_csv(os.path.join(TAB, "label_consistency_sensitivity.csv"), index=False)
print(consistency.round(4).to_string(index=False))
bad.sort_values("resid_log", ascending=False)[pred_cols + ["reviews", "owners_upper"]] \
    .to_csv(os.path.join(TAB, "test_inconsistent_labels.csv"), index=False)
print(bad.sort_values("resid_log", ascending=False)[["name", "estimated_owners", "reviews", "pred_owners", "resid_log"]]
      .head(10).round(2).to_string(index=False))

# =============================================================== Part 4: diagnostic plots
order = te.drop_duplicates("estimated_owners").sort_values("owners")["estimated_owners"].tolist()


def short(o):
    def f(n):
        n = int(n)
        return f"{n // 1_000_000}M" if n >= 1_000_000 else (f"{n // 1000}k" if n >= 1000 else str(n))
    lo, hi = o.split(" - ")
    return f"{f(lo)}–{f(hi)}"


# (b) predicted vs actual
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
hb = axes[0].hexbin(te["true_log"], te["pred_log"], gridsize=40, bins="log", cmap="viridis", mincnt=1)
lims = [te["true_log"].min() - 0.3, te["true_log"].max() + 0.3]
axes[0].plot(lims, lims, color=RED, ls="--", lw=1, label="perfect prediction")
axes[0].set_xlabel("actual log1p(owners)")
axes[0].set_ylabel("predicted log1p(owners)")
axes[0].set_title("(a) Each test game")
axes[0].legend(loc="upper left")
plt.colorbar(hb, ax=axes[0], label="games (log scale)")
box = [te.loc[te["estimated_owners"] == o, "pred_log"].values for o in order]
axes[1].boxplot(box, showfliers=False, widths=0.6)
axes[1].plot(range(1, len(order) + 1), [np.log1p(te.loc[te["estimated_owners"] == o, "owners"].iloc[0]) for o in order],
             color=RED, marker="o", label="actual value")
axes[1].set_xticks(range(1, len(order) + 1))
axes[1].set_xticklabels([f"{short(o)}\n(n={int((te['estimated_owners'] == o).sum())})" for o in order], rotation=60, fontsize=8)
axes[1].set_ylabel("log1p(owners)")
axes[1].set_title("(b) Distribution of predictions within each actual owner range")
axes[1].legend()
fig.suptitle("Predicted vs actual log owners, test set", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_b_predicted_vs_actual.png"), dpi=150, bbox_inches="tight")
plt.close()

# (c) residual vs predicted
fig, ax = plt.subplots(figsize=(9, 5))
hb = ax.hexbin(te["pred_log"], te["resid_log"], gridsize=40, bins="log", cmap="viridis", mincnt=1)
bins = pd.qcut(te["pred_log"], 20, duplicates="drop")
mean_line = te.groupby(bins, observed=True).agg(x=("pred_log", "mean"), y=("resid_log", "mean"))
ax.plot(mean_line["x"], mean_line["y"], color=RED, marker="o", label="mean residual (20 prediction bins)")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("predicted log1p(owners)")
ax.set_ylabel("residual (predicted − actual, log scale)")
ax.set_title("Residual vs predicted value, test set (positive = over-prediction)")
ax.legend()
plt.colorbar(hb, ax=ax, label="games (log scale)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_c_residual_vs_predicted.png"), dpi=150)
plt.close()

# (d) absolute error vs actual owners, and (e) bias/MAE by owner range
rng_stats = group_stats(te, "estimated_owners").reindex(order)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
x = np.arange(len(order))
axes[0].bar(x - 0.2, rng_stats["MAE_log"], 0.4, color=BLUE, label="MAE (log scale)")
axes[0].bar(x + 0.2, rng_stats["mean_bias_log"], 0.4, color=ORANGE, label="mean bias (log scale)")
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels([short(o) for o in order], rotation=60)
axes[0].set_ylabel("natural-log units (1.0 ≈ factor 2.7)")
axes[0].set_title("(a) Error and bias by actual owner range")
axes[0].legend()
axes[1].bar(x, rng_stats["MAE_orig_M"], color=GREY)
axes[1].set_yscale("log")
axes[1].set_xticks(x)
axes[1].set_xticklabels([short(o) for o in order], rotation=60)
axes[1].set_ylabel("mean absolute error (millions of owners, log scale)")
axes[1].set_title("(b) Original-scale absolute error by actual owner range")
for i, sh in enumerate(rng_stats["share_of_sq_error_orig"]):
    if sh >= 0.02:
        axes[1].text(i, rng_stats["MAE_orig_M"].iloc[i], f"{sh:.0%} of\nsq. error", ha="center", va="bottom", fontsize=7)
fig.suptitle("Error by actual owner range, test set", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_d_error_by_owner_range.png"), dpi=150, bbox_inches="tight")
plt.close()

# (f) error by release year and game type
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for ax, key, title in [(axes[0], "year_band", "Release year"), (axes[1], "business_model", "Business model"),
                       (axes[2], "play_mode", "Play mode")]:
    s = group_stats(te, key)
    xx = np.arange(len(s))
    ax.bar(xx - 0.2, s["MAE_log"], 0.4, color=BLUE, label="MAE (log)")
    ax.bar(xx + 0.2, s["mean_bias_log"], 0.4, color=ORANGE, label="mean bias (log)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels([f"{l}\n(n={n:,})" for l, n in zip(s.index.astype(str), s["games"])], rotation=20, fontsize=8)
    ax.set_title(title)
axes[0].legend()
fig.suptitle("Error and bias by release year and game type, test set (log scale)", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_e_error_by_year_and_type.png"), dpi=150, bbox_inches="tight")
plt.close()

# (g) blockbusters: ranking and compression
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].hist(both.loc[is_bb == 0, "pred_percentile"], bins=50, color=GREY, alpha=0.6, density=True, label="games below 10M")
axes[0].hist(both.loc[is_bb == 1, "pred_percentile"], bins=50, color=RED, alpha=0.8, density=True, label="10M+ games (n=86)")
axes[0].set_xlabel("percentile of the model's prediction among all games")
axes[0].set_ylabel("density")
axes[0].set_title(f"(a) Where blockbusters rank (ROC-AUC {auc:.3f})")
axes[0].legend()
axes[1].scatter(bb["owners"] / 1e6, bb["pred_owners"] / 1e6, color=RED, s=18)
axes[1].plot([1e-2, 200], [1e-2, 200], color="black", ls="--", lw=1)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlim(5, 200)
axes[1].set_xlabel("actual owners (millions)")
axes[1].set_ylabel("predicted owners (millions)")
axes[1].set_title("(b) Predicted vs actual owners, 10M+ games")
fig.suptitle("Blockbuster games: ranked high, size under-predicted (out-of-fold + test predictions)", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_f_blockbusters.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nOutputs written to", OUT)

# =============================================================== within-year accuracy (release-year confound)
wy = []
for year, g in te.groupby("release_year"):
    if len(g) >= 300:
        wy.append({"release_year": year, "games": len(g), "R2_log_within_year": r2_score(g["true_log"], g["pred_log"]),
                   "spearman_within_year": g["true_log"].corr(g["pred_log"], method="spearman")})
wy = pd.DataFrame(wy)
wy.to_csv(os.path.join(TAB, "within_year_r2_test.csv"), index=False)
print("\nWithin-year R2_log (test)\n", wy.round(3).to_string(index=False),
      f"\nGames-weighted mean: {np.average(wy['R2_log_within_year'], weights=wy['games']):.3f}")
