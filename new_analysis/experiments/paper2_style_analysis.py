"""Second-parent-paper-style analysis (Bondarenko, Cabrera & Fitzek, IEEE SmartNets 2026).

Borrowed from that paper (its variables are not available here, so only its approach is used):
  1. separate models for paid and free-to-play games;
  2. price bands instead of a single continuous price;
  3. a log-linear outcome, so every coefficient is reported as a percentage change;
  4. descriptive tables reporting n, mean and median per band, and coefficient tables with
     p-values and 95% confidence intervals.

Differences from the paper, stated in the write-up:
  - Outcome: lifetime estimated owners (SteamSpy range midpoints), not 30-day wishlist conversion;
    Steam wishlists are not in this dataset.
  - Rating: the paper's main predictor is the 30-day user rating, which only exists after release.
    It is therefore excluded from every model here and reported only as a labelled post-release
    diagnostic.
  - Controls: release-year dummies are added because this dataset spans 1997-2024, while the paper
    covers a five-month window.
  - Sample: all labeled games, not the top-N by review count, so the sample is not selected on the
    outcome.

Protocol for the predictive comparison is the harness protocol (same split and folds as every
earlier experiment). Outputs: paper2_style/{tables,figures}; a new run folder is created if one exists.
"""
import json
import os

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

import harness as H

RS = H.RANDOM_STATE
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

OUT = os.path.join(H.HERE, "paper2_style")
k = 2
while os.path.exists(OUT) and os.listdir(OUT):
    OUT = os.path.join(H.HERE, f"paper2_style_run{k}")
    k += 1
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
BLUE, GREEN, ORANGE, RED, GREY = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8C8C8C"

data, BASE_FEATURES = H.load_data()
data["log_owners"] = np.log(data["owners"])          # owners >= 10,000, so no epsilon is needed
data["segment"] = np.where(data["price"] > 0, "paid", "free-to-play")

# Price bands follow the paper (Section III-E), with one extra band for prices above $79.99.
BAND_EDGES = [0.0, 10, 20, 30, 40, 60, 80, np.inf]
BAND_LABELS = ["$0.99–9.99", "$10–19.99", "$20–29.99", "$30–39.99", "$40–59.99", "$60–79.99", "$80+"]
data["price_band"] = pd.cut(data["price"], BAND_EDGES, labels=BAND_LABELS, right=False)
data.loc[data["segment"] == "free-to-play", "price_band"] = np.nan

GENRES = ["genre_Action", "genre_Adventure", "genre_RPG", "genre_Simulation", "genre_Strategy",
          "genre_Sports", "genre_Casual", "genre_Indie", "genre_Massively_Multiplayer", "genre_Racing"]
PRESENTATION = ["n_languages", "n_audio_languages", "n_screenshots", "description_length"]
PLATFORM_PLAY = ["mac", "linux", "required_age", "cat_Single_player", "cat_Multi_player",
                 "cat_Online_PvP", "cat_Online_Co_op", "cat_Full_controller_support", "cat_VR_Only"]

print(f"{len(data):,} games | paid {int((data['segment'] == 'paid').sum()):,} | "
      f"free-to-play {int((data['segment'] == 'free-to-play').sum()):,}")

# ---------------------------------------------------------------- post-release diagnostic variables
path = kagglehub.dataset_download("fronkongames/steam-games-dataset/versions/49")
with open(os.path.join(path, "games.json"), encoding="utf-8") as f:
    raw = json.load(f)
rev = pd.DataFrame([{"appid": int(a), "positive": v.get("positive") or 0, "negative": v.get("negative") or 0}
                    for a, v in raw.items()])
del raw
rev["reviews"] = rev["positive"] + rev["negative"]
rev["rating"] = np.where(rev["reviews"] >= 10, 100 * rev["positive"] / rev["reviews"].replace(0, np.nan), np.nan)
data = data.merge(rev[["appid", "reviews", "rating"]], on="appid", how="left", validate="one_to_one")
RATING_EDGES = [0, 40, 70, 80, 95, 100.001]
RATING_LABELS = ["Negative (0–39)", "Mixed (40–69)", "Mostly Positive (70–79)", "Very Positive (80–94)",
                 "Overwhelmingly Positive (95–100)"]
data["rating_category"] = pd.cut(data["rating"], RATING_EDGES, labels=RATING_LABELS, right=False)


# ================================================================ 1. descriptive tables (paper style)
def describe(frame, key, label):
    g = frame.groupby(key, observed=True)["owners"]
    out = pd.DataFrame({"n": g.size(), "mean owners": g.mean(), "median owners": g.median(),
                        "share ≥100k owners": frame.groupby(key, observed=True)["owners"].apply(lambda s: (s >= 1e5).mean())})
    out.insert(0, "grouping", label)
    return out


paid = data[data["segment"] == "paid"]
free = data[data["segment"] == "free-to-play"]

t_segment = describe(data, "segment", "Business model (all games)")
t_band = describe(paid, "price_band", "Paid games: price band")
t_rating_paid = describe(paid.dropna(subset=["rating_category"]), "rating_category", "Paid games: Steam rating category (POST-RELEASE diagnostic)")
t_rating_free = describe(free.dropna(subset=["rating_category"]), "rating_category", "Free-to-play: Steam rating category (POST-RELEASE diagnostic)")

genre_rows = []
for seg, frame in [("paid", paid), ("free-to-play", free)]:
    for g in GENRES:
        sub = frame[frame[g] == 1]
        if len(sub) >= 30:
            genre_rows.append({"grouping": f"{seg}: genre (inclusion-based)", "level": g.replace("genre_", ""),
                               "n": len(sub), "mean owners": sub["owners"].mean(), "median owners": sub["owners"].median(),
                               "share ≥100k owners": (sub["owners"] >= 1e5).mean()})
t_genre = pd.DataFrame(genre_rows)

for name, t in [("t1_business_model", t_segment), ("t2_price_band_paid", t_band),
                ("t3_rating_category_paid", t_rating_paid), ("t4_rating_category_free", t_rating_free)]:
    t.to_csv(os.path.join(TAB, name + ".csv"))
    print(f"\n{t['grouping'].iloc[0]}\n", t.drop(columns=["grouping"]).round(3).to_string())
t_genre.to_csv(os.path.join(TAB, "t5_genre.csv"), index=False)
print("\nGenres\n", t_genre.round(3).to_string(index=False))


# ================================================================ 2. OLS with percentage effects
def ols_table(frame, columns, label):
    """OLS of log(owners) on `columns`; returns coefficients as percentage effects with p-values and 95% CIs.
    Classical and heteroskedasticity-robust (HC1) standard errors are both reported."""
    X = np.column_stack([np.ones(len(frame))] + [frame[c].astype(float).values for c in columns])
    y = frame["log_owners"].values
    names = ["(intercept)"] + list(columns)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    n, p = X.shape
    dof = n - p
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    meat = (X * (resid ** 2)[:, None]).T @ X * n / dof
    se_hc1 = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    tcrit = stats.t.ppf(0.975, dof)
    out = pd.DataFrame({
        "model": label, "variable": names, "coef": beta, "std_error": se,
        "p_value": 2 * stats.t.sf(np.abs(beta / se), dof),
        "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se,
        "p_value_robust_HC1": 2 * stats.t.sf(np.abs(beta / se_hc1), dof),
    })
    out["pct_effect"] = 100 * (np.exp(out["coef"]) - 1)
    out["pct_ci_low"] = 100 * (np.exp(out["ci_low"]) - 1)
    out["pct_ci_high"] = 100 * (np.exp(out["ci_high"]) - 1)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    out.attrs["r2"], out.attrs["n"] = r2, n
    return out


def year_dummies(frame, ref_year=2020):
    years = sorted(frame["release_year"].unique())
    cols = {}
    for y in years:
        if y != ref_year and (frame["release_year"] == y).sum() >= 30:
            cols[f"year_{y}"] = (frame["release_year"] == y).astype(int)
    return pd.DataFrame(cols, index=frame.index)


def band_dummies(frame, ref="$0.99–9.99"):
    return pd.DataFrame({f"price_{b}": (frame["price_band"] == b).astype(int)
                         for b in BAND_LABELS if b != ref}, index=frame.index)


results, model_specs = [], {}
for seg, frame in [("paid", paid), ("free-to-play", free)]:
    f = frame.copy()
    yd = year_dummies(f)
    f = pd.concat([f, yd], axis=1)
    genre_cols = [g for g in GENRES if f[g].sum() >= 30]
    if seg == "paid":
        f = pd.concat([f, band_dummies(f)], axis=1)
        price_cols = [c for c in f.columns if c.startswith("price_$")]
        spec_a = price_cols + genre_cols + list(yd.columns)
        spec_b = price_cols + genre_cols + PRESENTATION + PLATFORM_PLAY + list(yd.columns)
        # continuous-price variant, to report the paper's "per +$10" style effect
        f["price_per_10usd"] = f["price"] / 10
        spec_c = ["price_per_10usd"] + genre_cols + PRESENTATION + PLATFORM_PLAY + list(yd.columns)
        specs = {"A. paper-style (price bands + genres + year)": spec_a,
                 "B. full pre-release (adds store-page features)": spec_b,
                 "C. as B, continuous price per $10": spec_c}
    else:
        spec_a = genre_cols + list(yd.columns)
        spec_b = genre_cols + PRESENTATION + PLATFORM_PLAY + list(yd.columns)
        specs = {"A. paper-style (genres + year)": spec_a,
                 "B. full pre-release (adds store-page features)": spec_b}
    for spec_name, cols in specs.items():
        label = f"{seg} — {spec_name}"
        t = ols_table(f, cols, label)
        model_specs[label] = (f, cols)
        print(f"\n{label}: n = {t.attrs['n']:,}, in-sample R² = {t.attrs['r2']:.4f}")
        key = t[~t["variable"].str.startswith("year_")].copy()
        print(key[["variable", "coef", "pct_effect", "p_value", "pct_ci_low", "pct_ci_high"]].round(3).to_string(index=False))
        t["in_sample_r2"], t["n"] = t.attrs["r2"], t.attrs["n"]
        results.append(t)
ols_all = pd.concat(results, ignore_index=True)
ols_all.to_csv(os.path.join(TAB, "t6_ols_percentage_effects.csv"), index=False)


# ================================================================ 3. do separate models predict better?
def cv_compare():
    train, _ = H.get_split(data, "A_all_games")
    folds = list(KFold(n_splits=H.N_FOLDS, shuffle=True, random_state=RS).split(train))
    y = np.log1p(train["owners"].values)
    seg = train["segment"].values
    band_cols = pd.get_dummies(train["price_band"], prefix="band").astype(int)
    train_b = pd.concat([train.reset_index(drop=True), band_cols.reset_index(drop=True)], axis=1)
    feat_band = [c for c in BASE_FEATURES if c != "price"] + list(band_cols.columns)
    rows = []
    for model_name, model in [("Linear Regression", LinearRegression()),
                              ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=RS))]:
        for strategy in ["pooled", "separate paid / free-to-play"]:
            for feat_name, feats, frame in [("continuous price (base 42)", BASE_FEATURES, train),
                                            ("price bands", feat_band, train_b)]:
                oof = np.zeros(len(frame))
                for tr, va in folds:
                    if strategy == "pooled":
                        m = clone(model).fit(frame.loc[tr, feats], y[tr])
                        oof[va] = m.predict(frame.loc[va, feats])
                    else:
                        for s in ["paid", "free-to-play"]:
                            tr_s, va_s = tr[seg[tr] == s], va[seg[va] == s]
                            m = clone(model).fit(frame.loc[tr_s, feats], y[tr_s])
                            oof[va_s] = m.predict(frame.loc[va_s, feats])
                rows.append({"model": model_name, "strategy": strategy, "features": feat_name,
                             "R2_log_all": r2_score(y, oof), "MAE_log_all": mean_absolute_error(y, oof),
                             "R2_log_paid": r2_score(y[seg == "paid"], oof[seg == "paid"]),
                             "R2_log_free": r2_score(y[seg == "free-to-play"], oof[seg == "free-to-play"]),
                             "MAE_log_free": mean_absolute_error(y[seg == "free-to-play"], oof[seg == "free-to-play"])})
                print("   ", rows[-1]["model"], "|", strategy, "|", feat_name, "-> R2_all",
                      round(rows[-1]["R2_log_all"], 4))
    return pd.DataFrame(rows)


print("\nCross-validated comparison (harness protocol: training portion, 5 folds, log1p owners)")
cv = cv_compare()
cv.to_csv(os.path.join(TAB, "t7_pooled_vs_separate_cv.csv"), index=False)
print(cv.round(4).to_string(index=False))


# ================================================================ 4. figures
def effect_plot(label, ax, drop_prefix=("year_", "(intercept)")):
    t = ols_all[ols_all["model"] == label]
    t = t[~t["variable"].str.startswith(drop_prefix)].copy()
    t["ratio"] = np.exp(t["coef"])
    t = t.sort_values("ratio")
    ypos = np.arange(len(t))
    ax.errorbar(t["ratio"], ypos, xerr=[t["ratio"] - np.exp(t["ci_low"]), np.exp(t["ci_high"]) - t["ratio"]],
                fmt="o", color=BLUE, ecolor=GREY, capsize=3, markersize=5)
    ax.axvline(1, color=RED, lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([v.replace("genre_", "").replace("cat_", "").replace("price_", "").replace("_", " ") for v in t["variable"]],
                       fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("effect on expected owners (ratio; 1.0 = no effect)")
    ax.set_title(label, fontsize=10)


fig, axes = plt.subplots(1, 2, figsize=(15, 8))
effect_plot("paid — B. full pre-release (adds store-page features)", axes[0])
effect_plot("free-to-play — B. full pre-release (adds store-page features)", axes[1])
fig.suptitle("Multiplicative effects on expected owners, separate models (95% confidence intervals)", y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_effects_paid_vs_free.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
b = t_band.dropna()
axes[0].bar(b.index.astype(str), b["median owners"], color=BLUE)
axes[0].set_ylabel("median owners")
axes[0].set_title(f"(a) Paid games: median owners by price band")
axes[0].tick_params(axis="x", rotation=30)
for i, (m, n) in enumerate(zip(b["median owners"], b["n"])):
    axes[0].text(i, m, f"n={n:,}", ha="center", va="bottom", fontsize=7)
w = 0.38
xs = np.arange(len(RATING_LABELS))
for i, (lbl, t) in enumerate([("paid", t_rating_paid), ("free-to-play", t_rating_free)]):
    t = t.reindex(RATING_LABELS)
    axes[1].bar(xs + (i - 0.5) * w, t["median owners"], w, color=[BLUE, GREEN][i], label=lbl)
axes[1].set_yscale("log")
axes[1].set_xticks(xs)
axes[1].set_xticklabels([l.split(" (")[0] for l in RATING_LABELS], rotation=20, fontsize=8)
axes[1].set_ylabel("median owners (log scale)")
axes[1].set_title("(b) Median owners by Steam rating category\n(post-release diagnostic, not a model input)")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_price_bands_and_rating.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(9, 4.5))
piv = cv[cv["features"] == "continuous price (base 42)"].pivot(index="model", columns="strategy", values="R2_log_all")
piv.plot(kind="bar", ax=ax, color=[GREY, GREEN], rot=0)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
ax.set_ylabel("cross-validated R² (log1p owners)")
ax.set_title("Pooled model vs separate paid / free-to-play models")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig3_pooled_vs_separate.png"), dpi=150)
plt.close()
print("\nOutputs written to", OUT)
