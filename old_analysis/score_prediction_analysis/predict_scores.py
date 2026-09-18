"""
Direction 2, Part A - predicting Critic_Score and User_Score from Genre,
Platform, Year_of_Release, and Rating.

Unlike the parent paper's Global_Sales-from-regional-sales setup, none of
these predictors are algebraically related to either target, so there is no
leakage risk here.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "paper_replication", "video_game_sales_and_ratings.csv"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
FEATURE_COLS = ["Genre", "Platform", "Rating"]
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce") * 10  # scale 0-10 -> 0-100
    df = df.dropna(subset=["Year_of_Release", "Genre", "Platform"]).copy()
    df["Year_of_Release"] = df["Year_of_Release"].astype(int)
    print(f"Loaded {len(df)} rows after dropping missing Year/Genre/Platform")
    print(f"  Critic_Score available: {df['Critic_Score'].notna().sum()} rows")
    print(f"  User_Score available:   {df['User_Score'].notna().sum()} rows")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    cat = df[FEATURE_COLS].fillna("Unknown")
    features = pd.get_dummies(cat, drop_first=True)
    features["Year_of_Release"] = df["Year_of_Release"]
    return features


def evaluate_model(name, model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return {
        "model": name,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }


def train_and_evaluate_target(target: str, df: pd.DataFrame, out_dir: str):
    subset = df.dropna(subset=[target]).copy()
    X = build_features(subset)
    y = subset[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        metrics["target"] = target
        metrics["n_rows"] = len(subset)
        results.append(metrics)
        fitted[name] = model

    rf = fitted["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(15)
    plt.figure(figsize=(8, 6))
    top_features[::-1].plot(kind="barh")
    plt.title(f"Top feature importances - Random Forest - {target}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"feature_importance_{target}.png"), dpi=150)
    plt.close()

    return results, importances


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_clean(DATA_PATH)

    targets = ["Critic_Score", "User_Score"]
    all_results = []
    importances_by_target = {}
    for target in targets:
        print(f"\n=== {target} ===")
        results, importances = train_and_evaluate_target(target, df, OUT_DIR)
        importances_by_target[target] = importances
        for r in results:
            print(
                f"{r['model']:<20} n={r['n_rows']:<6} MSE: {r['MSE']:.2f}  RMSE: {r['RMSE']:.2f}  "
                f"MAE: {r['MAE']:.2f}  R2: {r['R2']:.4f}"
            )
        all_results.extend(results)

    summary = pd.DataFrame(all_results)[["target", "model", "n_rows", "MSE", "RMSE", "MAE", "R2"]]
    summary.to_csv(os.path.join(OUT_DIR, "score_metrics_summary.csv"), index=False)
    print("\n=== Summary (also saved to outputs/score_metrics_summary.csv) ===")
    print(summary.to_string(index=False))

    plt.figure(figsize=(8, 5))
    pivot = summary.pivot(index="target", columns="model", values="R2")
    pivot.plot(kind="bar")
    plt.ylabel("R^2 (test set)")
    plt.title("Model comparison: R^2 by target (Critic vs User Score)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "score_model_comparison_r2.png"), dpi=150)
    plt.close()

    print("\n=== Top 5 features by importance, per target (Random Forest) ===")
    for target, importances in importances_by_target.items():
        print(f"\n{target}:")
        print(importances.head(5).to_string())

    print(f"\nAll figures and metrics written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
