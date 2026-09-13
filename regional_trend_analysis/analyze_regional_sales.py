"""
Direction 1 - Regional sales trend prediction from Genre + Platform.

Predicts NA_Sales, EU_Sales, and JP_Sales independently from Genre, Platform,
and Year only. No regional sales column is used as a predictor for another
region's target, so there is no arithmetic-identity leakage (unlike the
parent paper's Global_Sales-from-regional-sales setup).
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

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "vgsales.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
REGIONS = ["NA_Sales", "EU_Sales", "JP_Sales"]
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=["Year", "Genre", "Platform"]).copy()
    df["Year"] = df["Year"].astype(int)
    print(f"Loaded {before} rows, {len(df)} after dropping missing Year/Genre/Platform")
    return df


def plot_trends(df: pd.DataFrame, out_dir: str) -> None:
    yearly = df.groupby("Year")[REGIONS].sum().reset_index()
    yearly = yearly[(yearly["Year"] >= 1990) & (yearly["Year"] <= 2016)]

    plt.figure(figsize=(9, 5))
    for region in REGIONS:
        plt.plot(yearly["Year"], yearly[region], marker="o", label=region.replace("_Sales", ""))
    plt.xlabel("Year")
    plt.ylabel("Total sales (millions of units)")
    plt.title("Regional sales trend over time (1990-2016)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trend_sales_over_time_by_region.png"), dpi=150)
    plt.close()

    top_genres = df.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False).head(6).index
    genre_yearly = (
        df[df["Genre"].isin(top_genres) & df["Year"].between(1990, 2016)]
        .groupby(["Year", "Genre"])["Global_Sales"]
        .sum()
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=genre_yearly, x="Year", y="Global_Sales", hue="Genre")
    plt.ylabel("Global sales (millions of units)")
    plt.title("Global sales trend by genre (top 6 genres, 1990-2016)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trend_sales_by_genre.png"), dpi=150)
    plt.close()

    region_genre = df.groupby("Genre")[REGIONS].sum()
    region_genre_share = region_genre.div(region_genre.sum(axis=0), axis=1)
    plt.figure(figsize=(10, 7))
    sns.heatmap(region_genre_share, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Share of each region's total sales by genre")
    plt.xlabel("Region")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "genre_share_by_region_heatmap.png"), dpi=150)
    plt.close()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.get_dummies(df[["Genre", "Platform"]], drop_first=True)
    features["Year"] = df["Year"]
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


def train_and_evaluate_region(region: str, X: pd.DataFrame, df: pd.DataFrame, out_dir: str):
    y = df[region]
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
        metrics["region"] = region.replace("_Sales", "")
        results.append(metrics)
        fitted[name] = model

    rf = fitted["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(15)
    plt.figure(figsize=(8, 6))
    top_features[::-1].plot(kind="barh")
    plt.title(f"Top feature importances - Random Forest - {region.replace('_Sales', '')}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"feature_importance_{region}.png"), dpi=150)
    plt.close()

    return results, importances


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_clean(DATA_PATH)

    plot_trends(df, OUT_DIR)

    X = build_features(df)

    all_results = []
    importances_by_region = {}
    for region in REGIONS:
        print(f"\n=== {region} ===")
        results, importances = train_and_evaluate_region(region, X, df, OUT_DIR)
        importances_by_region[region] = importances
        for r in results:
            print(
                f"{r['model']:<20} MSE: {r['MSE']:.4f}  RMSE: {r['RMSE']:.4f}  "
                f"MAE: {r['MAE']:.4f}  R2: {r['R2']:.4f}"
            )
        all_results.extend(results)

    summary = pd.DataFrame(all_results)[["region", "model", "MSE", "RMSE", "MAE", "R2"]]
    summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    print("\n=== Summary (also saved to outputs/metrics_summary.csv) ===")
    print(summary.to_string(index=False))

    plt.figure(figsize=(9, 5))
    pivot = summary.pivot(index="region", columns="model", values="R2")
    pivot.plot(kind="bar")
    plt.ylabel("R^2 (test set)")
    plt.title("Model comparison: R^2 by region")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison_r2.png"), dpi=150)
    plt.close()

    print("\n=== Top 5 genre/platform features by importance, per region (Random Forest) ===")
    for region, importances in importances_by_region.items():
        print(f"\n{region.replace('_Sales', '')}:")
        print(importances.head(5).to_string())

    print(f"\nAll figures and metrics written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
