"""
Direction 2, Part B - which genres/platforms get rated higher by critics vs.
users, is the effect statistically significant, and where do critics and
users disagree most?

Mirrors Direction 1's Part B structure (Location Quotient + chi-square) but
adapted to a continuous-score setting: mean score by group + one-way ANOVA
(since scores are continuous, not sales totals treated as counts).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, pearsonr

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "paper_replication", "video_game_sales_and_ratings.csv"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

sns.set_theme(style="whitegrid")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce") * 10
    df = df.dropna(subset=["Genre"]).copy()
    return df


def anova_by_group(df: pd.DataFrame, score_col: str, group_col: str):
    groups = [g[score_col].dropna().values for _, g in df.groupby(group_col) if g[score_col].notna().sum() >= 5]
    groups = [g for g in groups if len(g) > 1]
    f_stat, p_val = f_oneway(*groups)
    return f_stat, p_val


def plot_mean_scores_by_genre(df: pd.DataFrame, out_dir: str):
    means = df.groupby("Genre")[["Critic_Score", "User_Score"]].mean().sort_values("Critic_Score", ascending=False)
    means.plot(kind="bar", figsize=(11, 6))
    plt.ylabel("Mean score (0-100 scale)")
    plt.title("Mean Critic Score vs. User Score by Genre")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_score_by_genre.png"), dpi=150)
    plt.close()
    return means


def plot_critic_user_gap(means: pd.DataFrame, out_dir: str):
    gap = (means["Critic_Score"] - means["User_Score"]).sort_values()
    plt.figure(figsize=(8, 6))
    colors = ["#2980b9" if v < 0 else "#c0392b" for v in gap]
    gap.plot(kind="barh", color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Critic Score minus User Score (positive = critics rate higher)")
    plt.title("Critic-vs-User reception gap by genre")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "critic_user_gap_by_genre.png"), dpi=150)
    plt.close()
    return gap


def plot_scatter_correlation(df: pd.DataFrame, out_dir: str):
    paired = df.dropna(subset=["Critic_Score", "User_Score"])
    r, p = pearsonr(paired["Critic_Score"], paired["User_Score"])
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=paired, x="Critic_Score", y="User_Score", alpha=0.3, s=15)
    sns.regplot(data=paired, x="Critic_Score", y="User_Score", scatter=False, color="red")
    plt.title(f"Critic Score vs. User Score (Pearson r = {r:.2f}, n = {len(paired)})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "critic_vs_user_scatter.png"), dpi=150)
    plt.close()
    return r, p, len(paired)


def plot_mean_scores_by_platform(df: pd.DataFrame, out_dir: str, min_n=30):
    counts = df["Platform"].value_counts()
    keep = counts[counts >= min_n].index
    means = (
        df[df["Platform"].isin(keep)]
        .groupby("Platform")[["Critic_Score", "User_Score"]]
        .mean()
        .sort_values("Critic_Score", ascending=False)
    )
    means.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Mean score (0-100 scale)")
    plt.title(f"Mean Critic Score vs. User Score by Platform (n >= {min_n})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_score_by_platform.png"), dpi=150)
    plt.close()
    return means


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load(DATA_PATH)

    means = plot_mean_scores_by_genre(df, OUT_DIR)
    print("=== Mean Critic Score vs User Score by Genre ===")
    print(means.round(1).to_string())

    gap = plot_critic_user_gap(means, OUT_DIR)
    print("\n=== Critic-minus-User gap by genre (positive = critics rate higher) ===")
    print(gap.round(1).to_string())

    r, p, n = plot_scatter_correlation(df, OUT_DIR)
    print(f"\n=== Critic vs User Score correlation ===\nPearson r = {r:.3f}, p = {p:.3e}, n = {n}")

    platform_means = plot_mean_scores_by_platform(df, OUT_DIR)
    print("\n=== Mean Critic Score vs User Score by Platform (n >= 30) ===")
    print(platform_means.round(1).to_string())

    print("\n=== One-way ANOVA: does Genre significantly affect each score? ===")
    for score_col in ["Critic_Score", "User_Score"]:
        f_stat, p_val = anova_by_group(df, score_col, "Genre")
        sig = "significant" if p_val < 0.05 else "not significant"
        print(f"{score_col}: F = {f_stat:.2f}, p = {p_val:.3e} -> {sig}")

    print("\n=== One-way ANOVA: does Platform significantly affect each score? ===")
    for score_col in ["Critic_Score", "User_Score"]:
        f_stat, p_val = anova_by_group(df, score_col, "Platform")
        sig = "significant" if p_val < 0.05 else "not significant"
        print(f"{score_col}: F = {f_stat:.2f}, p = {p_val:.3e} -> {sig}")

    with open(os.path.join(OUT_DIR, "score_genre_stats.txt"), "w") as f:
        f.write(f"Pearson r (critic vs user) = {r:.3f}, p = {p:.3e}, n = {n}\n")
        for group_col in ["Genre", "Platform"]:
            for score_col in ["Critic_Score", "User_Score"]:
                f_stat, p_val = anova_by_group(df, score_col, group_col)
                f.write(f"ANOVA {score_col} by {group_col}: F = {f_stat:.2f}, p = {p_val:.3e}\n")

    print(f"\nAll figures and stats written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
