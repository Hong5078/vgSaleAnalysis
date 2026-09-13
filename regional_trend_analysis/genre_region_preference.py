"""
Research question: did certain genres do disproportionately well in one
region compared to others (e.g. JP favoring RPGs more than its overall
market size would predict)?

Method: Location Quotient (LQ), a standard regional-economics tool for this
exact question.

    LQ(genre, region) = (genre's share of that region's total sales)
                         / (genre's share of global total sales)

LQ > 1  -> the genre is OVER-represented in that region relative to how it
           does everywhere else (a regional preference).
LQ < 1  -> the genre is UNDER-represented in that region.
LQ = 1  -> the genre sells in that region exactly proportionally to its
           global popularity (no regional preference either way).

This is a different question from Direction 1's predictive modeling (can we
predict sales from genre+platform?) - here we're directly asking "which
genres over/under-perform in which regions", so we measure it directly
instead of inferring it from a regression's feature importances.

We also run a chi-square test of independence (Genre x Region, weighted by
sales) to check whether the genre/region association is statistically
significant, and Cramer's V for the effect size.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "vgsales.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
REGIONS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
REGION_LABELS = {"NA_Sales": "NA", "EU_Sales": "EU", "JP_Sales": "JP", "Other_Sales": "Other"}

sns.set_theme(style="whitegrid")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Genre"]).copy()
    return df


def compute_location_quotients(df: pd.DataFrame) -> pd.DataFrame:
    genre_region = df.groupby("Genre")[REGIONS].sum()
    genre_region = genre_region.rename(columns=REGION_LABELS)

    region_totals = genre_region.sum(axis=0)  # total sales per region
    grand_total = region_totals.sum()

    genre_share_in_region = genre_region.div(region_totals, axis=1)  # column-normalized
    genre_share_global = genre_region.sum(axis=1) / grand_total  # each genre's overall share

    lq = genre_share_in_region.div(genre_share_global, axis=0)
    return lq.sort_index(), genre_region


def chi_square_test(genre_region: pd.DataFrame):
    table = genre_region[["NA", "EU", "JP"]].values
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.sum()
    k = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k))
    return chi2, p, dof, cramers_v


def plot_lq_heatmap(lq: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        lq[["NA", "EU", "JP", "Other"]],
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=1.0,
        vmin=0.4,
        vmax=1.8,
        cbar_kws={"label": "Location Quotient (1.0 = proportional to global popularity)"},
    )
    plt.title("Regional preference by genre (Location Quotient)")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "genre_region_location_quotient.png"), dpi=150)
    plt.close()


def plot_top_over_under(lq: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    for ax, region in zip(axes, ["NA", "EU", "JP"]):
        ordered = lq[region].sort_values(ascending=False)
        colors = ["#c0392b" if v > 1 else "#2980b9" for v in ordered]
        ax.barh(ordered.index[::-1], ordered.values[::-1], color=colors[::-1])
        ax.axvline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{region}: genre LQ (red = over-represented)")
        ax.set_xlabel("Location Quotient")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "genre_lq_by_region_bars.png"), dpi=150)
    plt.close()


def plot_divergence(lq: pd.DataFrame, out_dir: str):
    divergence = lq[["NA", "EU", "JP"]].std(axis=1).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    divergence.plot(kind="barh", color="#8e44ad")
    plt.gca().invert_yaxis()
    plt.xlabel("Std. dev. of Location Quotient across NA/EU/JP")
    plt.title("Which genres diverge most by region? (higher = more regionally polarizing)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "genre_regional_divergence.png"), dpi=150)
    plt.close()
    return divergence


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load(DATA_PATH)

    lq, genre_region = compute_location_quotients(df)
    lq.to_csv(os.path.join(OUT_DIR, "genre_region_location_quotient.csv"))

    print("=== Location Quotient table (1.0 = proportional, >1 = over-represented) ===")
    print(lq.round(2).to_string())

    chi2, p, dof, cramers_v = chi_square_test(genre_region)
    print("\n=== Chi-square test of independence: Genre x Region (NA/EU/JP), weighted by sales ===")
    print(f"chi2 = {chi2:.1f}, dof = {dof}, p-value = {p:.3e}")
    print(f"Cramer's V (effect size) = {cramers_v:.4f}")
    if p < 0.05:
        print("-> statistically significant association between genre and region "
              "(genre preference is NOT independent of region).")
    else:
        print("-> no statistically significant association detected.")

    plot_lq_heatmap(lq, OUT_DIR)
    plot_top_over_under(lq, OUT_DIR)
    divergence = plot_divergence(lq, OUT_DIR)

    print("\n=== Genres ranked by how much their regional preference diverges (NA/EU/JP) ===")
    print(divergence.round(3).to_string())

    print("\n=== Per-region top 3 over-represented and under-represented genres ===")
    for region in ["NA", "EU", "JP"]:
        ordered = lq[region].sort_values(ascending=False)
        print(f"\n{region}:")
        print(f"  Over-represented (LQ high): {list(ordered.head(3).index)}")
        print(f"  Under-represented (LQ low): {list(ordered.tail(3).index)}")

    with open(os.path.join(OUT_DIR, "genre_region_stats.txt"), "w") as f:
        f.write(f"chi2 = {chi2:.1f}, dof = {dof}, p-value = {p:.3e}\n")
        f.write(f"Cramer's V = {cramers_v:.4f}\n")

    print(f"\nAll figures and tables written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
