# Proposal: Switching to a New Dataset for Video Game Sales Prediction

**Course:** DS 340W  **Date:** September 18, 2026
**Parent paper:** Gray, Shalan & Kadlec, *Sales Predictions for Video Games Using Predictive
Analytics of Market Data*, IEEE CICN 2024 (DOI 10.1109/CICN63059.2024.10847429)
**Current dataset:** Kaggle "Video Game Sales and Ratings" (VGChartz + Metacritic, 16,719 rows, 1980–2016)

---

## 1. Summary

The parent paper was replicated and then corrected in three stages. Each stage removed a
source of information leakage, and each time the honest result showed the same limit: **this
dataset contains almost no information that exists before a game is released.** The only
strong pre-release signal left is the past sales of the same franchise and publisher. To
build a model that can actually forecast sales for a new game, the project needs a dataset
with pre-release features. I propose moving to Steam store data.

## 2. What the old analysis found

| Stage | Predictors | Result | Problem found |
|---|---|---|---|
| Parent paper replication | Includes NA/EU/JP/Other sales | Linear Regression R² = 1.000 | `Global_Sales` = NA + EU + JP + Other. The model learns addition, not sales drivers (**arithmetic leakage**) |
| Phase 1: corrected replication | Genre, Platform, Rating, Year, Publisher, critic/user scores and counts | Random Forest R² = 0.44 (5-fold CV), 0.68 (validation) | Most important feature is `User_Count`, which grows *after* release roughly in line with copies sold (**post-release leakage**). The model cannot forecast an unreleased game |
| Phase 2: pre-release hit classifier | Only information known before launch; title-grouped CV to stop multi-platform duplicates leaking across folds (**duplicate-title leakage**) | Random Forest ROC-AUC 0.86, avg. precision 0.53 (base rate 0.125); 0.88 AUC on a train ≤2013 / test 2014–16 split | Honest, but the top four features are all prior franchise/publisher **sales**. Adding launch-day `Critic_Score` raises AUC only from 0.83 to 0.88 |

Code and outputs: `old_analysis/paper_replication/` and `old_analysis/week3/`.

## 3. Why the current dataset limits the project

1. **No pre-release demand signals.** There is nothing on price, marketing, trailers,
   pre-orders, wishlists, or hype. The only pre-release inputs are genre, platform, publisher,
   ESRB rating and year, which are weak on their own.
2. **Only lifetime totals.** Sales are one number per game with no weekly or monthly history,
   so early-sales forecasting ("predict lifetime sales from the first month") is impossible.
3. **Outdated.** The data stops in 2016 (VGChartz stopped publishing reliable estimates), so
   it misses digital distribution, which now makes up most of the market.
4. **Franchise information must be guessed.** Franchises were recovered with a title-text
   heuristic, which produces errors (e.g. "Yu-Gi-Oh" truncated to "yu").
5. **Heavy skew and missing data.** Critic scores are missing for about half the rows, and a
   single title (Wii Sports, 82.5M) moved the test R² by 0.14.

Further tuning cannot fix these problems; they come from what the dataset records.

## 4. Proposed new dataset

**Primary: Steam Games Dataset (Kaggle, fronkongames)** — a scrape of the Steam store and
SteamSpy covering tens of thousands of PC games through 2024–25.

- **Target:** estimated owners range (SteamSpy). It is an ordinal range rather than an exact
  number, so the task becomes ordinal classification of commercial success, which is still
  sales prediction.
- **Pre-release features:**
  - price
  - release date
  - developer and publisher (plus their track record, using the same expanding-window method from Phase 2)
  - genres and user tags
  - supported languages and platforms
  - number of screenshots and trailers
  - required age
  - the store description text
- **Excluded as post-release:** review counts and scores, playtime, peak concurrent users,
  recommendations.

**Optional additions:**
- IGDB API "hypes" (follows before release), a direct measure of anticipation.
- SteamCharts monthly player counts, for a second-stage "forecast from the first month" model.

## 5. What stays the same, and what is new

**Kept from the old analysis:**
- the research question (predicting video game sales)
- the leakage checks (arithmetic, post-release, duplicate titles)
- title-grouped cross-validation and a chronological test split
- the parent paper's models (Linear/Logistic Regression, Decision Tree) as baselines

**New contributions:**
1. Pre-release features, including text features from store descriptions (e.g. TF-IDF or
   sentence embeddings).
2. A two-stage design: a *before-launch* model and an *after the first month* model. This
   measures how much early post-release data adds, a question the parent paper could not ask.
3. Recent data (up to 2025) instead of data that ends in 2016.

## 6. Risks

- Steam covers PC only. Console sales are out of scope.
- Owner counts are SteamSpy **estimates** reported as ranges, not audited sales.
- The dataset's exact columns and coverage will be confirmed in the first notebook before
  committing to it.

## 7. Request

Approval to move the next phase of the project to Steam data, with the VGChartz analysis kept
in `old_analysis/` as the documented replication and the reason for the switch.
