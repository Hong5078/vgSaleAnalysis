# Week 2 — Candidate Research Directions Checklist

Context: the parent paper's Linear Regression result (R^2 = 1.000) is a data
leakage artifact — `Global_Sales = NA_Sales + EU_Sales + JP_Sales + Other_Sales`,
so feeding those columns in as predictors just re-learns addition (see
`paper_replication/README.md`). All three directions below are ways to ask a
non-trivial question with the same/related datasets.

## Direction 1 — Regional sales trend prediction from Genre + Platform
Predict NA/EU/JP sales using only Genre, Platform (and Year as the trend axis)
— no other regional column as a predictor, so no leakage.

- [x] Confirm dataset has no leakage risk for this framing (`vgsales.csv`: Genre, Platform, Year are independent of the sales columns)
- [x] Build data pipeline: clean, one-hot encode Genre/Platform, keep Year
- [x] Exploratory trend plots: sales-over-time by region, by genre, by platform
- [x] Train per-region models (Linear Regression, Decision Tree, Random Forest) for NA_Sales, EU_Sales, JP_Sales
- [x] Evaluate with MSE/RMSE/MAE/R^2 per region, compare models
- [x] Identify which genres/platforms drive sales differently by region (feature importance / coefficients)
- [ ] Write up findings: does genre+platform meaningfully predict regional sales, and do regional preferences diverge (e.g. JP vs NA taste)?
- [ ] Decide if this becomes the new project direction to present

## Direction 2 — Critic/User score as prediction target
Reframe the problem from "predict sales" to "predict reception": use
Genre/Platform/Publisher/Year (+ optionally sales as a feature, since now sales
isn't the leaky target) to predict Critic_Score and/or User_Score from
`paper_replication/video_game_sales_and_ratings.csv`.

- [x] Load and clean `video_game_sales_and_ratings.csv`, check missingness of Critic_Score/User_Score (many rows dropped: 7,983 rows with Critic_Score, 7,463 with User_Score)
- [x] Decide target(s): Critic_Score and User_Score, modeled separately
- [x] Decide feature set (Genre, Platform, Rating, Year_of_Release — Publisher/Developer excluded for high cardinality/missingness)
- [x] Baseline models: Linear Regression, Decision Tree, Random Forest
- [x] Evaluate: MSE/RMSE/MAE/R^2 — Critic R^2 = 0.103, User R^2 = 0.137 (Random Forest, best model for both) — low but leakage-free, as expected
- [x] Compare Critic_Score predictability vs User_Score predictability — User_Score slightly more predictable
- [x] Extended Part B: mean score by genre/platform, critic-vs-user reception gap, Pearson correlation (r=0.58), one-way ANOVA (Genre and Platform both significant, p<<0.05 for both scores)
- [x] Write up findings (`Week 2/Direction2_Findings.md` and `Week 2/Direction2_Findings.ipynb`)

## Direction 3 — POAFD-trend analysis replication/extension
"Predicting game ownership dynamics: a novel POAFD-trend analysis approach"
(in `Five research paper/`) — the PDF is password-protected so its method is
not yet confirmed; first step is understanding what POAFD actually is before
committing to this direction.

- [ ] Get a readable copy of the paper (ask instructor/library for unprotected PDF, or re-download from the source) and read Section II/III for the POAFD method definition
- [ ] Identify what data the paper uses (ownership/playtime data vs. sales data — may not be the same as our Kaggle dataset)
- [ ] Determine if our current dataset can approximate "ownership dynamics" or if a new dataset is needed (e.g. Steam ownership/playtime data)
- [ ] Sketch what "trend analysis" means in their method (time-series decomposition? decay curves? clustering trajectories?)
- [ ] Scope a feasible mini-replication once the method is understood
- [ ] Write up findings / feasibility assessment

## Cross-cutting
- [ ] Meet as a team to pick one direction (or a combination) to carry forward as the new project focus
- [ ] Update `NOTES.md` / activity log with the decision and rationale
