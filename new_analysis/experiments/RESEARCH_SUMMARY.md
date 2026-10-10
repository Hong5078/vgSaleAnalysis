# Research Summary — Predicting Steam Game Owners from Release-Time Information

Full experiment records: [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md). Reproduce with
`python experiments.py E0X` and `python final_checks.py` (requires notebook 01 to have produced
`data/processed/`).

## 1. Result in brief

| | Baseline (notebook 03) | Best model |
|---|---|---|
| Model | Random Forest, scikit-learn defaults | HistGradientBoostingRegressor, scikit-learn defaults |
| Target | Owners in millions | log1p(owners); back-transformed with expm1 (× Duan smearing factor for squared-error metrics) |
| Population | All games (Track A) | All games (Track A) |
| **CV R², log scale** | −0.119 ± 0.040 | **0.437 ± 0.011** |
| CV R², original scale | 0.013 ± 0.044 | 0.058 ± 0.029 (0.088 ± 0.044 with smearing) |
| CV MAE, original scale | 0.155 M owners | **0.104 M** (0.115 M with smearing) |
| CV RMSE, original scale | 1.485 M owners | 1.457 M (**1.435 M** with smearing) |
| Test set (used once), R² log scale | −0.117 | **0.426** |

**The target of CV R² ≥ 0.30 is met on the log scale (0.437), not on the original owner scale
(0.088).** The two scales measure different things (Section 6), and the difference must be
stated whenever the result is reported.

## 2. Best model specification

- **Dataset / population:** Steam Games Dataset (Kaggle, fronkongames, v49); all 76,727 labeled games
  released up to 30 June 2024 after the cleaning in notebook 01 (missing owner labels, playtests,
  demos and software removed). No games were removed for being hard to predict.
- **Features (42, all known at release):** list price, release year, required age, supported
  languages, full-audio languages, Mac and Linux support, screenshots, description length, 15 genre
  flags and 18 feature-category flags. Post-release variables (reviews, playtime, player counts,
  user tags, DLC, current price) and labels added after release are excluded (notebook 01, Section 2.6).
- **Preprocessing:** none beyond notebook 01 (the tree-based model needs no scaling; no missing values).
- **Target:** log1p(estimated owners), where owners is the midpoint of the SteamSpy range.
- **Model:** `HistGradientBoostingRegressor(random_state=42)`, defaults: learning rate 0.1, up to 100
  iterations, 31 leaves per tree, minimum 20 games per leaf, early stopping on 10% of each training fold.
- **Back-transformation:** expm1(prediction) when absolute error matters (typical game);
  expm1(prediction) × smearing factor (≈ 1.9, the mean of exp(training residuals)) when squared
  error matters.
- **Validation:** same 80/20 split as notebooks 03–04 (seed 42); 5-fold KFold (seed 42) on the 80%;
  the 20% test set used once after all decisions.

## 3. Experiments

| ID | Single change | Result (Track A) | Outcome |
|---|---|---|---|
| E00 | Reproduction of notebooks 03–04 | Numbers identical (RF R²_orig 0.013; Track B 0.255) | Harness verified |
| E01 | Target → log1p(owners) | RF R²_log −0.119 → 0.414; MAE_orig −32% | **Success**: largest single gain |
| E02 | Random Forest → HistGradientBoosting | R²_log 0.414 → 0.437; train–CV gap 0.50 → 0.06 | **Success**: better and less overfit |
| E03 | Duan smearing on back-transformation | R²_orig 0.058 → 0.088; RMSE 1.457 → 1.435; MAE 0.104 → 0.115 | **Trade-off**, as predicted |

Three meaningful experiments were run. Stopping condition 1 was first met at E01; E02 and E03 were
the two follow-ups the stopping rule allows. No hyperparameter search was performed.

**Validity checks.**
- **Grouped folds by title:** R²_log 0.434 against 0.437, so near-duplicate titles do not inflate the result.
- **Test set:** R²_log 0.426 on the held-out games, against 0.437 in cross-validation.

**What failed and what it taught us.**
- **Decision Tree (default settings)** stays worse than the mean on every target and population
  (R²_log −0.15; training R²_log ≈ 1.0). An unrestricted tree memorises games; depth limits or
  ensembling are required.
- **Linear Regression on the log target** reaches R²_log 0.291 but has negative R²_orig (−0.007): a
  single linear surface on log owners cannot represent the non-linear effects (e.g. localisation
  peaking at 9–15 languages) and under-predicts large games.
- **Raw-target models** (the baseline) are worse than a constant on the log scale: minimising
  squared error in owners makes them over-predict the typical game about fivefold.
- **Smearing** improves squared-error metrics but worsens MAE: no single back-transformation is best
  for both loss functions.
- **Removing popular games** (notebook 04) raised raw-scale R² but is not a usable modeling strategy,
  because popularity is unknown before release; thresholds below 10M owners gave no further benefit.

## 4. Track B (games below 1M owners) — separate question

For the population in the original 0.255 result, the best model is E03 (HistGB, log target,
smearing): CV R²_orig 0.263 ± 0.019 (+0.008 over 0.255, below the 0.01 threshold and within one
fold SD), R²_log 0.343, MAE 0.039 M, RMSE 0.093 M; test R²_orig 0.241 against 0.212 for the
baseline. This result describes only games that turn out to sell fewer than 1M copies and must not
be compared with Track A.

## 5. Effect of blockbuster games

- The 66 training games with 10M+ owners (0.11%) produce **93%** of original-scale squared error.
- Predictions are compressed towards the typical game: the smallest games are over-predicted by
  about 1.5×, and games with 50–150M owners are under-predicted by a factor of 180–330.
- The largest misses fall into three groups whose success depends on factors absent from the store page:
  - **Free-to-play multiplayer titles** (Apex Legends, PUBG, Dota 2, Lost Ark, Team Fortress 2), where
    network effects and free acquisition drive owner counts;
  - **Viral hits** (Among Us, Fall Guys, Unturned), driven by streaming and word of mouth after launch;
  - **Major franchises** (Call of Duty, Hogwarts Legacy), driven by brand and marketing budget.
- Consequently, original-scale R² reflects a few dozen games and is unstable: it varies by 0.13
  across grouped folds, and the baseline Random Forest's test R²_orig of 0.34 against its CV value
  of 0.01 is an artefact of which blockbusters fall into the test set. **R² on the log scale is the
  appropriate headline metric** for this target; original-scale MAE and RMSE should be reported with it.

## 6. Limitations

- **What R²_log means:** 44% of the variance in log owners is explained, i.e. relative accuracy. The
  typical error is still large. On the natural-log scale, the median absolute error is 0.39 (a
  factor of about 1.5 in owners) and the mean is 0.65 (a factor of about 1.9).
- **Target measurement:** SteamSpy owners are estimates reported in ranges; the target takes only 13
  distinct values, which caps achievable accuracy.
- **Blockbusters cannot be predicted** from store-page information; the model is unsuitable for
  forecasting the size of a potential hit.
- **Random split:** games from all years are mixed across folds, as in the parent paper's protocol.
  A forecast for future releases would be validated chronologically (earlier years for training,
  later for testing), which generally gives lower scores.
- **Snapshot values:** store-page fields are observed in 2025; a few may have changed after launch.
- **No hyperparameter tuning** was performed; results are for default settings.

## 7. Recommended next research direction

1. **Chronological validation** (train on releases up to 2021, test on 2022–2024) to measure forecasting
   performance for future games. It should be reported alongside, not instead of, the current protocol.
2. **Additional pre-release information**, one feature group per experiment:
   - developer and publisher track record, computed only from earlier releases;
   - store-page lead time;
   - franchise or sequel indicators.

   Notebook drafts in `notebooks/drafts/v2_engineered/` suggest lead time and localisation are the
   most informative groups.
3. **Modest tuning of HistGradientBoosting** with nested cross-validation (learning rate, leaf size,
   iterations), to keep the estimate unbiased.
4. **Blockbuster identification as its own task:** a classifier for the probability of reaching 1M+
   owners, evaluated with ROC-AUC and precision at the top of the ranking, rather than forcing one
   regression to fit both typical games and outliers.

## 8. Update: new pre-release variables (E06–E10, notebook 07)

Sections 1–7 describe the model with the original 42 variables. Items 1 and 2 of Section 7 have since
been carried out.

- **New reference model:** the same HistGradientBoosting on log1p(owners) with 49 variables: the original
  42, studio experience (5) and store-page lead time (2). CV R²_log 0.460 ± 0.009 (was 0.437), test 0.446
  (was 0.426), CV MAE 0.103 M owners. Grouped folds by title: 0.455.
- **Not adopted:** franchise and title variables (+0.001) and store-page extras (−0.001).
- **Chronological validation:** fitted on releases up to 2021 and scored on 2022 to June 2024, R²_log is
  0.161 with 42 variables and 0.165 with 49. Random folds overstate forecasting accuracy (see Section 9
  for the causes).
- **Past sales of a studio's earlier games (E10, sensitivity only):** chronological R²_log 0.165 → 0.252.
  Not adopted, because the owner counts are a 2025 snapshot and include sales made after the predicted
  game's release.
- **Next (at the time):** reduce the dependence on release year, for example with a target relative to the
  release-year cohort. This was tested in Section 9.

## 9. Update: forecasting later releases (E11–E14, notebook 08)

- **Where the gap between 0.460 and 0.165 comes from:** −0.162 appears when only 2022–2024 games are scored,
  even with random folds (recent games mostly share the lowest owner range and are harder to tell apart);
  −0.083 from having no games of those years in training; −0.051 from the level error (predictions about 23%
  too high). The release-year level is the smallest part.
- **Changes tested, none adopted:** removing release year (chronological R²_log −0.011), a year-relative
  target (+0.004), fitting on 2017–2021 only (+0.006, within seed noise), and a linear trend for the year
  level (+0.049 on the main split, but worse in the year-by-year check: 0.248 against 0.285).
- **Forecasting accuracy to report:** each of 2022–2024 predicted from all earlier years gives R²_log 0.246,
  0.234 and 0.277 (mean 0.253), with predictions roughly 10–15% too high; random folds give 0.295 on the same games.
- **Reference model:** unchanged (49 variables, Section 8).
- **Next:** a classification task for recent games (above the lowest owner range or not), past-sales
  information dated before each release, then tuning scored one year ahead.

## 10. Update: the low group and the breakout group (E15–E16, notebook 09)

- **Why:** 69.7% of games sit in the lowest owner range (0–20,000) and share one target value.
- **Profile:** with release year held equal, breakout games have longer store-page lead times, higher prices,
  fuller pages, more languages and online play, and are less often self-published, Casual or VR-only. The
  largest difference is 0.37 SD, so no single indicator separates the groups.
- **E15, classifying the group:** ROC-AUC 0.805 with random folds (0.755 inside release years; test 0.801) and
  0.71–0.75 one year ahead. Of the tenth of games ranked highest, 72.7% break out with random folds and 51–62%
  one year ahead, about three times the base rate. Probabilities run high for later years.
- **E16, two-stage prediction:** R²_log 0.461 against 0.460 with random folds; not adopted.
- **How to state the result:** as a ranking, not as an owner number for one game. For recent games no model
  here decides reliably which side of 20,000 owners a single game lands on.
- **Reference model:** unchanged (49 variables, Section 8).

## 11. Update: forecasting from the predecessor (E17–E19, notebook 10)

Approach borrowed from Schaer, Kourentzes & Fildes (EJOR 2026): forecast a new game from an analogy, its
predecessor.

- **Predecessors:** 48% of games have one (series 6.9%, developer's previous game 33.0%, publisher's 8.3%).
  After a low-group predecessor 18% of games rise above 20,000 owners; after one with more than 1M owners, 88%.
- **E17, the predecessor alone:** smaller absolute error than the 49-variable model on games with a
  predecessor (MAE_log 0.588 against 0.614; series 0.502 against 0.552) but much lower R².
- **E18–E19, predecessor and studio averages in the model (adopted):** random-fold R²_log 0.460 → 0.502,
  one-year-ahead 0.250 → 0.311, test 0.446 → 0.486.
- **Who can be forecast:** one year ahead, R²_log is 0.59 with a series predecessor, 0.40 with a developer's
  previous game and 0.09 with no predecessor (52% of games).
- **Not reproduced:** the paper's pre-release buzz model; no search data, and lead time fails as a substitute.
- **Limitation:** predecessor owner counts are the 2025 snapshot. The gain persists when only predecessors
  released one to three years earlier are used.
- **Reference models:** the analogy model (59 variables) and, for comparison, the store-page model (49).

