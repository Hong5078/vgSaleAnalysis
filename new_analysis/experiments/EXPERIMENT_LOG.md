# Experiment Log — Steam Owners Prediction

Persistent record of every experiment. Raw numbers: `results/experiment_log.csv` (append-only) and
one JSON file per run in `results/`. Reproduce any experiment with `python experiments.py <ID>`.

## Protocol (fixed before the first experiment)

| Item | Setting |
|---|---|
| Data | `data/processed/steam_model_table.csv.gz` from notebook 01 (76,727 labeled games, released up to 30 June 2024) |
| Split | Same as notebooks 03–04: `train_test_split(test_size=0.2, random_state=42)` |
| Validation | 5-fold `KFold(shuffle=True, random_state=42)` on the training portion only |
| Test set | Not used by any experiment; reserved for one final check of the selected model |
| Features (base) | The 42 release-time features of notebook 01 (9 numeric, 15 genre flags, 18 category flags) |
| Metrics | Log scale: R²_log, MAE_log on log1p(owners). Original scale: R²_orig, MAE_orig, RMSE_orig on owners in millions. Raw-target models are scored on the log scale via log1p(max(prediction, 0)); log-target models are scored on the original scale via expm1(prediction) |
| Main result | Mean ± SD of the five validation folds. Training-fold scores are recorded only as an overfitting diagnostic |

**Populations are separate research questions and are never pooled in comparisons.**
- **Track A — all games.** The population a real pre-release forecast must serve. Primary track.
- **Track B — games below 1M owners** (removed from both training and test data after splitting, as
  in notebook 04). Diagnostic only: popularity is unknown before release, so this population
  cannot be identified in advance.

**Why two R² scales.** R²_orig weights each error by its square in owners, so with 1% of games
holding 56% of all owners it mostly measures accuracy on about twenty blockbusters. R²_log
measures accuracy in relative terms (a prediction off by a factor of two costs the same for a
small and a large game) and describes the typical game. The two are different quantities and are
never compared with each other; every experiment reports both.

## Summary table

| ID | Change (one factor) | Track | Model | CV R²_log (± SD) | CV R²_orig (± SD) | MAE_orig (M) | RMSE_orig (M) | Δ vs best (primary metric) |
|---|---|---|---|---|---|---|---|---|
| E00 | Reproduction of notebooks 03–04 (raw target) | A | Random Forest | −0.119 ± 0.040 | 0.013 ± 0.044 | 0.155 | 1.485 | reference |
| E00 | 〃 | A | Linear Regression | −24.2 ± 0.68 | 0.044 ± 0.019 | 0.233 | 1.467 | reference |
| E00 | 〃 | B | Random Forest | −0.113 ± 0.027 | 0.255 ± 0.025 | 0.044 | 0.094 | reference |
| E01 | Target: log1p(owners) | A | Random Forest | **0.414 ± 0.009** | 0.056 ± 0.037 | 0.105 | 1.460 | +0.533 R²_log; +0.043 R²_orig |
| E01 | 〃 | A | Linear Regression | 0.291 ± 0.010 | −0.007 ± 0.023 | 0.114 | 1.503 | — |
| E01 | 〃 | B | Random Forest | 0.317 ± 0.010 | 0.191 ± 0.008 | 0.034 | 0.098 | −0.064 R²_orig vs E00-B |
| E02 | Model: HistGradientBoosting (defaults) | A | HistGB | **0.437 ± 0.011** | 0.058 ± 0.029 | 0.104 | 1.457 | +0.023 R²_log (new best) |
| E02 | 〃 | B | HistGB | 0.343 ± 0.008 | 0.190 ± 0.006 | 0.034 | 0.098 | — |
| E03 | Back-transform: Duan smearing | A | HistGB | 0.437 ± 0.011 (unchanged) | **0.088 ± 0.044** | 0.115 | **1.435** | +0.030 R²_orig |
| E03 | 〃 | B | HistGB | 0.343 ± 0.008 | 0.263 ± 0.019 | 0.039 | 0.093 | +0.008 R²_orig vs E00-B (below the 0.01 threshold) |
| E06 | Features: + studio experience (5) | A | HistGB | 0.450 ± 0.008 | 0.067 ± 0.038 | 0.104 | 1.451 | +0.013 R²_log (adopted) |
| E07 | Features: + franchise and title (4) | A | HistGB | 0.451 ± 0.008 | 0.078 ± 0.046 | 0.103 | 1.443 | +0.001 vs E06 (not adopted) |
| E08 | Features: + store-page lead time (2) | A | HistGB | **0.460 ± 0.009** | 0.086 ± 0.042 | 0.103 | 1.436 | +0.010 vs E06 (adopted; new best) |
| E09 | Features: + store-page extras (4) | A | HistGB | 0.459 ± 0.011 | 0.090 ± 0.048 | 0.103 | 1.434 | −0.001 vs E08 (not adopted) |
| E08 | 〃 | B | HistGB | 0.366 ± 0.006 | 0.205 ± 0.009 | 0.033 | 0.097 | +0.023 R²_log vs E02-B |

---

## E00 — Reproduction check (not counted as an experiment)

- **Purpose:** confirm that the harness reproduces notebooks 03 and 04 before anything is changed.
- **Population / target / features:** Tracks A and B; owners in millions (raw); base 42 features.
- **Models:** mean baseline, Linear Regression, Decision Tree, Random Forest, scikit-learn defaults.
- **Result:** identical to the notebooks — Track A: LR 0.0436, DT −1.3756, RF 0.0126; Track B: LR 0.1635,
  DT −0.5012, RF 0.2550 (R²_orig).
- **New information:** scored on the log scale, every raw-target model is worse than a constant
  prediction (RF R²_log −0.119; LR −24.2 because 39% of its predictions are negative and clip to
  zero owners). Raw-target models are therefore poor at relative accuracy: they over-predict the
  typical game to reduce squared error on the blockbusters.

## E01 — Log-transformed target

- **Hypothesis:** owners span four orders of magnitude, so squared-error fitting on the raw scale is
  dominated by about twenty games. Fitting log1p(owners) makes errors proportional, lets the models
  learn the multiplicative relationships shared by most games, and should improve accuracy for the
  typical game without discarding any game.
- **Change:** target only (log1p(owners) instead of owners in millions). Predictions are
  back-transformed with expm1 for the original-scale metrics.
- **Population / features / models:** as E00 (defaults).
- **Result (Track A):** Random Forest R²_log 0.414 ± 0.009 (from −0.119); R²_orig 0.056 ± 0.037
  (from 0.013); MAE_orig 0.105 M (−32%); RMSE_orig 1.460 M (−1.7%). Linear Regression R²_log 0.291,
  MAE_orig 0.114 M (−51%), R²_orig −0.007. Decision Tree still overfits (training R²_log 0.996, CV −0.154).
- **Result (Track B):** Random Forest R²_log 0.317; R²_orig 0.191, lower than the raw-target model's
  0.255, while MAE_orig improves from 0.044 to 0.034 M.
- **Interpretation:** the log target is the appropriate formulation. It improves every metric on
  all games except that the original-scale R² remains low: back-transformed log predictions
  estimate the geometric rather than the arithmetic mean and so under-predict the largest games,
  which dominate R²_orig. In Track B the raw target keeps the higher R²_orig because it directly
  minimises squared error on that scale. Random Forest shows a large gap between training
  (0.915) and validation R²_log (0.414), indicating high variance.
- **Status:** new best for Track A (primary metric R²_log). Stopping condition 1 is met on the log
  scale only; on the original scale R² is 0.056.
- **Next:** (E02) gradient boosting to address the variance shown by Random Forest; (E03) a
  retransformation-bias correction to improve the original-scale predictions.

## E02 — Gradient boosting instead of Random Forest

- **Hypothesis:** the default Random Forest grows unrestricted trees and overfits (E01: training
  R²_log 0.915 against 0.414 in validation). Gradient boosting builds shallow trees sequentially with
  a minimum leaf size and early stopping, which should reduce variance while still capturing
  non-linear effects and interactions.
- **Change:** model family only — `HistGradientBoostingRegressor` with scikit-learn defaults
  (learning rate 0.1, up to 100 iterations, 31 leaves per tree, at least 20 games per leaf, early
  stopping on an internal 10% split of each training fold, `random_state=42`). Target and features as E01.
- **Result (Track A):** R²_log 0.437 ± 0.011 (+0.023 over E01); R²_orig 0.058; MAE_orig 0.104 M;
  RMSE_orig 1.457 M. Training R²_log 0.494, so the training–validation gap falls from 0.50 to 0.06.
  Runtime 3.5 s against 16 s for Random Forest.
- **Result (Track B):** R²_log 0.343 (+0.026 over E01's Random Forest); R²_orig 0.190.
- **Interpretation:** the improvement comes with far less overfitting, so it reflects better
  generalisation rather than a lucky configuration. Defaults were not tuned; the gain exceeds the
  0.01 threshold without any search.
- **Status:** new best for Track A.

## E03 — Retransformation-bias correction (Duan smearing)

- **Hypothesis:** exponentiating a log-scale prediction estimates the geometric mean (the typical
  value), which is systematically below the arithmetic mean for a right-skewed outcome. This
  compresses predictions for large games and depresses R²_orig. Multiplying the back-transformed
  predictions by Duan's smearing factor, the mean of exp(residual) over the training fold, should
  raise R²_orig and lower RMSE_orig. MAE_orig is expected to worsen, because absolute error is
  minimised by the median rather than the mean.
- **Change:** back-transformation only (factor computed on each training fold's residuals; about 1.9
  for Track A). The fitted model and the log-scale predictions are identical to E02.
- **Result (Track A):** R²_orig 0.058 → 0.088; RMSE_orig 1.457 → 1.435 M; MAE_orig 0.104 → 0.115 M;
  R²_log unchanged (0.437).
- **Result (Track B):** R²_orig 0.190 → 0.263, just above the raw-target Random Forest's 0.255
  (+0.008, below the 0.01 threshold and within one fold SD); RMSE_orig 0.098 → 0.093; MAE_orig 0.034 → 0.039.
- **Interpretation:** confirmed as predicted. The two back-transformations serve different loss
  functions: no correction for absolute error (typical game), smearing for squared error. Even with
  the correction, original-scale R² for all games stays below 0.1, because the largest games are
  under-predicted by one to two orders of magnitude (see blockbuster analysis below).

## Stopping decision

Stopping condition 1 (a valid model with mean CV R² ≥ 0.30) was met at E01 and strengthened at E02,
**on the log scale** (R²_log 0.437). It is not met on the original owner scale, where the best CV
R² is 0.088 (E03). As allowed, two obvious follow-ups (E02, E03) were tested after the condition was
first met; no further tuning or feature experiments were run. Meaningful experiments completed: 3.

## Validity checks and test evaluation (`final_checks.py`)

**Grouped cross-validation.** Folds grouped by normalised title (so re-releases and editions sharing
a title cannot be split across training and validation):

| Track | Folds | R²_log | R²_orig (smearing) |
|---|---|---|---|
| A | KFold (protocol) | 0.437 ± 0.011 | 0.088 ± 0.044 |
| A | GroupKFold by title | 0.434 ± 0.012 | 0.158 ± 0.128 |
| B | KFold (protocol) | 0.343 ± 0.008 | 0.263 ± 0.019 |
| B | GroupKFold by title | 0.339 ± 0.012 | 0.248 ± 0.035 |

Grouping changes R²_log by less than 0.005, so near-duplicate titles do not inflate the result.
R²_orig moves by 0.07 with a fold SD of 0.13, which again shows that original-scale R² depends on
which blockbusters fall into each fold.

**Test set (used once, after all decisions).** Model fitted on the full training portion:

| Track | Model | R²_log | R²_orig | MAE_orig (M) | RMSE_orig (M) |
|---|---|---|---|---|---|
| A | HistGB, log1p target (E02) | 0.426 | 0.075 | 0.110 | 1.750 |
| A | HistGB + smearing (E03) | 0.426 | 0.126 | 0.123 | 1.702 |
| A | Random Forest, raw target (notebook baseline) | −0.117 | 0.340 | 0.157 | 1.479 |
| B | HistGB + smearing (E03) | 0.326 | 0.241 | 0.039 | 0.093 |
| B | Random Forest, raw target (notebook baseline) | −0.153 | 0.212 | 0.044 | 0.095 |

Test R²_log (0.426) agrees with cross-validation (0.437). On the original scale the baseline Random
Forest has the higher test R² and lower RMSE for Track A, but this is the split-specific effect
documented in notebook 03 (its cross-validated R²_orig is 0.013): it places a few test blockbusters
such as Counter-Strike 2 closer to their true size.

**Blockbuster analysis (E02 out-of-fold predictions).** The 66 training games with 10M+ owners (0.11%)
account for 93.4% of original-scale squared error. Predictions are compressed towards the typical
game: the lowest range is over-predicted by about 1.5× (mean log error +0.43), and games with
50–150 million owners are under-predicted by a factor of 180–330 (log error −5.2 to −5.8). The
largest misses are free-to-play multiplayer titles (Apex Legends, PUBG, Dota 2, Lost Ark,
Team Fortress 2), viral hits (Among Us, Fall Guys, Unturned), major franchises (Call of Duty,
Hogwarts Legacy) and one non-game utility that passed the software filter (Wallpaper Engine). Their
store-page features do not distinguish them from ordinary releases; what sets them apart (franchise
strength, marketing budget, network effects, virality) is not in the dataset.

---

## Second parent paper: approach applied (E04, E05)

Source: Bondarenko, Cabrera & Fitzek, IEEE SmartNets 2026. Three devices borrowed: separate paid and
free-to-play models, price bands, and percentage-change reporting from a log-linear outcome. Its own
variables (wishlists, 30-day rating) are unavailable or post-release, so only the approach is used.
Full write-up and tables: `paper2_style/PAPER2_STYLE_ANALYSIS.md`; script `paper2_style_analysis.py`.
Same protocol as every other experiment (training portion, 5-fold KFold, seed 42, log1p owners).

### E04 — Separate models for paid and free-to-play games

- **Hypothesis:** price does not apply to free-to-play games, and the earlier error analysis found they
  behave differently (MAE_log 0.94 against 0.63 for paid games). Fitting one model per segment lets
  every coefficient differ, which should help if the segments follow different relationships.
- **Change:** segmentation only (paid: price > 0, n = 68,380; free: price = 0, n = 8,347). Features,
  folds and models unchanged; predictions from the two models are pooled for the overall metric.
- **Result (CV R²_log, all games):** Linear Regression 0.291 → 0.297; HistGradientBoosting 0.437 → 0.433.
  On free-to-play games alone: Linear Regression 0.283 → 0.314 (MAE 0.943 → 0.921);
  HistGradientBoosting 0.400 → 0.376.
- **Interpretation:** separation helps the linear model, where it acts as a manual interaction, but not
  gradient boosting, which already models interactions and loses training data when split (the
  free-to-play model sees 8,347 games instead of 61,381). Useful for interpretation, not for accuracy.

### E05 — Price bands instead of a continuous price

- **Hypothesis:** the descriptive tables show a non-monotone price–sales relationship (rising to the
  $40–59.99 band, reversing above $80), which a single linear slope cannot represent.
- **Change:** the `price` column replaced by six band dummies (paper's bands plus $80+).
- **Result (CV R²_log, all games):** Linear Regression 0.291 → 0.309 (+0.018); with separate models as
  well, 0.315 (+0.023 over the pooled continuous-price baseline). HistGradientBoosting 0.437 → 0.429 (−0.008).
- **Interpretation:** banding supplies the non-linearity a linear model lacks; a tree-based model finds
  its own thresholds and is slightly hurt by the coarser encoding.

**Reported effects (log-linear OLS, percentage changes).** Paid games, relative to $0.99–9.99:
+38% ($10–19.99), +142% ($20–29.99), +212% ($30–39.99), +257% ($40–59.99), −49% ($80+), all p < 0.001;
online co-op +69%, full controller support +21%, each supported language +1.8%, VR-only −27%.
Free-to-play: massively multiplayer +34%, online PvP +32%, each supported language +7.2%, Indie −19%.
With a continuous price, +$10 is associated with **+9.8% owners**, the opposite sign to the paper's
−15% per $10 on wishlist conversion; the outcomes differ (total owners versus conversion of existing
wishlisters), so both can hold.

## Data version pinning (2026-09-27)

The Kaggle dataset was re-scraped (version 51 appeared). All results in this project use **version 49**,
so `kagglehub.dataset_download(...)` is now pinned to `fronkongames/steam-games-dataset/versions/49`
in notebook 01 and in the analysis scripts. Notebook 01 was re-run after pinning and the modeling table
is byte-identical (76,727 × 47, checksum edd5c72d781c2c3563c337409ef686e5).

---

## New pre-release variables (E06–E09) and past sales (E10)

Notebook: `notebooks/07_modification_new_prerelease_variables.ipynb`, which builds the variables from the
raw `games.json` (version 49) and writes `data/processed/engineered_features.csv.gz`. Run it before
`python experiments.py E06` … `E10`. Same protocol as every other experiment; model and target as in E02.

**Rules fixed in advance.** One group per experiment, in the order A, B, C, D. A group is adopted only if
it raises CV R²_log by at least 0.01 over the last adopted set. All variables must be known at release.

| Group | Variables |
|---|---|
| A. Studio experience | log earlier releases by the developer and by the publisher, developer's first game, self-published, number of developers |
| B. Franchise and title | sequel marker, subtitle, words in title, log earlier games with the same title stem by the same developer or publisher |
| C. Store-page lead time | log days between the estimated store-page creation (from AppID order) and release, missing flag |
| D. Store-page extras | website listed, support contact listed, English supported, log releases in the same week |

### E06 — Studio experience

- **Hypothesis:** the largest errors are games from established studios; release history is public before launch.
- **Result (Track A):** HistGB R²_log 0.437 → 0.450 (+0.013), higher in 5 of 5 folds. Linear Regression 0.291 → 0.302.
- **Outcome:** adopted.

### E07 — Franchise and title

- **Hypothesis:** sequels and series entries inherit an audience.
- **Result:** 0.450 → 0.451 (+0.001), higher in 3 of 5 folds. Linear Regression 0.302 → 0.305.
- **Outcome:** not adopted. Titles with a sequel marker do sell more (22.0% reach 100k owners against 10.1%),
  but only 4.9% of games have one and 7.1% continue an earlier title, so few predictions change.

### E08 — Store-page lead time

- **Hypothesis:** a longer time on the store before release gives more time to collect attention; it is the
  closest available substitute for the second parent paper's wishlist counts.
- **Result:** on top of E06, 0.450 → 0.460 (+0.0101), higher in 5 of 5 folds. Linear Regression 0.302 → 0.313.
- **Outcome:** adopted, just above the threshold. New reference model: 49 variables.
- **Caveat:** the page-creation date is estimated from AppID order, and Early Access games may have sold
  before their recorded release date, which lengthens their lead time.

### E09 — Store-page extras

- **Result:** on top of E08, 0.460 → 0.459 (−0.001), higher in 2 of 5 folds. Linear Regression 0.313 → 0.325.
- **Outcome:** not adopted. Helps the linear model only; these fields are also read at the 2025 snapshot.

### Validity checks for the adopted model (base 42 + A + C)

- **Grouped folds by title:** R²_log 0.455 against 0.460 with random folds (Notebook 5 model: 0.434 against 0.437).
- **Test set (used once for this model):** R²_log 0.446 against 0.426 for E02; MAE_log 0.637 against 0.653;
  MAE_orig 0.108 M against 0.110 M.
- **Permutation importance by group:** lead time 0.052 (4th of 12) and studio experience 0.030 (5th), after
  release year 0.350, localisation 0.133 and price 0.105.
- **By segment (out-of-fold):** paid 0.438 → 0.461; free-to-play 0.400 → 0.422.

### E10 — Past sales of the studio's earlier games (sensitivity, chronological split)

- **Variables:** mean and maximum log1p(owners) of the developer's, and the publisher's, earlier labeled
  games (4 variables; available for 48.2% of games).
- **Why not in the main model:** the owner counts are a 2025 snapshot, so they include sales made after the
  predicted game's release. Under random folds, one game's target would also be another game's feature.
- **Design:** training portion only; fit on releases up to 2021 (38,450 games), score on 2022 to June 2024
  (22,931 games). Raw numbers: `results/E10_chronological_past_sales.csv`.

| Feature set | Chronological R²_log | MAE_log | Random-fold CV R²_log (reference) |
|---|---|---|---|
| Base 42 (E02) | 0.161 | 0.622 | 0.437 |
| Base 42 + A + C (E08) | 0.165 | 0.621 | 0.460 |
| Base 42 + A + C + past sales | 0.252 | 0.576 | 0.499 |

- **Interpretation:**
  - Random folds overstate forecasting accuracy: for releases after the training period, R²_log is 0.16, not
    0.44–0.46. This was first attributed to release year; E11–E14 below show the year level is the smallest
    part of the gap.
  - Groups A and C add almost nothing on the chronological split (+0.004).
  - Past sales add +0.087, more than any pre-release group, but the figure is optimistic because of the
    snapshot problem.
- **Outcome:** not adopted. Chronological R² is reported next to random-fold R² from now on.

---

## Forecasting later releases (E11–E14)

Notebook: `notebooks/08_modification_forecasting_later_releases.ipynb`. Run with `python experiments.py E11`
(E11–E14 run together). Reference: E08 (HistGB, log1p target, 49 variables). Chronological split of the
training portion as in E10; the test set is not used. Raw numbers: `results/E11_E14_chronological.csv` and
`results/E11_E14_one_year_ahead.csv`.

**Adoption rule.** Gain of at least 0.01 in chronological R²_log (mean of five seeds), which must also hold
when each year is predicted from earlier years only.

**Where the gap comes from (E08 model).**

| Games scored | Trained on | R²_log | Within-year R²_log | Bias (log) |
|---|---|---|---|---|
| All training games | random folds | 0.460 | 0.358 | −0.002 |
| Released 2022–2024 | random folds | 0.298 | 0.295 | +0.006 |
| Released 2022–2024 | releases up to 2021 | 0.165 | 0.215 | +0.204 |

Within-year R² removes the release-year mean from actual and predicted values. The gap splits into −0.162
(recent games are harder, even with random folds), −0.083 (no games of those years in training) and −0.051
(level error). Mean log owners falls from 12.38 (2012 releases) to 9.53 (2024), and the share of games above
the lowest owner range from 95% to 15%.

**Experiments (fit up to 2021, score 2022–2024).**

| ID | Change | R²_log (seed 42) | Within-year R²_log | Bias (log) | R²_log, mean of 5 seeds (range) | Gain | Outcome |
|---|---|---|---|---|---|---|---|
| E08 | reference | 0.165 | 0.215 | +0.204 | 0.164 (0.158–0.171) | | |
| E11 | Release year removed | −0.013 | 0.132 | +0.354 | −0.011 | −0.175 | not adopted; random-fold CV 0.460 → 0.357 |
| E12 | Year-relative target (level: last fitting year) | 0.172 | 0.219 | +0.195 | 0.167 (0.162–0.175) | +0.004 | not adopted |
| E12+E11 | Year-relative target, release year removed | 0.127 | 0.168 | +0.183 | 0.121 | −0.043 | not adopted |
| E13 | Fitted on 2017–2021 only | 0.180 | 0.228 | +0.199 | 0.170 (0.162–0.180) | +0.006 | not adopted (within seed noise) |
| E14 | Year-relative target, level from a five-year trend | 0.215 | 0.219 | +0.064 | 0.212 (0.208–0.218) | +0.049 | passes on this split; fails the year-by-year check |

**One year ahead (each year predicted from all earlier years).**

| Release year | Random-fold R²_log | One-year-ahead R²_log | Bias (log) | E14 R²_log | E14 bias (log) |
|---|---|---|---|---|---|
| 2016 | 0.395 | 0.274 | +0.331 | 0.304 | +0.135 |
| 2017 | 0.416 | 0.275 | +0.361 | 0.304 | −0.286 |
| 2018 | 0.434 | 0.309 | +0.272 | 0.204 | −0.462 |
| 2019 | 0.403 | 0.343 | +0.117 | 0.255 | −0.355 |
| 2020 | 0.356 | 0.325 | −0.013 | 0.195 | −0.399 |
| 2021 | 0.304 | 0.276 | +0.012 | 0.207 | −0.280 |
| 2022 | 0.298 | 0.246 | +0.103 | 0.262 | +0.007 |
| 2023 | 0.292 | 0.234 | +0.142 | 0.233 | +0.132 |
| 2024 | 0.296 | 0.277 | +0.091 | 0.269 | +0.076 |

- **Interpretation:**
  - The year-relative target does not help: HistGB with release year as a variable already learns each year's level.
  - Release year also helps within a year (E11), so the link between store-page variables and sales changes over time.
  - E14's trend overshoots once the year level stops falling (2018–2021); its mean R²_log over 2016–2024 is 0.248
    against 0.285 without it.
  - One year ahead, the model reaches 0.253 on average for 2022–2024, against 0.295 with random folds.
  - Seed-to-seed variation on the chronological split is about 0.01.
- **Outcome:** no change adopted; E08 stays the reference. The forecasting figure to report is the
  one-year-ahead R²_log (about 0.25 for 2022–2024), next to the random-fold R²_log (0.460).

---

## The low group and the breakout group (E15–E16)

Notebook: `notebooks/09_modification_low_bracket_and_breakout.ipynb`. Run with `python experiments.py E15`
(E15 and E16 run together). Raw numbers: `results/E15_breakout_classifier.csv`, `results/E16_two_stage.csv`.

**Groups.** Low group: the 0–20,000 owner range, 53,446 games (69.7%). Breakout group: all others, 23,281 games
(30.3%). The group is the outcome, so it cannot be used to filter games before release; it can only be
predicted. Games inside the low group share one target value and cannot be told apart.

**Profile (release-year mix held equal).** Largest standardised differences, breakout against low group:
store-page lead time +0.37, Steam Cloud +0.36 (may partly follow success; pages are a 2025 snapshot),
free-to-play +0.28, self-published −0.25, description length +0.24, Casual −0.24. No variable differs by more
than 0.37 SD.

### E15 — Classifying the group

- **Change:** HistGradientBoostingClassifier (defaults) on the 49 variables of E08; outcome: above 20,000 owners.
- **Result, random folds:** ROC-AUC 0.805; within release years 0.755; precision among the top 10% of each
  year 72.7% (base rate 30.5%); Brier 0.151. Test set (used once): 0.801, 0.753, 72.1%.
- **References:** Logistic Regression 0.767 (within-year 0.711). The E08 regression used only to rank: 0.798
  (within-year 0.747, top-10% precision 72.1%).
- **100,000 owners or more:** ROC-AUC 0.886 (within-year 0.851); top-10% precision 45.3% (base rate 10.7%).
- **One year ahead:**

| Release year | Base rate | ROC-AUC | Precision in top 10% | Mean predicted probability |
|---|---|---|---|---|
| 2022 | 24.3% | 0.714 | 62.0% | 28.4% |
| 2023 | 19.5% | 0.751 | 59.9% | 26.2% |
| 2024 | 15.5% | 0.754 | 51.1% | 20.2% |

- **Interpretation:** the classifier ranks games usefully (top tenth breaks out about three times as often as
  average) but is no more accurate than the regression used as a ranker. Its probabilities are calibrated
  with random folds and too high for later years.
- **Odds of breaking out (logistic regression with release-year controls).** Paid: lead time 2 years or more
  +180%, price $20–59.99 +125% to +159%, online co-op +94%, RPG +42%, self-published −35%, VR only −59%,
  price $80+ −94%. Free-to-play: lead time 2 years or more +106%, online PvP +58%, developer with 10+ earlier
  releases +58%, self-published −20%, VR only −33%.

### E16 — Two-stage prediction

- **Change:** stage 1 = E15 probability p; stage 2 = HistGB regression of log owners fitted on breakout games
  only. Combined as p × stage 2 + (1 − p) × log1p(10,000), or by a cut at p = 0.5.
- **Adoption rule:** gain of at least 0.01 in random-fold R²_log that also holds one year ahead.

| Model | Random folds R²_log | MAE_log | One year ahead R²_log | MAE_log | Correct side of 20,000, one year ahead |
|---|---|---|---|---|---|
| Always the low group | −0.306 | 0.687 | −0.184 | 0.399 | 80.2% |
| Single regression (E08) | 0.460 | 0.634 | 0.250 | 0.556 | 78.7% |
| Two-stage, expected value | 0.461 | 0.632 | 0.265 | 0.552 | 79.4% |
| Two-stage, cut at 50% | 0.308 | 0.531 | 0.027 | 0.400 | 81.9% |

(One year ahead: mean of 2022–2024 and five seeds.)

- **Interpretation:** the expected-value version gains +0.001 with random folds and +0.015 one year ahead; the
  cut version lowers absolute error but loses R². For recent games a yes/no call barely beats calling every
  game low (81.9% against 80.2%).
- **Outcome:** not adopted. E08 stays the reference. Results are to be stated as a ranking (breakout share among
  the top-ranked games against the base rate), not as an owner number for one game.

---

## Forecasting from the predecessor (E17–E19)

Notebook: `notebooks/10_modification_predecessor_analogy.ipynb`, which writes
`data/processed/predecessor_features.csv.gz`. Run with `python experiments.py E17` (E17–E19 run together). Raw
numbers: `results/E17_E19_predecessor.csv`.

Source: Schaer, Kourentzes & Fildes, *Analogy-based life cycle forecasts with pre-release buzz*, European
Journal of Operational Research 333 (2026) 88–100. Borrowed: the analogy (a game's predecessor) and the
benchmark "new game = predecessor's sales". Not available: Google search interest before release.

**Predecessors.** Series predecessor (same title stem, same developer or publisher): 6.9% of games.
Developer's previous game: 33.0%. Publisher's previous game only: 8.3%. None: 51.8%. Correlation of log owners
with the predecessor: 0.78, 0.56, 0.51. Share above 20,000 owners by the predecessor's owners: 18.0% (low
group), 49.4% (20,000–100,000), 75.6% (100,000–1M), 88.3% (above 1M); no predecessor 25.5%.

**Adoption rule.** Gain of at least 0.01 in random-fold R²_log that also holds one year ahead.

### E17 — The predecessor alone (the paper's benchmark)

One year ahead, mean of 2022–2024, games with a predecessor only.

| Games | Forecast | R²_log | MAE_log | Bias (log) | MAE relative to benchmark |
|---|---|---|---|---|---|
| Series predecessor | Predecessor's owners | 0.160 | 0.502 | +0.204 | 1.00 |
| Series predecessor | Adjusted for release year | 0.191 | 0.500 | +0.092 | 1.00 |
| Series predecessor | × √(change in lead time) | −0.671 | 0.987 | +0.500 | 1.97 |
| Series predecessor | E08 (49 store-page variables) | 0.459 | 0.552 | +0.138 | 1.10 |
| Any predecessor | Predecessor's owners | −0.138 | 0.588 | +0.115 | 1.00 |
| Any predecessor | E08 (49 store-page variables) | 0.332 | 0.614 | +0.132 | 1.04 |

- **Interpretation:** the predecessor alone has a smaller absolute error than E08 but a much lower R²; it is
  right for the 60% of games that stay in the predecessor's range and far off otherwise. Lead time is not a
  substitute for search interest in the paper's formula.

### E18 — Predecessor variables in the model; E19 — plus studio averages

| ID | Variables | Random-fold R²_log | Gain | One-year-ahead R²_log | Gain | One-year-ahead MAE_log | Test R²_log |
|---|---|---|---|---|---|---|---|
| E08 | 49 store-page variables | 0.460 | | 0.250 | | 0.556 | 0.446 |
| E18 | + 6 predecessor variables | 0.490 | +0.030 | 0.292 | +0.041 | 0.531 | 0.475 |
| E19 | + 4 studio averages (E10) | **0.502** | +0.011 | **0.311** | +0.020 | 0.522 | **0.486** |

One-year-ahead R²_log by the predecessor available (E08 → E18 → E19): series 0.455 → 0.560 → 0.588;
developer's previous game 0.296 → 0.364 → 0.396; publisher's previous game 0.281 → 0.313 → 0.355; none
0.090 → 0.092 → 0.089.

**Snapshot check (E18).** Predecessors restricted to games released at least 0 / 1 / 2 / 3 years before:
coverage 48.2% / 36.6% / 28.1% / 21.6%; one-year-ahead gain +0.041 / +0.035 / +0.030 / +0.024. The gain per
game covered does not fall (0.086 → 0.109).

**Ranking, one year ahead.** Breakout share in the top tenth: 57.1% (E08), 60.4% (E18), 61.8% (E19); base rate 19.8%.

- **Outcome:** E18 and E19 adopted. New reference ("analogy model"): HistGB on log1p(owners) with 59 variables.
  E08 is kept as the store-page model, which uses nothing measured after any release. This reverses the E10
  decision; the limitation that predecessor owner counts are a 2025 snapshot is stated with every result.

