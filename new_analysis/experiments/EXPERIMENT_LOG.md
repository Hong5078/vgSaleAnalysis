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
