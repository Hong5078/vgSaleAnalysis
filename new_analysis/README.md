# Video Game Sales Prediction on Steam Data

**Objective.** Predict the sales of PC games, measured as estimated Steam owners, from information
available when a game is released.

**Approach.** Applies the modeling approach of Gray, Shalan & Kadlec (IEEE CICN 2024) to a new
dataset:
- **Retained:** Linear Regression and Decision Tree, evaluated with MSE, RMSE, MAE, MAPE and R².
- **Added:** Random Forest and 5-fold cross-validation.
- **Removed:** the parent paper's predictors that are unavailable at release (regional sales, which
  sum to the target, and review counts, which accumulate after release).

This version is the **baseline**: target on its original scale, default model settings, random
80/20 split. Notebook 3 ends with the issues found and the modifications planned for the next
iteration.

**Data.** Kaggle [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
(fronkongames, version 49): Steam store pages plus SteamSpy owner estimates. Downloaded
automatically by notebook 01 with `kagglehub`; not stored in git.

## Pipeline

| Stage | Notebook | Content |
|---|---|---|
| 1. Preprocessing | [`01_preprocessing_and_transformation`](notebooks/01_preprocessing_and_transformation.ipynb) | Data introduction, data wrangling, variable selection |
| 2. Transformation | 01 | Target variable and feature encoding |
| 3. Analysis | [`02_exploratory_analysis`](notebooks/02_exploratory_analysis.ipynb) | Descriptive statistics, target distribution, relationships with sales |
| 4. Modeling | [`03_modeling_validation_decision`](notebooks/03_modeling_validation_decision.ipynb) | Data split, Linear Regression, Decision Tree, Random Forest |
| 5. Validation | 03 | 5-fold cross-validation, held-out test set, error analysis |
| 6. Decision making | 03 | Model comparison, important variables, issues and next steps |

## Modifications

| Notebook | Modification | Result |
|---|---|---|
| [`04_modification_remove_popular_games`](notebooks/04_modification_remove_popular_games.ipynb) | Exclude the most popular games (10M+, 1M+, 100k+ owners); everything else identical to the baseline | Excluding the 86 games with 10M+ owners raises Random Forest CV R² from 0.01 to 0.23; kept as a diagnostic, not adopted, since popularity is unknown before release |
| [`05_modification_log_target_and_boosting`](notebooks/05_modification_log_target_and_boosting.ipynb) | Log-transformed target, then Gradient Boosting, then a smearing correction, one change at a time (experiments E01–E03) | CV R² on the log scale rises from −0.12 (baseline Random Forest) to 0.44 (test 0.43), and MAE falls from 0.155M to 0.104M owners; adopted as the reference model |
| [`06_modification_second_parent_paper`](notebooks/06_modification_second_parent_paper.ipynb) | The second parent paper's approach: separate paid and free-to-play models, price bands, percentage effects (experiments E04–E05) | Helps Linear Regression (CV R² 0.291 → 0.314) but not Gradient Boosting (0.437 → 0.428); kept for interpretation, e.g. +7.2% owners per language for free-to-play games against +1.8% for paid games |
| [`07_modification_new_prerelease_variables`](notebooks/07_modification_new_prerelease_variables.ipynb) | New pre-release variables, one group at a time (experiments E06–E09), a chronological split, and past sales as a sensitivity check (E10) | Studio experience and store-page lead time are adopted: CV R² 0.437 → 0.460 (test 0.446). On releases after the training period R² is only 0.165; a studio's past sales would raise it to 0.252, but they are a 2025 snapshot, so they are not adopted |
| [`08_modification_forecasting_later_releases`](notebooks/08_modification_forecasting_later_releases.ipynb) | Where the drop on later releases comes from, and four changes aimed at it: no release year, a year-relative target, recent training years only, a trend for the year level (experiments E11–E14) | None is adopted. Most of the drop is that recent games are harder to tell apart, not the release-year level. Predicting each year from earlier years gives R² about 0.25 for 2022–2024, against 0.30 with random folds |
| [`09_modification_low_bracket_and_breakout`](notebooks/09_modification_low_bracket_and_breakout.ipynb) | The 70% of games in the lowest owner range against the rest: how they differ, a classifier for the group, and a two-stage prediction (experiments E15–E16) | Breakout games have longer lead times, higher prices, a publisher and online play, but no single variable separates the groups. The classifier reaches ROC-AUC 0.805 (0.71–0.75 one year ahead); of the tenth of games it ranks highest, 51–62% break out one year ahead, about three times the base rate. The two-stage model does not beat the single regression and is not adopted |
| [`10_modification_predecessor_analogy`](notebooks/10_modification_predecessor_analogy.ipynb) | A third paper's approach (Schaer, Kourentzes & Fildes, EJOR 2026): forecast a game from its predecessor (experiments E17–E19) | Adopted. With the predecessor's owners and the studio's averages added, one-year-ahead R² rises from 0.250 to 0.311, random-fold R² from 0.460 to 0.502 and test R² from 0.446 to 0.486. Games with a series predecessor reach 0.59 one year ahead; the 52% of games with no predecessor stay at 0.09 |

## Experiments

[`experiments/`](experiments/) contains a shared evaluation harness, one function per experiment,
an append-only results log and the write-ups:
- [`EXPERIMENT_LOG.md`](experiments/EXPERIMENT_LOG.md): protocol, hypothesis, result and interpretation of each experiment
- [`RESEARCH_SUMMARY.md`](experiments/RESEARCH_SUMMARY.md): best model, comparison with the baseline, blockbuster analysis, limitations
- [`interpretation/INTERPRETATION_SUMMARY.md`](experiments/interpretation/INTERPRETATION_SUMMARY.md): permutation importance, test-set error analysis, error patterns and blockbuster diagnostics for the best model (`interpret_best_model.py`)

Two models are reported, both HistGradientBoosting on log1p(owners):
- **Store-page model (notebook 07):** 49 release-time variables. Cross-validated R² 0.460 on the log scale
  (test 0.446), against −0.119 for the baseline Random Forest. Predicting each of 2022–2024 from earlier
  years gives R² 0.250.
- **Analogy model (notebook 10):** the same plus the owners of each game's predecessor and the studio's
  earlier games, 59 variables. Cross-validated R² 0.502 (test 0.486); one year ahead 0.311. Its extra
  variables are owner counts read in 2025, which include sales made after the predicted game's release.

## How to run

```
pip install kagglehub pandas numpy matplotlib scikit-learn
```

Run the notebooks in order from `notebooks/`. Notebook 01 downloads the data (~300 MB) and writes
`data/processed/`; notebooks 02–10 read from there (07 and 10 also re-read the raw file; 08–10 need 07's feature file). Figures and tables are saved to
`outputs/figures/` and `outputs/tables/`. Each notebook runs in one to three minutes.

## Earlier versions

`notebooks/drafts/` and `outputs/drafts/` keep earlier exploratory work for reference, including
`v2_engineered/`, a version with a year-relative log target, chronological hold-out, tuned models
and additional engineered indicators. Its techniques are candidates for the modifications listed
in Notebook 3, Section 8.3.
