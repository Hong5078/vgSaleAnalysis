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

## How to run

```
pip install kagglehub pandas numpy matplotlib scikit-learn
```

Run the notebooks in order from `notebooks/`. Notebook 01 downloads the data (~300 MB) and writes
`data/processed/`; notebooks 02 and 03 read from there. Figures and tables are saved to
`outputs/figures/` and `outputs/tables/`. Each notebook runs in about a minute.

## Earlier versions

`notebooks/drafts/` and `outputs/drafts/` keep earlier exploratory work for reference, including
`v2_engineered/`, a version with a year-relative log target, chronological hold-out, tuned models
and additional engineered indicators. Its techniques are candidates for the modifications listed
in Notebook 3, Section 8.3.
