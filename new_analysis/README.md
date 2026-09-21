# Pre-Release Sales Prediction for Steam Games

**Question.** How accurately can the commercial performance of a PC game be predicted using only
information available before or on its release day?

**Relation to the parent paper.** Gray, Shalan & Kadlec (IEEE CICN 2024) predicted video game sales
with Linear Regression and a Decision Tree and reported R² = 1.000, using regional sales (which sum
to the target) and post-release review counts as predictors. This study keeps their modeling
methods, adds Random Forest and 5-fold cross-validation, and restricts every predictor to
information that exists at or before launch.

**Data.** Kaggle [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)
(fronkongames, version 49): Steam store pages plus SteamSpy owner estimates. It is downloaded
automatically by notebook 01 with `kagglehub` and is not stored in git.

## Pipeline

| Stage | Notebook | Content |
|---|---|---|
| 1. Preprocessing | [`01_preprocessing_and_transformation`](notebooks/01_preprocessing_and_transformation.ipynb) | Data introduction, data wrangling, label validation, leakage audit |
| 2. Transformation | 01 | Target construction, pre-release and release-day indicators |
| 3. Analysis | [`02_exploratory_analysis`](notebooks/02_exploratory_analysis.ipynb) | Descriptive statistics, target distribution, temporal structure, indicator relationships |
| 4. Modeling | [`03_modeling_validation_decision`](notebooks/03_modeling_validation_decision.ipynb) | Data split, baseline, Linear Regression, Decision Tree, Random Forest |
| 5. Validation | 03 | 5-fold grouped cross-validation, chronological hold-out test (2022–2024) |
| 6. Decision making | 03 | Model selection, key drivers, screening use case, comparison with the parent paper |

## How to run

```
pip install kagglehub pandas numpy matplotlib scikit-learn
```

Run the notebooks in order from the `notebooks/` folder. Notebook 01 downloads the data (~300 MB)
and writes `data/processed/`; notebooks 02 and 03 read from there. Figures and tables are saved to
`outputs/figures/` and `outputs/tables/`. Notebook 03 takes several minutes because of the Random
Forest grid search.

`notebooks/drafts/` and `outputs/drafts/` hold earlier exploratory work kept for reference; they
are superseded by the three notebooks above.
