# Novel Approach: Modifying the Parent Paper's Method

[`novel_approach.ipynb`](novel_approach.ipynb) starts from the first parent paper's method and changes one block of
the pipeline at a time. The first parent paper is Gray, Shalan and Kadlec (IEEE CICN 2024), which uses Linear
Regression and a Decision Tree on raw sales. Here that method is applied to Steam with release-day information only.

The techniques come from the second and third parent papers, plus two of my own additions. The notebook is saved
with its outputs, so the results can be read on GitHub without running it.

The previous step, reproducing the parent paper, is in
[`../parent_paper_1_implementation/`](../parent_paper_1_implementation/).

## Techniques compared

| Step | Block changed | Technique | Source |
|---|---|---|---|
| Baseline | – | Linear Regression and Decision Tree on raw owners | Parent paper 1 |
| M1 | Target | Log of owners (log-linear model) | Parent paper 2 (Bondarenko et al., IEEE SmartNets 2026) |
| M2 | Segments | Separate models for paid and free-to-play games | Parent paper 2 |
| M3 | Inputs | Price bands | Parent paper 2 |
| M4 | Model | Gradient Boosting | My addition |
| M5 | Segments | Gradient Boosting with paid and free-to-play models (not kept) | Parent paper 2 |
| M6 | Inputs | Studio experience and store-page lead time | My addition |
| M7 | Inputs | Sales of the game's predecessor and the studio's earlier games (analogy) | Parent paper 3 (Schaer et al., EJOR 2026) |

## Training, test and validation sets

The 76,727 games are split once, stratified by owner range, into three files:

| File | Share | Games | Used for |
|---|---|---|---|
| `train.csv.gz` | 70% | 53,708 | Fitting every model |
| `test.csv.gz` | 20% | 15,346 | Comparing models and deciding which changes to keep |
| `validation.csv.gz` | 10% | 7,673 | Unseen data, scored once after the final model is chosen |

Following the course's convention, *test* is the set checked during development and *validation* is the unseen
final set. No game appears in more than one file.

The validation set is not read during development. The Git history shows this order: the split files and the
development results were committed before the validation section was added.

## Results during development (R² on log owners)

| Model | Random folds | Test set | One year ahead |
|---|---|---|---|
| Baseline: Linear Regression, raw owners | −25.47 | −25.58 | −70.65 |
| Baseline: Decision Tree, raw owners | −0.193 | −0.215 | −0.875 |
| M1: Linear Regression, log owners | 0.288 | 0.290 | 0.055 |
| M2: + paid / free-to-play models | 0.294 | 0.296 | 0.063 |
| M3: + price bands | 0.313 | 0.311 | 0.089 |
| M4: Gradient Boosting, log owners | 0.435 | 0.431 | 0.244 |
| M5: M4 + paid / free-to-play models | 0.432 | 0.431 | 0.238 |
| M6: M4 + studio experience and lead time | 0.457 | 0.455 | 0.257 |
| **M7: + predecessor and studio sales (final)** | **0.499** | **0.495** | **0.312** |

**How the models are evaluated:**

- Every model is fitted on the training set.
- **Random folds:** 5-fold cross-validation inside the training set.
- **Test set:** the 20% test set.
- **One year ahead:** each of 2022, 2023 and 2024 is predicted from earlier training-set releases only.

**Where the final model works:** one year ahead, it reaches R² = 0.589 for games with a series predecessor and 0.096
for games with no earlier title from their series, developer or publisher.

## Validation results (unseen 10%, scored once)

After the final model (M7) was chosen on the training and test sets, the validation set was read once. Each model
was fitted on the training set and scored on the 7,673 validation games.

| Model | Test: R² (log) | Validation: R² (log) | Validation: MAE (owners) |
|---|---|---|---|
| Baseline: Linear Regression, raw owners | −25.58 | −25.92 | 195,404 |
| Baseline: Decision Tree, raw owners | −0.215 | −0.219 | 196,448 |
| M3: Linear Regression with Paper 2's changes | 0.311 | 0.314 | 102,562 |
| M4: Gradient Boosting, log owners | 0.431 | 0.440 | 95,099 |
| **M7: final model** | **0.495** | **0.506** | **89,381** |

The validation scores are close to the test scores, and every model keeps its rank. So the improvement holds on
games that were never used during development.

The modeling ideas were first explored in `new_analysis/notebooks/03`–`11`, on a different 80/20 split made
before this validation set existed. This notebook itself never used the validation games for fitting or for
choosing a model.

## Files

| File | Contents |
|---|---|
| `novel_approach.ipynb` | The notebook, saved with outputs |
| `steam_features.csv.gz` | The 59 release-day variables, owners and predecessor type for the 76,727 Steam games |
| `train.csv.gz`, `test.csv.gz`, `validation.csv.gz` | The 70% / 20% / 10% split of `steam_features.csv.gz`, written by Section 2 of the notebook |
| `feature_groups.json` | Which variables belong to each group |

`steam_features.csv.gz` was built by
[`new_analysis/notebooks/01`](../new_analysis/notebooks/01_preprocessing_and_transformation.ipynb),
[`07`](../new_analysis/notebooks/07_modification_new_prerelease_variables.ipynb) and
[`10`](../new_analysis/notebooks/10_modification_predecessor_analogy.ipynb). The source is the
[Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) (Kaggle, version 49, MIT
license). Kaggle no longer offers version 49 for download, so the variables are saved here and the notebook runs
from a fresh clone.

## How to run

```
pip install pandas numpy scikit-learn
```

Open the notebook from this folder and run all cells. It takes about one minute and was tested with Python 3.10
and scikit-learn 1.7.2.
