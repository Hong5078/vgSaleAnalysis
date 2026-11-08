# Parent Paper 1: Implementation and Steam Check

**Parent paper:** D. Gray, A. Shalan, and C. Kadlec, "Sales Predictions for Video Games Using Predictive
Analytics of Market Data," *IEEE CICN 2024*, pp. 1510–1516.
[IEEE Xplore](https://ieeexplore.ieee.org/document/10847429), DOI 10.1109/CICN63059.2024.10847429

The paper built its models in Orange Data Mining, a visual tool, so it has no code to download.
[`parent_paper_1_implementation.ipynb`](parent_paper_1_implementation.ipynb) implements its method in
Python. The notebook is saved with its outputs, so the results can be read on GitHub without running it.

## What the notebook does

**Part 1** reproduces the paper on its own dataset, the Kaggle
[Video Game Sales and Ratings](https://www.kaggle.com/datasets/thedevastator/video-game-sales-and-rating) data.
It uses the paper's inputs and Orange's default settings: 10-fold cross-validation, missing values filled
with the column mean, and Orange's default tree settings. It reproduces each numerical table in the paper:

| Paper | Notebook |
|---|---|
| Table I: feature statistics | Same statistics, next to the paper's values |
| Table II: Test and Score results | MSE, RMSE, MAE, MAPE and R², next to the paper's values |
| Table III: Linear Regression coefficients | Coefficients next to the paper's values |
| Table IV: predicted against original sales | First ten games |

**Part 2** runs the same models on the
[Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) (version 49), using the
Steam variables closest to the paper's inputs. All of these variables are measured after release, as the
paper's are.

## Results (R²)

| Model | Paper | Part 1: paper's dataset | Part 2: Steam owners | Part 2: Steam log owners |
|---|---|---|---|---|
| Linear Regression | 1.000 | 1.000 | 0.569 | 0.659 |
| Decision Tree | 0.939 | 0.912 (0.888 to 0.946 over 20 seeds) | 0.448 | 0.495 |

Part 1 matches the paper. Linear Regression reaches R² = 1.000 because `Global_Sales` is the sum of the four
regional sales columns used as inputs. Steam has no inputs that add up to the number of owners, so the same
method falls well short of 1 there.

## Files

| File | Contents |
|---|---|
| `parent_paper_1_implementation.ipynb` | The notebook, saved with outputs |
| `steam_appids.csv.gz` | App IDs of the 76,727 Steam games used in Part 2; these are the games kept by the cleaning in [`new_analysis/notebooks/01`](../new_analysis/notebooks/01_preprocessing_and_transformation.ipynb) |

The paper's dataset is read from
[`../old_analysis/paper_replication/video_game_sales_and_ratings.csv`](../old_analysis/paper_replication/video_game_sales_and_ratings.csv).

## How to run

1. Clone the repository.
2. Install the packages:
   ```
   pip install pandas numpy scikit-learn matplotlib kagglehub
   ```
3. Open the notebook from this folder and run all cells. Part 2 downloads the Steam dataset (version 49)
   through `kagglehub` on the first run.

The notebook runs in about 2 minutes. It was tested with Python 3.10, scikit-learn 1.7.2 and pandas 2.3.3.
