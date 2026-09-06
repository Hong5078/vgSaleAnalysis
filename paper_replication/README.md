# Paper Replication - matches the parent paper's actual dataset/features

Parent paper: Gray, Shalan, Kadlec, "Sales Predictions for Video Games using
Predictive Analytics of Market Data," IEEE CICN 2024.
https://ieeexplore.ieee.org/document/10847429

## Why this folder exists
The paper used the Kaggle "Video Game Sales and Ratings" dataset (16,928 rows -
includes Critic_Score, Critic_Count, User_Score, User_Count, Rating), NOT the
simpler classic vgsales.csv in the parent folder (which has no critic/user
columns). The paper also used Orange Data Mining, a visual/no-code tool, not a
Python script - so there's no original code to "find and run" for this specific
paper. This folder is a from-scratch, code-based re-implementation of what the
paper describes: same feature set, same two models, same evaluation metrics.

## Dataset
video_game_sales_and_ratings.csv - a public mirror of the same dataset lineage
the paper cites (16,719 rows here vs. their 16,928 - Kaggle reposts of this
dataset vary slightly by cleanup pass, but same columns/source data: regional
sales, critic score/count, user score/count, platform, genre, publisher,
developer, ESRB rating, 1980-2020).
Canonical citation for your Task 1/3 write-up:
https://www.kaggle.com/datasets/thedevastator/video-game-sales-and-rating

## What replicate_paper.py does
Reproduces Section III/IV of the paper:
- Features: index, Year_of_Release, NA/EU/JP/Other_Sales, Critic_Score,
  Critic_Count, User_Count (same as the paper's Table III coefficient list)
- Target: Global_Sales
- Models: Linear Regression and Decision Tree Regressor
- Metrics: MSE, RMSE, MAE, MAPE, R^2 (same as the paper's Table II)

## Verified to run - output
    Loaded 16719 rows, 16 columns
    After dropping rows with missing values: 6894 rows

    Linear Regression
      MSE:  0.000   RMSE: 0.006   MAE:  0.004   MAPE: 0.033   R2:   1.000
    Decision Tree
      MSE:  0.051   RMSE: 0.226   MAE:  0.016   MAPE: 0.002   R2:   0.988

    Linear Regression coefficients (all ~1.0 for the four regional sales
    columns, ~0 for everything else) - matches the paper's Table III pattern
    and its conclusion that linear regression outperforms the decision tree.

## The catch (worth mentioning in your own writeup)
Global_Sales is arithmetically the sum of NA+EU+JP+Other_Sales in this dataset,
so a linear model that's handed those four columns as predictors mostly just
re-learns addition - which is why R^2 lands at ~1.000 for both the paper's
version and this replication. It's a legitimate result, but it demonstrates
"the model found the arithmetic identity" more than "the model predicts sales
from the games' features" (genre, platform, critic reception, etc. all get
near-zero coefficients here, same as in the paper).

## How to run it yourself (for your recording)
1. Install Python 3 + packages: pip install pandas scikit-learn numpy
2. Make sure video_game_sales_and_ratings.csv is in the same folder as the script.
3. Run: python replicate_paper.py
4. Record your screen running it top to bottom with no errors.
