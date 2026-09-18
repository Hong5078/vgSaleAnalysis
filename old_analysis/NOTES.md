# Video Game Sales Prediction - Baseline Code (Task 2/3)

Two code sets are included, covering both replication angles:

## 1. vgsales.csv + vgsales.py / vgsales.ipynb (this folder, top level)
Someone else's existing GitHub code (found, not written by me), verified to run
with no errors. Uses the simpler classic Kaggle "Video Game Sales" dataset
(no critic/user score columns) and compares Decision Tree, Linear Regression,
and Random Forest predicting Global_Sales.
Source: https://github.com/sainathdevulapalli/Video-Game-Sales (CC0 license)

## 2. paper_replication/ (new)
A from-scratch, code-based re-implementation matching the parent paper's exact
dataset and feature set (see paper_replication/README.md for full details). The
parent paper itself used Orange Data Mining (a visual/no-code tool), not a
Python script, so there was no existing code to find for this exact paper -
this recreates what the paper describes as plain, runnable Python instead.

## Parent paper (Task 1)
"Sales Predictions for Video Games Using Predictive Analytics of Market Data"
Gray, D.; Shalan, A. M.; Kadlec, C.
Published in: 2024 IEEE 16th International Conference on Computational Intelligence and
Communication Networks (CICN), Dec 22-23 2024, Indore, India. Added to IEEE Xplore
Jan 27, 2025. DOI: 10.1109/CICN63059.2024.10847429
https://ieeexplore.ieee.org/document/10847429

I've now read the actual paper (you uploaded the PDF) - key facts confirmed
directly from the text:
- Dataset: Kaggle "Video Game Sales and Ratings" (thedevastator), 16,928 rows,
  1980-2020, with regional sales + critic/user score & count + platform/genre/
  rating. Cited in the paper as reference [18]:
  https://www.kaggle.com/datasets/thedevastator/video-game-sales-andrating
- Tool: Orange Data Mining v3.36.2 (visual/no-code workflow), not a script.
- Models: Linear Regression and Decision Tree, evaluated with MSE, RMSE, MAE,
  MAPE, R^2.
- Result: Linear Regression won with R^2 = 1.000 (RMSE 0.005); Decision Tree
  scored R^2 = 0.939 (RMSE 0.383). The paper attributes this to regional sales
  being the dominant predictors of global sales.
- Caveat worth noting in your own writeup: Global_Sales is arithmetically the
  sum of the four regional sales columns, so the near-perfect linear regression
  fit is expected rather than a deep predictive insight - see
  paper_replication/README.md for the full note.

## Dataset (Task 3)
Two datasets are bundled, matching the two code sets above:
- vgsales.csv (classic, simpler) - https://www.kaggle.com/datasets/gregorut/videogamesales
- paper_replication/video_game_sales_and_ratings.csv (matches the parent paper) -
  https://www.kaggle.com/datasets/thedevastator/video-game-sales-and-rating
