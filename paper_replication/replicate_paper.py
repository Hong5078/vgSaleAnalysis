"""
Replication of:
Gray, D., Shalan, A. M., & Kadlec, C. (2024). "Sales Predictions for Video Games
using Predictive Analytics of Market Data." 2024 IEEE 16th International
Conference on Computational Intelligence and Communication Networks (CICN).
DOI: 10.1109/CICN63059.2024.10847429

The original paper built its models in Orange Data Mining, a no-code visual
workflow tool. This script reproduces the same methodology (feature set,
preprocessing, models, and evaluation metrics) as plain, runnable Python
using pandas / scikit-learn.

Dataset: Kaggle "Video Game Sales and Ratings"
https://www.kaggle.com/datasets/thedevastator/video-game-sales-and-rating
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 5078
CSV_PATH = "video_game_sales_and_ratings.csv"

# ---------------------------------------------------------------------------
# Section III-A (Data Collection) — load the raw dataset.
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# The paper's feature list uses "index" as a feature, i.e. the row number of
# the dataset after loading. We reproduce that literally: it is simply the
# positional index of each row, carrying no real predictive information.
df = df.reset_index(drop=False)  # creates an "index" column: 0, 1, 2, ...

FEATURES = [
    "index",
    "Year_of_Release",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Critic_Score",
    "Critic_Count",
    "User_Count",
]
TARGET = "Global_Sales"

# ---------------------------------------------------------------------------
# Section III-B (Data Preprocessing) — the paper describes a cleaning step
# to handle missing critic/user data before modeling. We drop any row with a
# missing value in a feature or the target column, matching that description.
# ---------------------------------------------------------------------------
model_df = df[FEATURES + [TARGET]].dropna().reset_index(drop=True)

print(f"Rows before cleaning: {len(df)}")
print(f"Rows after dropping missing values in features/target: {len(model_df)}")
print()

# ---------------------------------------------------------------------------
# IMPORTANT CAVEAT (not addressed in the paper):
# Global_Sales is arithmetically defined as
#     Global_Sales = NA_Sales + EU_Sales + JP_Sales + Other_Sales
# in this dataset. Since all four regional sales columns are included as
# input features, a Linear Regression model can recover Global_Sales almost
# exactly (coefficients ~1.0 on each regional feature, ~0 on the rest) simply
# by re-deriving this near-identity sum. A very high R^2 here is therefore
# expected from this arithmetic relationship in the data, and should NOT be
# interpreted as evidence that the model is learning deeper market drivers
# of video game sales.
# ---------------------------------------------------------------------------

X = model_df[FEATURES]
y = model_df[TARGET]

# ---------------------------------------------------------------------------
# Section III-C (Model Building) — held-out train/test split.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

tree_reg = DecisionTreeRegressor(random_state=RANDOM_STATE)
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE: exclude rows where y_true == 0 to avoid division by zero.
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


metrics_lin = compute_metrics(y_test, y_pred_lin)
metrics_tree = compute_metrics(y_test, y_pred_tree)

# ---------------------------------------------------------------------------
# Section IV, Table II — model performance comparison (MSE, RMSE, MAE, MAPE, R^2).
# ---------------------------------------------------------------------------
print("=" * 70)
print("TABLE II - Model Performance Comparison")
print("=" * 70)
header = f"{'Model':<20}{'MSE':>10}{'RMSE':>10}{'MAE':>10}{'MAPE (%)':>12}{'R^2':>10}"
print(header)
print("-" * len(header))
for name, m in [("Linear Regression", metrics_lin), ("Decision Tree", metrics_tree)]:
    print(
        f"{name:<20}{m['MSE']:>10.4f}{m['RMSE']:>10.4f}{m['MAE']:>10.4f}"
        f"{m['MAPE']:>12.4f}{m['R2']:>10.4f}"
    )
print()

# ---------------------------------------------------------------------------
# Section IV, Table III — Linear Regression coefficients per feature.
# ---------------------------------------------------------------------------
print("=" * 70)
print("TABLE III - Linear Regression Coefficients")
print("=" * 70)
print(f"{'Feature':<20}{'Coefficient':>15}")
print("-" * 35)
print(f"{'(Intercept)':<20}{lin_reg.intercept_:>15.6f}")
for feature, coef in zip(FEATURES, lin_reg.coef_):
    print(f"{feature:<20}{coef:>15.6f}")
print()

print("Reminder: NA_Sales + EU_Sales + JP_Sales + Other_Sales already equals")
print("Global_Sales by construction in this dataset, so the near-1.0")
print("coefficients on those four features (and the resulting high R^2)")
print("reflect that arithmetic identity, not a learned market relationship.")
