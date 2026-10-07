import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import subprocess
from sklearn.metrics import r2_score, mean_absolute_error
df = pd.read_csv("data/train-test.csv", parse_dates=["date"])

# Exploration and cleaning
print(df.head(3))
df.info()
print(df.shape)

# Checking the distribution of Market_Index by month
df["month"] = df["date"].dt.month 
# fig, ax = plt.subplots(figsize=(10, 5))
# df.boxplot(column="market_index", by="month", ax=ax)
# plt.title("Market Index Distribution by Month Number")
# plt.suptitle("")
# plt.xlabel("Month (1-12)")
# plt.ylabel("Market Index")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.show()

# Checking the distribution of quote by month
# fig, ax = plt.subplots(figsize=(10, 5))
# df.boxplot(column="quote_signal", by="month", ax=ax)
# plt.title("Quote Distribution by Month Number")
# plt.suptitle("")
# plt.xlabel("Month (1-12)")
# plt.ylabel("Quote_signal")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.show()

# Strong seasonal pattern observed, with dense clusters of points (like the low-end drops in months 4, 5, 7, and 10, or the heavy upper tails    

# Feature Engineering
df["weight"] = df["weight"].abs()
df["weight_missing"] = df["weight"].isna().astype(int)
df["mi_missing"] = df["market_index"].isna().astype(int)
df["dayofweek"] = df["date"].dt.dayofweek
df["equipment"] = df["equipment"].astype("category")
df["ldist"] = np.log(df["distance"])
df["lrpm"] = np.log(df["posted_rate"] / df["distance"]) 
print(df.columns)
print(df.head(3))

# As the missing values were at random and were lesser than 1%, relied on LightGBM to handle it natively
# Spltting data with fixed dates to mimic the validation and used lrpm instead of posted cost to avoid conversion in dollars
train = df[df["date"] < "2025-09-01"]
test = df[df["date"] >= "2025-09-01"]
drop_cols = ["load_id", "date", "pickup", "delivery","posted_rate", "lrpm"]

X_train, y_train = train.drop(columns=drop_cols), train["lrpm"]
X_test, y_test = test.drop(columns=drop_cols), test["lrpm"]
print("train:", X_train.shape, " test:", X_test.shape)
print(list(X_train.columns))

# Train LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500, 
    learning_rate=0.03,
    objective="huber", 
    alpha=0.3, 
    verbose=-1
    )

model.fit(X_train, y_train)

# Evaluation in dollars
# Convert log rate per mile predictions back to total dollar price
pred_dollars = np.exp(model.predict(X_test)) * test["distance"]
actual_dollars = test["posted_rate"]

mae_dollars = mean_absolute_error(actual_dollars, pred_dollars)
r2_dollars = r2_score(actual_dollars, pred_dollars)

# Model Performance
print(f"MAE (Dollars): ${mae_dollars:.2f}")
print(f"R² (Dollars): {r2_dollars:.4f}")
print(f"R² (Log RPM): {r2_score(y_test, model.predict(X_test)):.4f}")

# Processing Validation Data
val_df = pd.read_csv("data/validation.csv", parse_dates=["date"])

# Applying the exact same feature engineering used on the training set
val_df["month"] = val_df["date"].dt.month
val_df["weight"] = val_df["weight"].abs()
val_df["weight_missing"] = val_df["weight"].isna().astype(int)
val_df["mi_missing"] = val_df["market_index"].isna().astype(int)
val_df["dayofweek"] = val_df["date"].dt.dayofweek
val_df["equipment"] = val_df["equipment"].astype("category")
val_df["ldist"] = np.log(val_df["distance"])

X_val = val_df[X_train.columns]
val_pred_log = model.predict(X_val)
val_pred_dollars = np.exp(val_pred_log) * val_df["distance"]

# Attaching predictions to validation dataframe mapped by load_id
val_df["predicted_rate"] = val_pred_dollars
template = pd.read_csv("data/validation-predictions-template.csv")

# Merge safely on load_id to ensure exact row alignment
final_submission = template[["load_id"]].merge(
    val_df[["load_id", "predicted_rate"]], 
    on="load_id", 
    how="left"
)
# Saving to the required output file
final_submission.to_csv("validation_predictions.csv", index=False)
print("Validation predictions successfully generated and saved to validation_predictions.csv!")
print(final_submission.head())

# December Predictions for the Scorer
dec_path = "data/december-chart-inputs.csv"
original_df = pd.read_csv(dec_path)
if "predicted_rate" in original_df.columns:
    original_df = original_df.drop(columns=["predicted_rate"])

dec_df = pd.read_csv(dec_path, parse_dates=["date"])

dec_df["month"] = dec_df["date"].dt.month
dec_df["weight"] = dec_df["weight"].abs()
dec_df["weight_missing"] = dec_df["weight"].isna().astype(int)
dec_df["dayofweek"] = dec_df["date"].dt.dayofweek
dec_df["equipment"] = dec_df["equipment"].astype("category")
dec_df["ldist"] = np.log(dec_df["distance"])

# Filling missing columns that were present in training but absent in december-chart-inputs.csv
dec_df["mi_missing"] = 1
dec_df["market_index"] = train["market_index"].median()
dec_df["quote_signal"] = train["quote_signal"].median()

# Fill latitude/longitude with 0.0 (or model training means/medians) since they are missing from this file
for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
    dec_df[col] = 0.0

# Align columns precisely with training order
X_dec = dec_df[X_train.columns]
dec_pred_dollars = np.exp(model.predict(X_dec)) * dec_df["distance"]

final_dec_df = original_df.copy()
final_dec_df["predicted_rate"] = dec_pred_dollars
final_dec_df.to_csv(dec_path, index=False)
print(f"Successfully updated predictions in {dec_path}")

# Official Scorer
print("\nRunning official scorer...")
subprocess.run([
    "python", "score.py", 
    "--predictions", "validation_predictions.csv", 
    "--december-predictions", "data/december-chart-inputs.csv"
], check=True)
print("Scorer execution completed successfully! Check the 'scorer_results/' folder for your candidate_december.png chart.")