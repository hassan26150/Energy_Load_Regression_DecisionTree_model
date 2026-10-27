
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# 1. Load the existing split
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

split = np.load(OUTPUT_DIR / "data_split.npz")

X_train = split["X_train"].astype(float)
y_train = split["y_train"].astype(float)

X_val = split["X_val"].astype(float)
y_val = split["y_val"].astype(float)


# 2. Load the saved from-scratch model
saved = np.load(OUTPUT_DIR / "scratch_model.npz")

w = saved["w"]
b = float(saved["b"])

X_mean = saved["X_mean"]
X_std = saved["X_std"]


# 3. Predict with the from-scratch model
X_val_scaled = (X_val - X_mean) / X_std

scratch_predictions = np.dot(X_val_scaled, w) + b

scratch_mae = mean_absolute_error(
    y_val,
    scratch_predictions
)


# 4. Train scikit-learn LinearRegression
linear_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_val)

linear_mae = mean_absolute_error(
    y_val,
    linear_predictions
)


# 5. Train Ridge regression
ridge_model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

ridge_model.fit(X_train, y_train)

ridge_predictions = ridge_model.predict(X_val)

ridge_mae = mean_absolute_error(
    y_val,
    ridge_predictions
)


# 6. Train degree-3 polynomial Ridge regression
polynomial_model = make_pipeline(
    PolynomialFeatures(
        degree=3,
        include_bias=False
    ),
    StandardScaler(),
    Ridge(alpha=1.0)
)

polynomial_model.fit(X_train, y_train)

polynomial_predictions = polynomial_model.predict(X_val)

polynomial_mae = mean_absolute_error(
    y_val,
    polynomial_predictions
)


# 7. Display a consistent comparison
results = [
    ("From-scratch linear regression", scratch_mae),
    ("Scikit-learn LinearRegression", linear_mae),
    ("Scikit-learn Ridge", ridge_mae),
    ("Degree-3 polynomial Ridge", polynomial_mae),
]

print("========== VALIDATION COMPARISON ==========")

for name, mae in results:
    print(f"{name}: MAE = {mae:.4f}")


# 8. Check the fixed scratch-vs-sklearn criterion
difference_percent = (
    abs(scratch_mae - linear_mae) / linear_mae
) * 100

print("\n========== CONSISTENCY CHECK ==========")
print(
    "Scratch vs scikit-learn MAE difference:",
    f"{difference_percent:.2f}%"
)

if difference_percent <= 5:
    print("PASS: Difference is within the 5% criterion.")
else:
    print("NOT PASSED: Difference exceeds the 5% criterion.")


# 9. Report the best model among those compared
best_name, best_mae = min(
    results,
    key=lambda item: item[1]
)

print("\nBest model in this comparison:", best_name)
print(f"Best validation MAE: {best_mae:.4f}")

print("\nPhase 6 complete.")
print("Test data was not used.")
