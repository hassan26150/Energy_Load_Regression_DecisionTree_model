
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error


# 1. Locate the project folder
PROJECT_DIR = Path(__file__).resolve().parent


# 2. Load the saved data split
split = np.load(
    PROJECT_DIR / "outputs" / "data_split.npz"
)

X_train = split["X_train"]
y_train = split["y_train"]

X_val = split["X_val"]
y_val = split["y_val"]


# 3. Create a simple mean baseline
baseline = DummyRegressor(strategy="mean")


# 4. Train the baseline using training data only
baseline.fit(X_train, y_train)


# 5. Predict heating load for validation buildings
y_pred = baseline.predict(X_val)


# 6. Calculate Mean Absolute Error (MAE)
baseline_mae = mean_absolute_error(y_val, y_pred)


# 7. Display results
print("========== BASELINE MODEL ==========")

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

print(
    "Training target mean:",
    round(float(np.mean(y_train)), 3)
)

print(
    "First 5 actual heating loads:",
    np.round(y_val[:5], 3)
)

print(
    "First 5 predicted heating loads:",
    np.round(y_pred[:5], 3)
)

print(
    "Baseline validation MAE:",
    round(float(baseline_mae), 4)
)

print("\nPhase 3 complete.")
print("Test data was not used.")
