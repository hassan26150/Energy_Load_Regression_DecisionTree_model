
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


# 1. Load the saved data split
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

split = np.load(OUTPUT_DIR / "data_split.npz")

X_train = split["X_train"].astype(float)
y_train = split["y_train"].astype(float)

X_test = split["X_test"].astype(float)
y_test = split["y_test"].astype(float)


# 2. Re-create the selected baseline
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)


# 3. Re-create the selected decision tree
tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train, y_train)


# 4. Evaluate both models on the test set
# This is our final held-out evaluation.
baseline_test_predictions = baseline.predict(X_test)
tree_test_predictions = tree_model.predict(X_test)

baseline_test_mae = mean_absolute_error(
    y_test,
    baseline_test_predictions
)

tree_test_mae = mean_absolute_error(
    y_test,
    tree_test_predictions
)


# 5. Calculate improvement over the baseline
improvement = (
    (baseline_test_mae - tree_test_mae)
    / baseline_test_mae
) * 100


# 6. Report final results
print("========== FINAL TEST EVALUATION ==========")

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

print("\nBaseline model")
print(
    "Test MAE:",
    round(float(baseline_test_mae), 4)
)

print("\nSelected decision tree")
print("Maximum depth:", tree_model.max_depth)
print(
    "Test MAE:",
    round(float(tree_test_mae), 4)
)

print("\nImprovement over baseline:")
print(f"{improvement:.2f}%")

print("\n========== FINAL STATUS ==========")
print("Model selection was completed using validation data.")
print("The test set was not used for training or tuning.")
print("Phase 8 complete: final test evaluation performed.")
