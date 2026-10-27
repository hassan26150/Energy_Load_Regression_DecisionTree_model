
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


# 1. Load the existing data split
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

split = np.load(OUTPUT_DIR / "data_split.npz")

X_train = split["X_train"].astype(float)
y_train = split["y_train"].astype(float)

X_val = split["X_val"].astype(float)
y_val = split["y_val"].astype(float)


# 2. Create one decision tree
tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)


# 3. Train using training data only
tree_model.fit(X_train, y_train)


# 4. Make predictions
train_predictions = tree_model.predict(X_train)
val_predictions = tree_model.predict(X_val)


# 5. Calculate MAE
train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

val_mae = mean_absolute_error(
    y_val,
    val_predictions
)


# 6. Compare with the current best model
current_best_mae = 1.6846

print("========== DECISION TREE RESULTS ==========")
print("Maximum tree depth:", tree_model.max_depth)

print("Training MAE:", round(float(train_mae), 4))
print("Validation MAE:", round(float(val_mae), 4))

print("\n========== COMPARISON ==========")
print("Polynomial degree-3 Ridge MAE:", current_best_mae)
print("Decision tree MAE:", round(float(val_mae), 4))

if val_mae < current_best_mae:
    print("Decision tree has the lower validation MAE.")
else:
    print("Polynomial degree-3 Ridge remains the better candidate.")


# 7. Check the training-validation gap
gap = val_mae - train_mae

print("\nValidation MAE minus training MAE:", round(float(gap), 4))

if gap > 1.0:
    print("The gap is noticeable; inspect possible overfitting.")
else:
    print("The training-validation gap is not large by this check.")


print("\nPhase 7 complete.")
print("Test data was not used.")
