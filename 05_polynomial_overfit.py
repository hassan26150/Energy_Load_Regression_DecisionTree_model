
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# 1. Load the existing data split
PROJECT_DIR = Path(__file__).resolve().parent
split = np.load(PROJECT_DIR / "outputs" / "data_split.npz")

X_train = split["X_train"].astype(float)
y_train = split["y_train"].astype(float)

X_val = split["X_val"].astype(float)
y_val = split["y_val"].astype(float)


# 2. Record the from-scratch linear model as a reference
scratch_val_mae = 2.1266

print("========== MODEL COMPARISON ==========")
print(f"Scratch linear validation MAE: {scratch_val_mae:.4f}")


# 3. Train ordinary linear regression
linear_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

linear_model.fit(X_train, y_train)

linear_train_pred = linear_model.predict(X_train)
linear_val_pred = linear_model.predict(X_val)

linear_train_mae = mean_absolute_error(
    y_train, linear_train_pred
)

linear_val_mae = mean_absolute_error(
    y_val, linear_val_pred
)

print("\nOrdinary linear regression")
print(f"Training MAE:   {linear_train_mae:.4f}")
print(f"Validation MAE: {linear_val_mae:.4f}")


# 4. Try polynomial feature degrees 2 and 3
results = []

for degree in [2, 3]:
    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    results.append(
        (degree, train_mae, val_mae, model)
    )

    print(f"\nPolynomial degree {degree} with Ridge")
    print(f"Training MAE:   {train_mae:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")


# 5. Compare the validation results
print("\n========== SUMMARY ==========")
print(f"Scratch linear MAE: {scratch_val_mae:.4f}")
print(f"Sklearn linear MAE: {linear_val_mae:.4f}")

for degree, train_mae, val_mae, model in results:
    print(
        f"Degree {degree}: "
        f"training MAE = {train_mae:.4f}, "
        f"validation MAE = {val_mae:.4f}"
    )


# 6. Identify the best validation model among the tested options
candidates = [
    ("Sklearn linear", linear_val_mae, linear_model)
]

for degree, train_mae, val_mae, model in results:
    candidates.append(
        (f"Polynomial degree {degree}", val_mae, model)
    )

best_name, best_mae, best_model = min(
    candidates,
    key=lambda item: item[1]
)

print(f"\nBest model in this phase: {best_name}")
print(f"Best validation MAE: {best_mae:.4f}")

print("\nPhase 5 complete.")
print("The test set was not used.")
