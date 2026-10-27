
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error


# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load the saved data split
split = np.load(OUTPUT_DIR / "data_split.npz")

X_train = split["X_train"].astype(float)
y_train = split["y_train"].astype(float)

X_val = split["X_val"].astype(float)
y_val = split["y_val"].astype(float)


# 3. Feature scaling
# Learn the mean and standard deviation from training data only.
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

# Prevent division by zero if a feature has no variation.
X_std[X_std == 0] = 1.0

X_train_scaled = (X_train - X_mean) / X_std
X_val_scaled = (X_val - X_mean) / X_std


# 4. Prediction function
def predict(X, w, b):
    return np.dot(X, w) + b


# 5. Cost function
def compute_cost(X, y, w, b):
    m = len(y)

    predictions = predict(X, w, b)
    errors = predictions - y

    cost = np.sum(errors ** 2) / (2 * m)

    return cost


# 6. Calculate gradients
def compute_gradients(X, y, w, b):
    m = len(y)

    predictions = predict(X, w, b)
    errors = predictions - y

    dj_dw = np.dot(X.T, errors) / m
    dj_db = np.sum(errors) / m

    return dj_dw, dj_db


# 7. Gradient descent
def gradient_descent(X, y, w, b, learning_rate, iterations):
    cost_history = []

    for i in range(iterations):
        dj_dw, dj_db = compute_gradients(X, y, w, b)

        # Update all weights and the bias.
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        # Record cost after the update.
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if i % 500 == 0:
            print(f"Iteration {i}: cost = {cost:.6f}")

    return w, b, cost_history


# 8. Initialize parameters
number_of_features = X_train_scaled.shape[1]

w = np.zeros(number_of_features)
b = 0.0

learning_rate = 0.03
iterations = 5000


# 9. Train the model
print("========== TRAINING FROM-SCRATCH MODEL ==========")

w, b, cost_history = gradient_descent(
    X_train_scaled,
    y_train,
    w,
    b,
    learning_rate,
    iterations
)


# 10. Evaluate on validation data
y_val_pred = predict(X_val_scaled, w, b)

scratch_val_mae = mean_absolute_error(
    y_val,
    y_val_pred
)

print("\n========== VALIDATION RESULTS ==========")
print("Number of features:", number_of_features)
print("Final training cost:", round(cost_history[-1], 6))
print("Validation MAE:", round(float(scratch_val_mae), 4))
print("Baseline validation MAE: 9.5648")

improvement = (
    (9.5648 - scratch_val_mae) / 9.5648
) * 100

print("Improvement over baseline:", round(float(improvement), 2), "%")

if scratch_val_mae <= 9.5648 * 0.75:
    print("Success criterion passed: MAE improved by at least 25%.")
else:
    print("Success criterion not yet met: improvement is below 25%.")


# 11. Plot the cost history
plt.figure(figsize=(9, 5))

plt.plot(cost_history)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent: Cost vs Iteration")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "scratch_cost_history.png",
    dpi=150
)

plt.close()


# 12. Save parameters for later comparison and reuse.
np.savez(
    OUTPUT_DIR / "scratch_model.npz",
    w=w,
    b=b,
    X_mean=X_mean,
    X_std=X_std,
    cost_history=np.array(cost_history)
)

print("\nSaved cost graph to:", OUTPUT_DIR / "scratch_cost_history.png")
print("Saved model parameters to:", OUTPUT_DIR / "scratch_model.npz")
print("\nPhase 4 training and validation evaluation complete.")
print("The test set was not used.")
