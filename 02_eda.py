
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1. Load the cleaned dataset and saved training split
PROJECT_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    PROJECT_DIR / "data" / "energy_efficiency_clean.csv"
)

split = np.load(
    PROJECT_DIR / "outputs" / "data_split.npz"
)

feature_names = split["feature_names"].tolist()
target_name = str(split["target_name"].item())

X_train = split["X_train"]
y_train = split["y_train"]

# EDA uses training data only.
train_df = pd.DataFrame(
    X_train,
    columns=feature_names
)

train_df[target_name] = y_train

output_dir = PROJECT_DIR / "outputs" / "eda"
output_dir.mkdir(parents=True, exist_ok=True)


# 2. Basic summary
print("========== TRAINING DATA SUMMARY ==========")
print("Training rows:", len(train_df))
print("Input features:", len(feature_names))
print("Target:", target_name)

print("\nFeature statistics:")
print(
    train_df[feature_names]
    .describe()
    .round(3)
    .to_string()
)

print("\nHeating-load statistics:")
print(train_df[target_name].describe().round(3).to_string())


# 3. Target distribution
plt.figure(figsize=(8, 5))

plt.hist(
    train_df[target_name],
    bins=25,
    edgecolor="black"
)

plt.xlabel("Heating load")
plt.ylabel("Number of buildings")
plt.title("Distribution of Heating Load")
plt.tight_layout()

plt.savefig(
    output_dir / "01_target_distribution.png",
    dpi=150
)

plt.close()


# 4. Feature distributions
train_df[feature_names].hist(
    bins=20,
    figsize=(12, 9),
    edgecolor="black"
)

plt.suptitle("Training Feature Distributions")
plt.tight_layout()

plt.savefig(
    output_dir / "02_feature_distributions.png",
    dpi=150
)

plt.close()


# 5. Feature-to-target relationships
fig, axes = plt.subplots(
    4,
    2,
    figsize=(12, 14)
)

for index, feature in enumerate(feature_names):
    ax = axes.flat[index]

    ax.scatter(
        train_df[feature],
        train_df[target_name],
        alpha=0.6
    )

    ax.set_xlabel(feature)
    ax.set_ylabel("Heating load")
    ax.set_title(f"{feature} vs heating load")

plt.tight_layout()

plt.savefig(
    output_dir / "03_feature_target_relationships.png",
    dpi=150
)

plt.close()


# 6. Correlation matrix
correlation = train_df.corr(numeric_only=True)

print("\n========== FEATURE CORRELATIONS WITH TARGET ==========")

print(
    correlation[target_name]
    .drop(target_name)
    .sort_values(key=abs, ascending=False)
    .round(3)
    .to_string()
)

plt.figure(figsize=(11, 9))

image = plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(image, label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.index)),
    correlation.index
)

plt.title("Training Data Correlation Matrix")
plt.tight_layout()

plt.savefig(
    output_dir / "04_correlation_matrix.png",
    dpi=150
)

plt.close()


# 7. Save the correlation table
correlation.to_csv(
    output_dir / "correlation_matrix.csv"
)

print("\nEDA files saved in:")
print(output_dir)

print("\nPhase 2 complete.")
print("No rows were removed or modified.")
print("Validation and test data were not used.")
