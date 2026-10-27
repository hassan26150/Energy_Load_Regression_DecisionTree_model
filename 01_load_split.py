
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Set up project folders
PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Download the official UCI dataset if it is not already saved
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/242/data.csv"
)

RAW_DATA_PATH = DATA_DIR / "uci_energy_efficiency_raw.csv"
CLEAN_DATA_PATH = DATA_DIR / "energy_efficiency_clean.csv"
SPLIT_PATH = OUTPUT_DIR / "data_split.npz"

if not RAW_DATA_PATH.exists():
    print("Downloading the UCI Energy Efficiency dataset...")

    try:
        urlretrieve(DATA_URL, RAW_DATA_PATH)
    except Exception as error:
        raise SystemExit(
            "\nDataset download failed. Check your internet connection "
            "and run this script again.\n"
            f"Details: {error}"
        )

    print("Download complete.")


# 3. Load the raw data
df = pd.read_csv(RAW_DATA_PATH)

print("\n========== RAW DATA ==========")
print("Raw dataset shape:", df.shape)


# 4. Confirm the expected dataset columns
source_columns = [
    "X1", "X2", "X3", "X4",
    "X5", "X6", "X7", "X8",
    "Y1", "Y2"
]

if list(df.columns) != source_columns:
    raise ValueError(
        "The downloaded dataset has unexpected columns.\n"
        f"Expected: {source_columns}\n"
        f"Received: {list(df.columns)}"
    )


# 5. Give columns readable names
column_names = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
    "heating_load",
    "cooling_load"
]

df.columns = column_names


# 6. Apply our one predefined cleaning pass
# Convert columns to numeric values.
# Replace infinite values with missing values.
# Drop rows that contain missing or invalid values.
# Do not remove rows just because they look unusual.

rows_before_cleaning = len(df)

for column in column_names:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.replace([np.inf, -np.inf], np.nan)

missing_cells_before_cleaning = int(
    df.isna().sum().sum()
)

df = df.dropna(
    subset=column_names
).reset_index(drop=True)

rows_removed = rows_before_cleaning - len(df)

print("\n========== CLEANING SUMMARY ==========")
print("Rows before cleaning:", rows_before_cleaning)
print("Missing/invalid cells before cleaning:",
      missing_cells_before_cleaning)
print("Rows removed by the fixed cleaning rule:", rows_removed)
print("Rows after cleaning:", len(df))
print("Remaining missing values:", int(df.isna().sum().sum()))

# Report duplicates but do not automatically remove them.
print("Exact duplicate rows retained:", int(df.duplicated().sum()))

df.to_csv(CLEAN_DATA_PATH, index=False)

print("\nClean dataset saved to:")
print(CLEAN_DATA_PATH)


# 7. Select eight features and one target
feature_columns = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution"
]

target_column = "heating_load"

X = df[feature_columns].to_numpy(dtype=np.float64)
y = df[target_column].to_numpy(dtype=np.float64)


# 8. Create the test split first
# 80% goes to the development set; 20% stays held out for testing.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 9. Split the development data into training and validation
# 25% of the remaining 80% becomes validation.
# Overall proportions are approximately 60% train,
# 20% validation, and 20% test.

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42
)


# 10. Save all splits for the later phases
# Later scripts will load these same arrays rather than
# creating new random splits.

np.savez_compressed(
    SPLIT_PATH,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    feature_names=np.array(feature_columns),
    target_name=np.array(target_column)
)


# 11. Report the result
print("\n========== FINAL DATA SPLIT ==========")
print("Features:", len(feature_columns))
print("Target:", target_column)
print("Training rows:", len(y_train))
print("Validation rows:", len(y_val))
print("Test rows:", len(y_test))

print("\nExpected shapes:")
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)

print("\nTraining target summary:")
print(pd.Series(y_train).describe().round(3).to_string())

print("\nSplit file saved to:")
print(SPLIT_PATH)

print("\nPhase 1 complete.")
print("The test data has not been used for model evaluation.")
