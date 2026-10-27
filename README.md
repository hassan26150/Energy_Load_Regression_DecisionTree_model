# 🏠 Building Energy Load Predictor

A machine learning project that predicts building heating load using building design features and a Decision Tree Regressor. The project includes data analysis, baseline comparison, model evaluation, and an interactive Streamlit web application.

## Features

- Predicts heating load from eight building design features.
- Interactive Streamlit interface.
- Decision Tree Regressor selected using validation performance.
- Comparison against a mean-prediction baseline.
- Separate training, validation, and test datasets.
- Displays the user's input features alongside the prediction.

## Model Performance

| Metric | Result |
|---|---:|
| Validation MAE | 0.6904 |
| Test MAE | 0.7453 |
| Baseline Test MAE | 9.2783 |
| Improvement over baseline | 91.97% |

MAE stands for Mean Absolute Error. Lower values indicate smaller average absolute prediction errors. The baseline predicts the mean heating load rather than learning relationships from building features.

The test set was used for final evaluation after model selection.

## Dataset

This project uses the UCI Machine Learning Repository's Energy Efficiency dataset.

- 768 building configurations
- Eight building design features
- Heating load (`Y1`) as the prediction target

The model uses the following features:

1. Relative compactness
2. Surface area
3. Wall area
4. Roof area
5. Overall height
6. Orientation
7. Glazing area
8. Glazing area distribution

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## Project Structure

```text
Energy_Load_Regression/
├── app.py
├── data/
│   ├── energy_efficiency_clean.csv
│   └── uci_energy_efficiency_raw.csv
├── outputs/
│   ├── data_split.npz
│   ├── scratch_cost_history.png
│   ├── scratch_model.npz
│   └── eda/
├── 01_load_split.py
├── 02_eda.py
├── 03_baseline.py
├── 04_scratch_linear_regression.py
├── 05_polynomial_overfit.py
├── 06_sklearn_comparison.py
├── 07_tree_comparison.py
├── 08_final_test_evaluation.py
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/hassan26150/Energy_Load_Regression_DecisionTree_model.git
cd Energy_Load_Regression_DecisionTree_model
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Start the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL shown in your terminal, usually `http://localhost:8501`.

## How It Works

1. Building design features are entered through the Streamlit interface.
2. The application loads the saved training split.
3. The selected Decision Tree Regressor is fitted using the training data.
4. The model predicts heating load for the submitted building features.
5. The prediction and input values are displayed in the interface.

The application uses the existing saved data split and does not retrain or compare models on the validation or test sets during prediction.

## Limitations

This is an educational machine learning project using a fixed dataset.

## Author

Muhammad Hassan Arshad

## License

No license has been added yet. Unless a license is selected, others do not automatically receive permission to reuse, modify, or distribute this project.