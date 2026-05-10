# 🏠 California Housing Price Prediction

A machine learning project that predicts median house values in California using the California Housing dataset. The notebook walks through the full ML pipeline — from exploratory data analysis to hyperparameter-tuned model deployment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Models & Results](#models--results)
- [Output](#output)
- [Usage](#usage)

---

## Overview

This project builds a regression model to predict **median house values** for California districts. It covers:

- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering
- Data preprocessing and scaling
- Training and comparing multiple regression models
- Cross-validation and hyperparameter tuning with GridSearchCV
- Saving the best model for inference

---

## Dataset

**File:** `housing.csv` (California Housing Dataset)

| Feature | Description |
|---|---|
| `longitude` | Longitude of the district |
| `latitude` | Latitude of the district |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | District population |
| `households` | Number of households |
| `median_income` | Median income of residents |
| `ocean_proximity` | Proximity to the ocean (categorical) |
| `median_house_value` | **Target variable** — median house value ($) |

---

## Project Structure

```
california-housing-prediction/
│
├── housing.csv                  # Input dataset
├── housing_price_prediction.ipynb  # Main Colab notebook
├── house_price_model.pkl        # Saved best model (generated after training)
└── README.md
```

---

## Requirements

Install the following Python libraries before running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

> **Note:** The notebook is designed for **Google Colab**, which has most of these pre-installed. The `google.colab` library is used for file upload and is only available in the Colab environment.

---

## Getting Started

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run the first cell — it will prompt you to **upload `housing.csv`**.
3. Run all subsequent cells in order.

---

## Pipeline Walkthrough

### 1. Exploratory Data Analysis (EDA)
- Distribution plot of `median_house_value`
- Scatter plot of `median_income` vs `median_house_value`
- Correlation heatmap of all numeric features
- Geographic scatter plot (latitude vs longitude, sized by population)

### 2. Feature Engineering
Three new features are derived to improve model performance:

| New Feature | Formula |
|---|---|
| `rooms_per_household` | `total_rooms / households` |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` |
| `population_per_household` | `population / households` |

### 3. Preprocessing
- **Missing values:** `total_bedrooms` nulls filled with the column median
- **Encoding:** `ocean_proximity` one-hot encoded using `pd.get_dummies`
- **Scaling:** Features standardized with `StandardScaler` (fit on train, transform on test)

### 4. Train/Test Split
```
Test size : 20%
Random state : 42
```

### 5. Model Training & Comparison

Three models are trained and evaluated:

| Model | Metric |
|---|---|
| Linear Regression | R² score |
| Decision Tree Regressor | R² score |
| Random Forest Regressor | R² score |

### 6. Cross-Validation
5-fold cross-validation is run on Linear Regression to assess generalization.

### 7. Hyperparameter Tuning
GridSearchCV is used to tune the Random Forest:

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20]
}
```

Best parameters and best CV score are printed after fitting.

### 8. Final Evaluation
The best model from GridSearchCV is evaluated on the test set:
- **R² Score**
- **Mean Squared Error (MSE)**

---

## Models & Results

| Model | R² Score (approx.) |
|---|---|
| Linear Regression | ~0.63 |
| Decision Tree | ~0.63 |
| Random Forest (tuned) | ~0.82 |

> Actual scores may vary slightly depending on the dataset version used.

---

## Output

After training, the best model is saved using `joblib`:

```python
joblib.dump(best_model, "house_price_model.pkl")
```

A sample prediction is also demonstrated:

```python
sample = X_test[:1]
prediction = best_model.predict(sample)
print("Predicted Price:", prediction)
```

---

## Usage

To load and use the saved model in a new script:

```python
import joblib
import numpy as np

model = joblib.load("house_price_model.pkl")

# Pass a preprocessed feature array (same shape as training data)
prediction = model.predict(sample_features)
print("Predicted House Price: $", prediction[0])
```

> Make sure to apply the same `StandardScaler` transformations before passing features to the model.

---

## Notes

- The notebook contains some repeated preprocessing cells (e.g., `fillna` and feature engineering appear multiple times). These are harmless but can be cleaned up for production use.
- The `google.colab` file upload step is Colab-specific. For local use, replace it with `pd.read_csv("housing.csv")`.

---

## License

This project is for educational purposes. The California Housing dataset is publicly available and commonly used for ML benchmarking.
