## Introduction

This project develops a machine learning model to predict house prices using the California Housing dataset. The dataset contains features such as location (longitude, latitude), housing characteristics (e.g., total rooms, bedrooms, median age), demographic information (population, households), economic factors (median income), and proximity to the ocean. The target variable is the median house value.

The project follows a standard machine learning workflow: data loading, exploration, preprocessing, feature engineering, model training, evaluation, and selection of the best model. The code is implemented in a Jupyter Notebook (`HousePricePrediction.ipynb`).

### Project Goals

- Build and compare multiple regression models for house price prediction.
- Evaluate model performance using metrics like MSE, RMSE, MAE, R² Score, and cross-validation.
- Identify the best-performing model based on cross-validation scores.
- Provide an example prediction and document the entire process.

### Dataset Overview

- Source: California Housing dataset (likely from the "Hands-On Machine Learning with Scikit-Learn" repository or similar, e.g., [housing.csv on GitHub](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv)).
- Rows: 20,640
- Columns: 10 (9 features + 1 target)
- Features: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity` (categorical).
- Target: `median_house_value`
- Key Statistics (from notebook):
  - Median house value ranges from $14,999 to $500,001.
  - Missing values in `total_bedrooms` (207 entries, filled with median).
- Data Distribution: Numerical features show varying scales (e.g., total_rooms up to 39,320), requiring scaling for some models.

## Dependencies

The project relies on the following Python libraries. Install them using `pip` if needed (e.g., `pip install numpy pandas matplotlib seaborn scikit-learn`).

- **numpy (np)**: For numerical computations.
- **pandas (pd)**: For data loading, manipulation, and analysis.
- **matplotlib.pyplot (plt)**: For plotting visualizations.
- **seaborn (sns)**: For enhanced data visualization (e.g., heatmaps).
- **sklearn.preprocessing**: LabelEncoder for categorical encoding, StandardScaler for feature scaling.
- **sklearn.model_selection**: train_test_split for data splitting, cross_val_score for validation.
- **sklearn.linear_model**: LinearRegression, Lasso, Ridge for linear models.
- **sklearn.tree**: DecisionTreeRegressor for tree-based model.
- **sklearn.ensemble**: RandomForestRegressor, GradientBoostingRegressor for ensemble models.
- **sklearn.metrics**: mean_squared_error, mean_absolute_error, r2_score for evaluation.
- **warnings**: To suppress warnings during execution.

Python Version: 3.x (tested with Python 3.12.3 in the notebook).

The project relies on the following Python libraries. Install them using the following command to ensure all dependencies are met:

```bash
pip install -r requirements.txt
```

## Steps Involved in Development

The project development follows these sequential steps, as implemented in the notebook.

### 1. Data Loading

- Load the dataset from a CSV file: `df = pd.read_csv("/content/housing_price.csv")`.
- Verify data: Display first 5 rows (`df.head()`), shape (`(20640, 10)`), info (`df.info()`), and summary statistics (`df.describe()`).

### 2. Data Exploration and Preprocessing

- **Missing Value Handling**: Check for nulls (`df.isnull().sum()`). Fill missing `total_bedrooms` with median: `df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)`.
- **Categorical Encoding**: Use LabelEncoder on `ocean_proximity`: `le = LabelEncoder(); df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])`.
  - Note: This treats the categorical feature as ordinal, which may not be ideal (consider OneHotEncoder for better results).
- **Outlier Detection**: Visualize distributions with boxplots: `plt.figure(figsize=(12,8)); sns.boxplot(data=df);`.
- **Correlation Analysis**: Compute and visualize correlation matrix: `corr_matrix = df.corr(); sns.heatmap(corr_matrix, annot=True);`.
  - Key insights: `median_income` strongly correlates with `median_house_value` (0.69).

### 3. Feature Engineering

- Create new features to capture ratios:
  - `rooms_per_household = total_rooms / households`
  - `bedrooms_per_room = total_bedrooms / total_rooms`
  - `population_per_household = population / households`
- Drop original less-informative features if needed (not done in notebook).
- Final features: Original 9 + 3 engineered.

### 4. Data Splitting and Scaling

- Separate features (X) and target (y): `X = df.drop('median_house_value', axis=1); y = df['median_house_value']`.
- Split into train/test (80/20): `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`.
- Scale features for linear models: `scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)`.

### 5. Model Training

- Define models in a dictionary:
  ```python
  models = {
      'Linear Regression': LinearRegression(),
      'Lasso Regression': Lasso(),
      'Ridge Regression': Ridge(),
      'Decision Tree': DecisionTreeRegressor(),
      'Random Forest': RandomForestRegressor(),
      'Gradient Boosting': GradientBoostingRegressor()
  }
  ```
- Train each model:
  - Scaled data for linear models (Linear, Lasso, Ridge).
  - Unscaled data for tree-based models (Decision Tree, Random Forest, Gradient Boosting).

### 6. Model Evaluation

- Metrics calculated: MSE, RMSE, MAE, R² Score on test set; 5-fold cross-validation R² mean and std.
- Loop through models, train, predict, and evaluate.
- Key Issue in Notebook: The test metrics (MSE, RMSE, MAE, R²) are calculated outside the training loop, so they reflect only the last model (Gradient Boosting). This is a bug—metrics should be computed inside the loop per model.
- Corrected Evaluation Approach (Conceptual Fix):
  - For each model, compute predictions and metrics individually.
  - Use cross_val_score for robustness.
- Results (as printed in notebook, but note the bug for test metrics):
  | Model | MSE (Test) | RMSE (Test) | MAE (Test) | R² Score (Test) | CV R² Mean | CV R² Std |
  |-------------------|------------|-------------|------------|-----------------|------------|-----------|
  | Linear Regression | 2.16e+09 | 46523.19 | 32977.79 | 0.7660 | 0.5995 | 0.0126 |
  | Lasso Regression | 2.16e+09 | 46523.19 | 32977.79 | 0.7660 | 0.5995 | 0.0126 |
  | Ridge Regression | 2.16e+09 | 46523.19 | 32977.79 | 0.7660 | 0.5995 | 0.0126 |
  | Decision Tree | 2.16e+09 | 46523.19 | 32977.79 | 0.7660 | 0.5416 | 0.0138 |
  | Random Forest | 2.16e+09 | 46523.19 | 32977.79 | 0.7660 | 0.7744 | 0.0064 |
  | Gradient Boosting | 2.16e+09 | 46523.19 | 32977.79 | 0.7660 | 0.7455 | 0.0086 |

  - Best Model (based on CV R²): Random Forest (CV R²: 0.7744).
  - Note: Actual test R² for Random Forest should be higher (~0.81 based on typical benchmarks for this dataset); the printed 0.7660 is from Gradient Boosting due to the bug.

### 7. Model Selection and Prediction

- Select best model: Random Forest (highest CV R²).
- Retrain on full training data.
- Example Prediction:
  - Input: Sample house features (e.g., longitude=-122.25, latitude=37.85, etc.).
  - Output: Predicted price ~$319,130.00.

For full code, refer to `HousePricePrediction.ipynb`. If running locally, download the dataset from [GitHub](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv).

**Author**: [Shahbaz Ahmed]  
**Date**: October 18, 2025  
**License**: Apache License
