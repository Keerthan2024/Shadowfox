
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=== Hyperparameter Tuning - Gradient Boosting Regressor ===")

# Load cleaned data
try:
    data = pd.read_csv('BostonHousing.csv', header=0)
    data.columns = data.iloc[0]
    data = data.drop(data.index[0]).reset_index(drop=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    print("✓ Data loaded and cleaned")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Prepare feature and target
X = data.drop(columns=['MEDV'])
y = data['MEDV']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Setup GridSearchCV
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit grid search
print("Starting grid search... This may take several minutes.")
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters found: {grid_search.best_params_}")

# Train best model
best_gbr = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_gbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Evaluation on Test Set:")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Save the best model
joblib.dump(best_gbr, 'best_gradient_boosting_model_tuned.pkl')
print("Best Gradient Boosting model saved as 'best_gradient_boosting_model_tuned.pkl'")
