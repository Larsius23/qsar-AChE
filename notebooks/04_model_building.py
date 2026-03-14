import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load processed data
X_train = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/X_train.npy')
X_test = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/X_test.npy')
y_train = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/y_train.npy')
y_test = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/y_test.npy')

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# Define models
models = {
    'MLR': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=10, epsilon=0.1)
}

# Train and evaluate each model
print(f"\n{'Model':<20} {'R²_train':<12} {'R²_test':<12} {'RMSE_test':<12}")
print("-" * 56)

results = {}
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    results[name] = {
        'model': model,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'y_pred_test': y_pred_test
    }

    print(f"{name:<20} {r2_train:<12.3f} {r2_test:<12.3f} {rmse_test:<12.3f}")

import joblib
import os

os.makedirs('/Users/larsius/Desktop/codex/qsar-AChE/results/models', exist_ok=True)

for name, res in results.items():
    safe_name = name.replace(' ', '_')
    joblib.dump(res['model'], f'/Users/larsius/Desktop/codex/qsar-AChE/results/models/{safe_name}.pkl')

print("Models saved to results/models/")