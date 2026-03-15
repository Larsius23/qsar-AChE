import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
X_train = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/X_train.npy')
X_test  = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/X_test.npy')
y_train = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/y_train.npy')
y_test  = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/y_test.npy')

# Load best model (SVR)
svr = joblib.load('/Users/larsius/Desktop/codex/qsar-AChE/results/models/SVR.pkl')
rf  = joblib.load('/Users/larsius/Desktop/codex/qsar-AChE/results/models/Random_forest.pkl')

# Predictions
y_pred_train = svr.predict(X_train)
y_pred_test  = svr.predict(X_test)

print(f"R²_train: {r2_score(y_train, y_pred_train):.3f}")
print(f"R²_test: {r2_score(y_test, y_pred_test):.3f}")
print("Ready for visualization")

# Plot 1: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('AChE Inhibitor 2D-QSAR Results', fontsize=14, fontweight='bold')

# Train points
axes[0].scatter(y_train, y_pred_train, color='steelblue',
                alpha=0.4, s=10, label='Train')

# Test points
axes[0].scatter(y_test, y_pred_test, color='coral',
                alpha=0.7, s=15, label='Test')

# Ideal liine
lims = [4, 11]
axes[0].plot(lims, lims, 'k--', linewidth=1.5, label='y=x')
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)
axes[0].set_xlabel('Actual pIC50')
axes[0].set_ylabel('Predicted pIC50')
axes[0].set_title(f'SVR: Actual vs Predicted\nR²_train=0.908 R²_test=0.672')
axes[0].legend()

# Plot 2: Residuals
residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, color='steelblue',
                alpha=0.6, s=15)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Predicted pIC50')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals Plot (Test Set)')

plt.tight_layout()
plt.savefig('/Users/larsius/Desktop/codex/qsar-AChE/results/figures/phase6_actual_vs_predicted.png', dpi=300)
plt.show()
print("Saved: results/figures/phase6_actual_vs_predicted.png")

# Plot 2: Feature Importance (Random Forest)
desc_names = pd.read_csv('../data/processed/descriptor_names.csv')['0'].tolist()

importances = rf.feature_importances_
feat_imp = pd.DataFrame({
    'descriptor': desc_names,
    'importance': importances
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(8, 6))
plt.barh(feat_imp['descriptor'][::-1], feat_imp['importance'][::-1], 
         color='steelblue', edgecolor='black', alpha=0.8)
plt.xlabel('Feature Importance')
plt.title('Top 15 Descriptors for AChE Inhibition\n(Random Forest)')
plt.tight_layout()
plt.savefig('/Users/larsius/Desktop/codex/qsar-AChE/results/figures/phase6_feature_importance.png', dpi=300)
plt.show()
print("Saved: results/figures/phase6_feature_importance.png")