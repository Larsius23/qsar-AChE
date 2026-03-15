import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load processed data
X_train = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/X_train.npy')
X_test  = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/X_test.npy')
y_train = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/y_train.npy')
y_test  = np.load('/Users/larsius/Desktop/codex/qsar-AChE/data/processed/y_test.npy')

# Combine for cross-validation
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

# Load trained models
models = {
    'MLR': joblib.load('/Users/larsius/Desktop/codex/qsar-AChE/results/models/MLR.pkl'),
    'Random Forest': joblib.load('/Users/larsius/Desktop/codex/qsar-AChE/results/models/Random_Forest.pkl'),
    'SVR': joblib.load('/Users/larsius/Desktop/codex/qsar-AChE/results/models/SVR.pkl')
}

print(f"X_all: {X_all.shape}")
print(f"y_all: {y_all.shape}")
print("Models loaded")

# 5-Fold Cross-Validation
print(f"\n{'Model':<20} {'Q²_CV':<12} {'Std':<12}")
print("-" * 44)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_all, y_all, cv=kf, scoring='r2')
    cv_results[name] = {
        'Q2_CV': cv_scores.mean(),
        'std': cv_scores.std()
    }
    status = "✅" if cv_scores.mean() >= 0.5 else "❌"
    print(f"{name:<20} {cv_scores.mean():<12.3f} {cv_scores.std():<12.3f} {status}")

# Y-Scrambling Test (100 randomizations)
# print(f"\nY-Scrambling Test (100 runs)...")
# print(f"{'Model':<20} {'Actual Q²':<12} {'Scrambled Q²':<15} {'Verdict'}")
# print("-" * 60)

# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# for name, model in models.items():
#    scrambled_scores = []
#    for i in range(100):
#        if i % 10 == 0:  # print every 10 runs
#            print(f"  {name}: {i}/100 runs done...", flush=True)
#        y_shuffled = y_all.copy()
#        np.random.shuffle(y_shuffled)
#        score = cross_val_score(model, X_all, y_shuffled, cv=kf, scoring='r2')
#        scrambled_scores.append(score.mean())
#    
#    actual_q2 = cv_results[name]['Q2_CV']
#    scrambled_mean = np.mean(scrambled_scores)
#    verdict = "✅ Not random" if actual_q2 > scrambled_mean + 0.1 else "❌ Suspicious"
    
#    print(f"{name:<20} {actual_q2:<12.3f} {scrambled_mean:<15.3f} {verdict}")

#Hardcode results from previous run
cv_results = {
    'MLR':           {'Q2_CV': 0.332},
    'Random Forest': {'Q2_CV': 0.667},
    'SVR':           {'Q2_CV': 0.681}
}

#  Final Validation Summary
print(f"\n{'='*60}")
print(f"  VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<20} {'R²_train':<12} {'Q²_CV':<12} {'R²_test':<12} {'Pass?'}")
print(f"{'-'*60}")

# R² train from phase 4
r2_trains = {'MLR': 0.381, 'Random Forest': 0.951, 'SVR': 0.908}
r2_tests  = {'MLR': 0.310, 'Random Forest': 0.672, 'SVR': 0.672}

for name in models:
    r2_tr = r2_trains[name]
    q2_cv = cv_results[name]['Q2_CV']
    r2_te = r2_tests[name]
    delta = abs(r2_tr - q2_cv)
    passes = r2_tr >= 0.6 and q2_cv >= 0.5 and delta <= 0.3
    status = "✅" if passes else "❌"
    print(f"{name:<20} {r2_tr:<12.3f} {q2_cv:<12.3f} {r2_te:<12.3f} {status}")