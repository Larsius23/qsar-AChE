import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load descriptor data
df = pd.read_csv('/Users/larsius/Desktop/codex/qsar-AChE/data/AChE_descriptors.csv')

# Separate metadata, X and y
meta_cols = ['chembl_id', 'smiles', 'IC50_nM']
y = df['pIC50'].values
X = df.drop(columns=meta_cols + ['pIC50'])

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"NaN values in X: {X.isnull().sum().sum()}")

# Step 1: Fill NaN values with column median
X = X.fillna(X.median())
print(f"Step 1 - After filling NaN: {X.isnull().sum().sum()} NaN values remaining")

# Step 2: Remove low-variance descriptors
sel = VarianceThreshold(threshold=0.01)
X_var = sel.fit_transform(X)
X = pd.DataFrame(X_var, columns=X.columns[sel.get_support()])
print(f"Step 2 - After variance filter: {X.shape}")

# Step 3: Remove highly correlated descriptors (r > 0.95)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X = X.drop(columns=to_drop)
print(f"Step 3 - After correlation filter: {X.shape}")
print(f"         Dropped {len(to_drop)} correlated descriptors")

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(f"Step 4 - Scaling done: mean≈0, std≈1")

# Step 5: Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

print(f"Step 5 - Train/test split done")
print(f"         Training set: {X_train.shape}")
print(f"         Test set:     {X_test.shape}")

# Save processed data
import os
os.makedirs('../data/processed', exist_ok=True)

np.save('../data/processed/X_train.npy', X_train.values)
np.save('../data/processed/X_test.npy', X_test.values)
np.save('../data/processed/y_train.npy', y_train)
np.save('../data/processed/y_test.npy', y_test)

# Save descriptor names for later
pd.Series(X.columns.tolist()).to_csv('../data/processed/descriptor_names.csv', index=False)

print(f"\nSaved all processed arrays to data/processed/")