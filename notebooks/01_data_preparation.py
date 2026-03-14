import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('../data/raw_chembl-AChE.csv', sep=';')

print(f"Raw data shape: {df.shape}")
print(f"\nColumn names:\n{list(df.columns)}")

# Keeping the essential columns
cols_needed = [
    'Molecule ChEMBL ID',
    'Smiles',
    'Standard Value',
    'Standard Units',
    'pChEMBL Value',
    'Data Validity Comment'
]

df = df[cols_needed]

# Rename columns
df.columns = ['chembl_id', 'smiles', 'IC50_nM', 'units', 'pIC50', 'validity']

print(f"\nAfter column selection: {df.shape}")
print(df.head())

# Drop rows with missing SMILES or pIC50
df = df.dropna(subset=['smiles', 'pIC50'])
print(f"\nAfter dropping missing SMILES/pIC50: {df.shape}")

# Drop rows flagged with data validity issues
df = df[df['validity'].isna()]
print(f"After removing flagged data: {df.shape}")

# Drop the validity column
df = df.drop(columns=['validity', 'units'])

# Handle duplicate compounds - same compound tested multiple times
# Keep the median pIC50 per compound
df = df.groupby('chembl_id').agg({
    'smiles': 'first',
    'IC50_nM': 'median',
    'pIC50': 'median'
}).reset_index()

print(f"After removing duplicates: {df.shape}")
print(f"\npIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
print(f"Mean pIC50: {df['pIC50'].mean():.2f}")

# Validate SMILES using RDKit
def validate_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol is not None

df['valid'] = df['smiles'].apply(validate_smiles)
print(f"\nValid SMILES: {df['valid'].sum()}")
print(f"Invalid SMILES: {(~df['valid']).sum()}")

# Keep only valid SMILES
df = df[df['valid']].drop(columns=['valid'])
print(f"Final dataset shape: {df.shape}")

# Save clean dataset
df.to_csv('../data/AChE-cleaned.csv', index=False)
print(f"\nSaved: data/AChE-cleaned.csv")

# EDA - pIC50 distribution
plt.figure(figsize=(8, 5))
plt.hist(df['pIC50'], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
plt.axvline(df['pIC50'].mean(), color='red', linestyle='--',
            label=f"Mean = {df['pIC50'].mean():.2f}")
plt.xlabel('pIC50')
plt.ylabel('Count')
plt.title('Distribution of pIC50 Values - AChE Inhibitors')
plt.legend()
plt.tight_layout()
plt.savefig('../results/figures/phase1_pIC50_distribution.png', dpi=300)
plt.show()
print("SAVED: results/figures/phase1_pIC50_distribution.png")