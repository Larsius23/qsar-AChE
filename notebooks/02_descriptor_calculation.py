import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import warnings
warnings.filterwarnings('ignore')

# Load clean dataset
df =pd.read_csv('../data/AChE-cleaned.csv')
print(f"Loaded: {df.shape}")
print(df.head())

# Get all RDKit 2D descriptor names
descriptor_names = [name for name, _ in Descriptors.descList]
print(f"Number of RDKit descriptors: {len(descriptor_names)}")

# Set-up the calculator
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# Calculate descriptors for all compounds
print("\nCalculating descriptors... (hulati boiii)")

desc_list = []
for smi in df['smiles']:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        desc_list.append(list(calc.CalcDescriptors(mol)))
    else:
        desc_list.append([np.nan] * len(descriptor_names))

# Create descriptor dataframe
desc_df = pd.DataFrame(desc_list, columns=descriptor_names)

print(f"Descriptor matrix shape: {desc_df.shape}")
print("Done!")

# Combine with original dataframe
df_final = pd.concat([df.reset_index(drop=True), desc_df], axis=1)

print(f"Final shape: {df_final.shape}")
print(f"Any NaN values: {df_final.isnull().sum().sum()}")

# Save
df_final.to_csv('../data/AChE_descriptors.csv', index=False)
print("SAVED: /data/AChE_descriptors.csv")