import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\PMLS\Desktop\maintainance predictive system\raw_dataset_50k.csv")

print("Original shape:", df.shape)

# 1. Remove UDI and Product ID columns
df = df.drop(columns=['UDI', 'Product ID'])

# 2. Map Type column: H=1, M=2, L=3
type_mapping = {'H': 1, 'M': 2, 'L': 3}
df['Type'] = df['Type'].map(type_mapping)

# 3. Check for garbage/missing values
null_percent = (df.isnull().sum() / len(df)) * 100
print("\nNull % per column:\n", null_percent)

# Drop rows with nulls only if null % < 30 in that column
for col in df.columns:
    pct = null_percent[col]
    if pct == 0:
        continue
    elif pct < 30:
        df = df.dropna(subset=[col])
    elif pct >= 50:
        df = df.drop(columns=[col])
        print(f"Dropped column {col} (>=50% missing)")

# 4. Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"\nDuplicates removed: {before - len(df)}")

print("\nFinal shape:", df.shape)
print(df.head())

df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
df['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Tool_Torque'] = df['Tool wear [min]'] * df['Torque [Nm]']

df.to_csv('cleaned_dataset.csv', index=False)