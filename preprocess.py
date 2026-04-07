import pandas as pd

# 1. Load dataset
df = pd.read_csv("data.csv")

# 2. DATA CLEANING
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))  # numeric columns
df = df.fillna(method='ffill')              # non-numeric

# 3. ENCODING (for categorical data)
df = pd.get_dummies(df)

# 4. SPLIT FEATURES & TARGET
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 5. SCALING / NORMALIZATION
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 👉 Choose ONE based on dataset

# Option A: Standardization (default choice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Option B: Normalization (uncomment if needed)
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# 6. HANDLE IMBALANCED DATA (only if classification)
from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_res, y_res = sm.fit_resample(X_scaled, y)

# 7. CREATE FINAL DATAFRAME
df_final = pd.concat(
    [pd.DataFrame(X_res), pd.DataFrame(y_res)],
    axis=1
)

# Rename columns properly
df_final.columns = list(df.columns)

# 8. SAVE PREPROCESSED DATA
df_final.to_csv("preprocessed_data.csv", index=False)

print("Preprocessing completed successfully!")