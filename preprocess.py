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



#NEXTNEWW

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


df=pd.read_csv("covid.csv")

#clean

df = df.drop_duplicates()

#filling

df = df.fillna(df.mean(numeric_only=True))
df = df.ffill()

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

scaler= StandardScaler()
X_scaled=scaler.fit_transform(X).astype('float32')

sm=SMOTE()
X_res,Y_res=sm.fit_resample(X_scaled,Y)

df_fin=pd.concat([pd.DataFrame(X_res),pd.DataFrame(Y_res)],axis=1)

df_fin.columns=list(df.columns)

df_fin.to_csv("preprocessed.csv")
print("Successfully preprocessed")
