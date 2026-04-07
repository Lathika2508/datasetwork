import pandas as pd

# Load dataset
df = pd.read_csv("covid.csv")

# Clean
df = df.dropna()

# Features (input)
X = df[['Deaths', 'Recovered', 'Active']]

# Target (numeric)
y = df['Confirmed']

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import r2_score

print("R2 Score:", r2_score(y_test, y_pred))