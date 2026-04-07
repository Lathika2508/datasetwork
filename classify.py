import pandas as pd

# 1. Load dataset
df = pd.read_csv("covid.csv")

# 2. Data Cleaning
df = df.dropna()
df = df.drop_duplicates()

# 3. Select Features (avoid leakage)
X = df[['Confirmed', 'Deaths', 'Recovered']]

# 4. Select Target
y = df['Region']

# 5. Encode Target (text → numbers)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# 6. Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Scaling (important for KNN & Logistic Regression)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Import Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# 9. Initialize Models
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier()

# 10. Train Models
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
knn.fit(X_train, y_train)

# 11. Predictions
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_knn = knn.predict(X_test)

# 12. Evaluation
from sklearn.metrics import accuracy_score

acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_knn = accuracy_score(y_test, y_pred_knn)

# 13. Print Results
print("Decision Tree Accuracy:", acc_dt)
print("Random Forest Accuracy:", acc_rf)
print("Logistic Regression Accuracy:", acc_lr)
print("KNN Accuracy:", acc_knn)

# 14. Find Best Model
accuracies = {
    "Decision Tree": acc_dt,
    "Random Forest": acc_rf,
    "Logistic Regression": acc_lr,
    "KNN": acc_knn
}

best_model = max(accuracies, key=accuracies.get)

print("\nBest Model:", best_model)
print("Best Accuracy:", accuracies[best_model])