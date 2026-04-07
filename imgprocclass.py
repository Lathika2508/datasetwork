import os
import cv2
import numpy as np
import pandas as pd

# 1. Load dataset
data = []
labels = []

path = "dataset/"   # folder containing class folders

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        # Preprocessing
        img = cv2.resize(img, (50, 50))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data.append(img.flatten())
        labels.append(folder)

# Convert to numpy
X = np.array(data)
y = np.array(labels)

# 2. Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# 3. Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model (KNN)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# 5. Evaluate
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# 🔥 6. PREDICTION ON NEW IMAGE

# Load new image
new_img = cv2.imread("test.jpg")

# Same preprocessing (VERY IMPORTANT)
new_img = cv2.resize(new_img, (50, 50))
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

# Flatten
new_img = new_img.flatten().reshape(1, -1)

# Predict
pred = model.predict(new_img)

# Convert back to label
pred_label = le.inverse_transform(pred)

print("Predicted Class:", pred_label[0])