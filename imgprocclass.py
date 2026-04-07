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


#nexttneww

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models,layers


data=[]
labels=[]
import os
print(os.getcwd())

data_path = "."
for category in os.listdir(data_path):
    category_path = os.path.join(data_path, category)

    # ✅ skip files like image.py
    if not os.path.isdir(category_path):
        continue

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        try:
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128,128))
            img = cv2.GaussianBlur(img, (5,5), 0)

            data.append(img)
            labels.append(category)

        except:
            pass
X=np.array(data)/255.0
y=np.array(labels)

le=LabelEncoder()
y=le.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')
 ])

model.compile(
   optimizer='adam',
   loss='binary_crossentropy',
   metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)
img = cv2.imread("cats/cat_1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128,128))
img = img / 255.0

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
