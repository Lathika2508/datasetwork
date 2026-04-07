import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers

data = []
labels = []
data_path = "dataset"

for category in sorted(os.listdir(data_path)):  # fix 3: sorted
    category_path = os.path.join(data_path, category)
    if not os.path.isdir(category_path):
        continue
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            data.append(img)
            labels.append(category)
        except Exception as e:  # fix 6: don't swallow errors silently
            print(f"Skipping {img_path}: {e}")

# fix 4: guard against empty dataset
if len(data) == 0:
    raise ValueError("No images loaded. Check your dataset path and folder structure.")

X = np.array(data, dtype=np.float32) / 255.0
y = np.array(labels)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_classes = len(le.classes_)  # fix 1: cleaner
print("Number of classes:", num_classes)
print("Classes:", le.classes_)

if num_classes < 2:
    raise ValueError("Need at least 2 classes to train a classifier.")  # fix 5

# fix 7: stratified split for balanced classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
])

if num_classes == 2:
    model.add(layers.Dense(1, activation='sigmoid'))
    loss_fn = 'binary_crossentropy'
else:
    model.add(layers.Dense(num_classes, activation='softmax'))
    loss_fn = 'sparse_categorical_crossentropy'

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# Predict
img = cv2.imread("test.jpg")
if img is None:
    print("Test image not found!")
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    prediction = model.predict(img)

    if num_classes == 2:
        pred_index = int(prediction[0][0] > 0.5)  # fix 2: unambiguous
    else:
        pred_index = int(np.argmax(prediction))

    print("Predicted Class:", le.inverse_transform([pred_index])[0])


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


#nextneww2

import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
data = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/EC1003/Downloads/ds/xray_dataset_covid19/train",
    image_size=(128, 128),
    batch_size=32
)

# Normalize
data = data.map(lambda x, y: (x / 255.0, y))

# Model
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, epochs=5)
model.save("model.h5")


