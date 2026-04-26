import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# ── CONFIG ── only change these 3 lines for any dataset
FILE      = "Corona_NLP_train.csv"
TEXT_COL  = "OriginalTweet"
LABEL_COL = "Sentiment"

# Load & clean
df = pd.read_csv(FILE, encoding="latin-1")[[TEXT_COL, LABEL_COL]].dropna()

# Encode labels
df[LABEL_COL] = LabelEncoder().fit_transform(df[LABEL_COL])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL], df[LABEL_COL], test_size=0.2, random_state=42
)

# TF-IDF
tf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tf.fit_transform(X_train)
X_test_tfidf  = tf.transform(X_test)

# SMOTE
X_res, y_res = SMOTE().fit_resample(X_train_tfidf, y_train)

# Train
model = LogisticRegression(max_iter=200)
#model = LinearSVC(max_iter=200, C=1.0)
model.fit(X_res, y_res)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
