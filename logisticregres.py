import pandas as pd

# Load dataset
df = pd.read_csv("covid.csv")

# Clean
df = df.dropna()

# Features only (no target)
X = df[['Confirmed', 'Deaths', 'Recovered']]

# Scaling (important)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Cluster labels
df['Cluster'] = kmeans.labels_

print(df[['Confirmed','Deaths','Recovered','Cluster']].head())

import matplotlib.pyplot as plt

plt.scatter(df['Confirmed'], df['Deaths'], c=df['Cluster'])
plt.xlabel("Confirmed")
plt.ylabel("Deaths")
plt.title("K-Means Clustering")
plt.show()