import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("covid.csv")

# Clean data
df = df.dropna()

# 1. Histogram (clear)
plt.figure()
plt.hist(df['Confirmed'], bins=20)
plt.title("Distribution of Confirmed Cases")
plt.xlabel("Confirmed")
plt.ylabel("Frequency")
plt.show()

#  2. Boxplot
plt.figure()
sns.boxplot(x=df['Confirmed'])
plt.title("Boxplot of Confirmed Cases")
plt.show()

#  3. Scatter plot
plt.figure()
sns.scatterplot(x=df['Confirmed'], y=df['Deaths'])
plt.title("Confirmed vs Deaths")
plt.show()

#  4. Bar plot (use meaningful values)
plt.figure()
sns.barplot(x=df.head(10).index, y=df.head(10)['Confirmed'])
plt.title("Top 10 Confirmed Cases")
plt.xticks(rotation=45)
plt.show()

# 5. Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()