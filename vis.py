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

#nexttneww

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("covid.csv")

# Safe cleaning
print("Columns:", df.columns.tolist())
print("Rows before:", len(df))
df = df.dropna(subset=['Confirmed', 'Deaths'])
print("Rows after:", len(df))

# 1. Histogram
plt.figure()
plt.hist(df['Confirmed'], bins=20, color='steelblue', edgecolor='black')
plt.title("Distribution of Confirmed Cases")
plt.xlabel("Confirmed")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Boxplot
plt.figure()
sns.boxplot(x=df['Confirmed'])
plt.title("Boxplot of Confirmed Cases")
plt.tight_layout()
plt.show()

# 3. Scatter plot
plt.figure()
sns.scatterplot(x=df['Confirmed'], y=df['Deaths'])
plt.title("Confirmed vs Deaths")
plt.tight_layout()
plt.show()

# 4. Bar plot — actual top 10, with country names
plt.figure()
top10 = df.nlargest(10, 'Confirmed')
sns.barplot(x=top10['Country/Region'], y=top10['Confirmed'])  # adjust column name
plt.title("Top 10 Countries by Confirmed Cases")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 5. Heatmap — focused columns only
plt.figure()
cols = ['Confirmed', 'Deaths', 'Recovered', 'Active']
available = [c for c in cols if c in df.columns]  # only use columns that exist
sns.heatmap(df[available].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#nextnew2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("your_file.csv")
df = df.dropna()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categoric_cols = df.select_dtypes(include='object').columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categoric_cols)

# Histogram for every numeric column
for col in numeric_cols:
    plt.figure()
    plt.hist(df[col], bins=20, edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Countplot for every categorical column
for col in categoric_cols:
    plt.figure()
    order = df[col].value_counts().index[:10]
    sns.countplot(y=df[col], order=order)
    plt.title(f"Top values in {col}")
    plt.tight_layout()
    plt.show()

# Scatter for first 2 numeric columns if available
if len(numeric_cols) >= 2:
    plt.figure()
    sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]])
    plt.title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
    plt.tight_layout()
    plt.show()

# Heatmap only if enough numeric columns
if len(numeric_cols) >= 2:
    plt.figure()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
