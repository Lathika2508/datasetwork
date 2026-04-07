import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt

df=pd.read_csv("covid.csv")
print("________head________")
print(df.head())
print("________mean________")
print(df.mean(numeric_only=True))
print
print(df.mean(axis=1, numeric_only=True))
print("________mode________")
print(df.mode(numeric_only=True))
print("________median________")
print(df.median(numeric_only=True))
print("________quantile________")

print(df["Confirmed"].quantile(0.25))


print(df.describe())

print(df.var(numeric_only=True))
print(df.std(numeric_only=True))
print("________min________")

print(df["Confirmed"].min())
print("________max________")
print(df["Confirmed"].max())
print("________skewness________")
print(df["Confirmed"].skew())
print("________kurtosis________")
print(df["Confirmed"].kurt())