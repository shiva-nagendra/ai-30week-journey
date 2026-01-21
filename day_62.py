#Outliers, Skewness & Data Issues (EDA reality check)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("day_62.csv")
print("\nData:\n",df) 

#compare correlation with and without outlier
corr_with = df.corr(numeric_only=True)["score"]
print("\nCorrelation WITH outlier:\n", corr_with)

df_no_outlier = df[df["income"]<100000]
corr_without = df_no_outlier.corr(numeric_only=True)["score"]
print("\nCorrelation without outlier:\n",corr_without)

#box plot to dataset outliers
plt.figure()
plt.boxplot(df["score"])
plt.title("Score Boxplot")
plt.ylabel("Score")
plt.show()

#Histogram to see skewness
plt.figure()
plt.hist(df["income"], bins=6)
plt.title("Income distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

#Scatter plot to see effect of outliers
plt.figure()
plt.scatter(df["study_hours"], df["score"])
plt.title("Study hours vs score")
plt.xlabel("study hours")
plt.ylabel("score")
plt.show()



