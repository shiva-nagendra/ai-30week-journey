#EDA Mini-Project: Student Performance Analysis

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Load data

df = pd.read_csv("DAy_64.csv")
print("\nData: \n", df)

#Describe
print("\nDescribe data:\n",df.describe())

#Correlation with target
corr = df.corr(numeric_only=True)["score"]
print("\nCorrelation with score:\n",corr)

#remove outlier
df_clean=df[df["income"]<=100000]
print("\nCorrelation after removing the outlier:\n")
print(df_clean.corr(numeric_only=True)["score"])

#log transform income
df_clean["income_log"]=np.log(df_clean["income"])
print("\nCorrelation with log income:\n",
      df_clean[["income_log","score"]].corr())

#Boxplot
plt.figure()
plt.boxplot(df_clean["score"])
plt.title("score boxplot")
plt.ylabel("score")
plt.show()

#Histogram
plt.figure()
plt.hist(df_clean["income"], bins=6)
plt.title("income distribution")
plt.ylabel("frequency")
plt.xlabel("income")
plt.show()

#scatterplot
plt.figure()
plt.scatter(df_clean["study_hours"],df_clean["score"])
plt.title("study hours vs score")
plt.xlabel("study hours")
plt.ylabel("score")
plt.show()






