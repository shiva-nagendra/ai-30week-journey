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

#Boxplot

plt.figure()
plt.boxplot(df["score"])
plt.title("score boxplot")
plt.xlabel("score")
plt.show()