#plottting using pandas

import pandas as pd
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("day_60.csv")
print(df)

#basic stats
print("\nDescribe: \n", df.describe())

#Histogram -score distribution
plt.figure()
plt.hist(df["score"], bins=5)
plt.title("Score distribution")
plt.xlabel("score")
plt.ylabel("Frequency")
plt.show()

#scatter plot - study hours vs score
plt.figure()
plt.scatter(df["score"],df["study_hours"])
plt.xlabel("score")
plt.ylabel("study hours")
plt.title("score vs study hours")
plt.show()

#scatter plot - income vs score
plt.figure()
plt.scatter(df["income"], df["score"])
plt.xlabel("income")
plt.ylabel("score")
plt.title("income vs score")
plt.show()

#plotting income vs score
plt.figure()
plt.plot(df["income"], df["score"], marker = "o", markersize = 10)
plt.xlabel("income")
plt.ylabel("score")
plt.title("income vs score")
plt.show()
