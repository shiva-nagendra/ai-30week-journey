#plottting using pandas

import pandas as pd
import matplotlib.pyplot as plt

#load data

df = pd.read_csv("day_60.csv")
print(df)

#basic stats
print("\nDescribe: \n", df.describe())

#Histogram
plt.figure()
plt.hist(df["score"], bins=5)
plt.title("Score distribution")
plt.xlabel("score")
plt.ylabel("Frequency")
plt.show()