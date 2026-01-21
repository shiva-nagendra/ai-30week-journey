#Outliers, Skewness & Data Issues (EDA reality check)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("day_62.csv")
print("\nData:\n",df) 

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

