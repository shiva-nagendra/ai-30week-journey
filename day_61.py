#correlation practice

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("day_60.csv")
print("\nData:\n",df)

#correlation matrix
corr = df.corr(numeric_only=True)
print("\nCorrelation matrix\n",corr)

#correlation score
score_corr = corr["score"].sort_values(ascending=False)
print("\nscore correlation:\n",score_corr)

#scatter plots with strongest relationship

#score vs study hours
plt.figure()
plt.scatter(df["score"],df["study_hours"])
plt.title("score vs study_hours")
plt.xlabel("score")
plt.ylabel("study_hours")
plt.show()



#score vs income
plt.figure()
plt.scatter(df["score"],df["income"])
plt.xlabel("Score")
plt.ylabel("income")
plt.title("Score vs income")
plt.show()
