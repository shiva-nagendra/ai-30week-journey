#weekly project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data

df = pd.read_csv("telecom_churn.csv")

#clean data
df["TotalCharges"]= pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

#overall churn distribution
sns.countplot(data=df, x="Churn")
plt.title("Overall churn distribution")
plt.show()

