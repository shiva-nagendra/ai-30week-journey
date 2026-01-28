#Correlation Heatmaps & Feature Relationships

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data
df = pd.read_csv("telecom_churn.csv")

#clean datagit status
df["TotalCharges"]= pd.to_numeric(df["TotalCharges"], errors="coerce")
df= df.dropna(subset=["TotalCharges"])

#correlation matrix
corr = df.corr(numeric_only=True)
print(corr)

#correlatiion heatmap
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation heatmap")
plt.show()

plt.figure()
sns.countplot(data=df, x="MonthlyCharges",hue="Churn")
plt.show()