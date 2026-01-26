#Seaborn(smart drawing for data) + Categorical Comparisons (Seeing churn patterns clearly)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("telecom_churn.csv")

#clean total charges
df["TotalCharges"] = pd. to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

#contract vs churn
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract type")
plt.show()

#internet service vs churn

sns.countplot(data=df, x="InternetService", hue="Churn")
plt.title("Internet service by churn")
plt.show()