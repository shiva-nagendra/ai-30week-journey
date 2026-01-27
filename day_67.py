#Numerical Distributions with Seaborn 

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("telecom_churn.csv")

df["TotalCharges"]= pd.to_numeric(df["TotalCharges"],errors="coerce")
df = df.dropna(subset=["TotalCharges"])

#Distribution of monthly charges
sns.histplot(data=df, x="MonthlyCharges",bins=10, kde=True)
plt.title("Distribution of Monthly charges")
plt.show()

#Monthly charges by churn
sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=10, kde=True)
plt.title("Monthly distribution by churn")
plt.show()