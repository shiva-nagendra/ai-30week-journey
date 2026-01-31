#weekly project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data

df = pd.read_csv("telecom_churn.csv")

# 1. clean data
df["TotalCharges"]= pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# 2. overall churn distribution
sns.countplot(data=df, x="Churn")
plt.title("Overall churn distribution")
plt.show()

# 3. churn by contract type
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by contract type")
plt.show()

# 4. Monthly charges vs churn
sns.boxplot(data=df, y="MonthlyCharges", x="Churn")
plt.title("Monthly charges vs churn")
plt.show()

# 5. Tenure vs churned
sns.violinplot(data=df, y="tenure", x="Churn")
plt.title("Tenure vs churn")
plt.show()

# 6. correlation heatmap

corr = df.corr(numeric_only=True)

plt.figure(figsize=(6,4))
sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.title("Correlation heatmap")
plt.show()

# Final Conclusion
# The strongest churn drivers observed are:
# - Contract type
# - Tenure
# - Monthly charges

