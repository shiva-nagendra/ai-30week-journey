#Visualization mini-project for telecom churn analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data
df = pd.read_csv("telecom_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
df = df.dropna(subset=["TotalCharges"])


## Plot 1: Overall Churn Distribution
# - What it shows:
#   Shows the number of churned vs non-churned customers.
# - Key insight:
#   Churn is (balanced / imbalanced), which impacts ML model choice.
# - Business implication:
#   Class imbalance may require careful evaluation metrics.

sns.countplot(data=df,x="Churn")
plt.title("Churn Distribution")
plt.show()


## Plot 2: Churn by Contract Type
# - What it shows:
#   Month-to-month customers churn more than long-term contracts.
# - Key insight:
#   Contract duration is a strong churn indicator.
# - Business implication:
#   Incentivizing longer contracts may reduce churn.


sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract vs Churn")
plt.show()

#monthly charges vs churn
# What it shows:

# Displays the distribution of monthly charges for churned vs non-churned customers.

# Key insight:
# Churned customers tend to have higher monthly charges compared to those who stay.

# Business implication:
# High pricing may increase churn risk; pricing strategies or discounts could help
# retain customers.

sns.boxplot(data=df,x="Churn", y="MonthlyCharges")
plt.title("MOnthly charges vs churn")
plt.show()

#Tenure vs churn
sns.violinplot(data=df, x="Churn", y="tenure")
plt.title("Tenure vs churn")
plt.show()

#Correlation heatmap
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature correlation heatmap")
plt.show()

