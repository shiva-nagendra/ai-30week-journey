#week 9 project Telecom churn EDA

import pandas as pd
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("telecom_churn.csv")

print("\nNo. of columns:\n",df.columns)
print("\nHead:\n",df.head)
print("\nShape:\n",df.shape) #no.of rows and columns

#Basic cleanup
analyse_data = df.isnull().sum()
print("\nMissing values:\n",analyse_data)

df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].mean())
print("clean data:\n",df.isnull().sum())#check cleaned data


#customers churned vs did not churned
churned = df["Churn"].count
print("\ncustomers churned:\n",churned)
print("\nchurn count:\n",df["Churn"].value_counts())
# INSIGHT:
# - This shows how many customers churned vs stayed.
# - Use this to understand overall churn rate and class imbalance.

#monthly charges vs churned
monthly = df.groupby("Churn")["MonthlyCharges"].mean()
print("\nMonthly charges vs churn\n",monthly)
# INSIGHT:
# - Compare average MonthlyCharges for Churn = Yes vs No.
# - If churned customers pay higher monthly charges, pricing may be a churn driver.

#total charges churned vs non churned
total = df.groupby("Churn")["TotalCharges"].mean()
print("\nTotal charges avg. vs churned\n",total)
# INSIGHT:
# - TotalCharges reflects long-term customer value.
# - Lower average TotalCharges for churned users usually means shorter tenure.

#Tenure compairsion
print("\nAvg tenure by churn:\n", 
      df.groupby("Churn")["tenure"].mean())
# INSIGHT:
# - Shorter tenure for churned customers indicates early drop-off risk.
# - Tenure is typically one of the strongest churn predictors.

#contract type analysis
print("\nContract analysis:\n",
      df.groupby("Churn")["Contract"].value_counts())

print("\nChurn by Contract:\n",
      pd.crosstab(df["Contract"], df["Churn"], normalize="index"))
# INSIGHT:
# - Month-to-month contracts usually show the highest churn rate.
# - Long-term contracts (1-year, 2-year) reduce churn significantly.

#sr.citizen impact
print("\nsenior citizen impact on churn:\n",
      pd.crosstab(df["SeniorCitizen"],df["Churn"], normalize= "index"))
# INSIGHT:
# - SeniorCitizen = 1 often has a higher churn probability.
# - Age-related support, pricing, or usability may influence churn.



# FINAL CONCLUSION:
# - Key churn drivers observed: MonthlyCharges, tenure, Contract type.
# - Customers with high monthly charges, short tenure, and month-to-month contracts churn more.
# - This dataset is suitable for visualization.