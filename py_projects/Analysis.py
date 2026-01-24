#week 9 project Telecom churn EDA

import numpy as np
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

#monthly charges vs churned
monthly = df.groupby("Churn")["MonthlyCharges"].mean()
print("\nMonthly charges vs churn\n",monthly)

#total charges churned vs non churned
total = df.groupby("Churn")["TotalCharges"].mean()
print("\nTotal charges avg. vs churned\n",total)

#Tenure compairsion
print("\nAvg tenure by churn:\n", 
      df.groupby("Churn")["tenure"].mean())
