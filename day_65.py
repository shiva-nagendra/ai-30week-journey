#week 10 day 1 
#Matplotlib Basics: Visualizing EDA Insights

import pandas as pd
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("telecom_churn.csv")

#convert total charges safely
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
df.dropna(subset=["TotalCharges"])

#Churncount plot
churn_counts = df["Churn"].value_counts()

plt.figure()
plt.bar(churn_counts.index, churn_counts.values)
plt.title("Monthly charges by churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly charges")
plt.show()

#Month;y charges by churn (boxplot)
plt.figure()
plt.boxplot(column="MonthlyCharges", by="Churn")
plt.title("MOnthly charges by churn")
plt.xlabel("Churn")
plt.ylabel("Monthly charges")
plt.show()


