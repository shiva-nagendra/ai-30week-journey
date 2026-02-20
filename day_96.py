#week 15 day 5
#cust segmentation project p-1

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Load dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

#Select relevant features
x = df[["Annual Income (k$)","Spending Score (1-100)"]]

