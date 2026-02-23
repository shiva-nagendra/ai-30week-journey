#week 15 project - Customer segmentation

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

#Load data
df = pd.read_csv("Mall_customers.csv")
print(df.head())

#Feature selection

x = df[["Annual Income (k$)","Spending Score (1-100)"]]


#Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

