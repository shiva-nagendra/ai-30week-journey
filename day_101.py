#week 16 day 5
# PCA + Cluster visualisation

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

#Load data
df = pd.read_csv("Mall_Customers.csv")

#Encode Gender and x

df["Gender"] = df["Gender"].map({"Male":0, "Female":1})

x = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

#Scale data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

