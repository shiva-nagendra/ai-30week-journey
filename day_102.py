#week 16 Day 6
# PCA + Kmeans pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

#Load data
df = pd.read_csv("Mall_Customers.csv")

#encode gender and x
df["Gender"] = df["Gender"].map({"Male":0,"Female":1})
x = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]


#scale data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Apply pca
pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(x_scaled)


