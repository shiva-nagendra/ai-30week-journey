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

#apply kmeans
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(x_scaled)

#apply pca
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
