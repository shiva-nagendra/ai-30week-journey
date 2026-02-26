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
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(x_scaled)

#Add cluster column
df["Cluster"] = clusters 

#apply pca
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

#Plot pca with cluster
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:,1],
    s=200,
    marker="x",
    color = "red"
)
plt.title("Customer segments(PCA view)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()