#Week 16 day 3
#PCA + Clustering(Kmeans)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

#Load data
df = pd.read_csv("Mall_Customers.csv")

#Gendr encode and x
df["Gender"] = df["Gender"].map({"Male":0,"Female":1})

x = df[["Gender","Age","Annual Income (k$)", "Spending Score (1-100)"]]

#Scale data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Apply pca
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

#Apply kmeans
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(x_pca)

print("PCA Variance:\n")


#plot
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters, cmap="viridis" )
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker="x",
    s=200,
    color="red" 
)
plt.title("PCA-Kmeans")
plt.xlabel("PC1")
plt.ylabel("PC2")

