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

#Apply kmeans
Kmeans = KMeans(n_clusters=5, random_state=42)
clusters = Kmeans.fit_predict(x_pca)

#Visualize

plt.scatter(x_pca[:,0], x_pca[:,1], c=clusters)
plt.scatter(
    Kmeans.cluster_centers_[:,0],
    Kmeans.cluster_centers_[:,1],
    s=200,
    color="red",
    marker="x"
)
plt.title("Customer segmentation (PCA view)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#Add cluster column
df["Cluster"] = clusters 
print("\n Cluster mean:")
print(df.groupby("Cluster").mean(numeric_only=True))