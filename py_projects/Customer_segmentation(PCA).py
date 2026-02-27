#week 16 project
#Customer segmentation (PCA)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

#Load data
df = pd.read_csv("Mall_Customers.csv")

#Encode gender
df["Gender"] = df["Gender"].map({"Male":0,"Female":1})

#feature selection
x = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

#Scale data
scaler = StandardScaler()
x_Scaled = scaler.fit_transform(x)

#Apply pca
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_Scaled)

#apply kmeans
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(x_pca)

#add cluster column
df["Cluster"] = clusters

#Visualize 
plt.scatter(x_pca[:,0],x_pca[:,1],c=clusters)
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker="x",
    s=200,
    color="red"
)
plt.title("Customer segentation(PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#Interpretation
cluster_mean = df.groupby("Cluster").mean(numeric_only=True)
print("\ncluster mean:", cluster_mean)

#PCA explained variance
print("\nPCA variance:",pca.explained_variance_ratio_)
print("\nCummulative ratio:",pca.explained_variance_ratio_.cumsum())