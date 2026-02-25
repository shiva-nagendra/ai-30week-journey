#Week 16 day 4
#Scree plot and choosing PCA    

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

pca_full = PCA()
pca_full.fit(x_scaled)

plt.plot(pca_full.explained_variance_ratio_, marker='o')
plt.title("Scree Plot")
plt.xlabel("Principal Components")
plt.ylabel("Variance Explained")
plt.show()

import numpy as np

cum_var = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(cum_var, marker='o')
plt.title("Cumulative Variance")
plt.xlabel("Components")
plt.ylabel("Total Variance")
plt.show()