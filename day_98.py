# week 16 day 1 & day 2
# Dimensionality Reduction - Principal componenet analysis(PCA)

#PCA intuition - Visual thinking

from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

x = df[["Gender","Age","Annual Income (k$)", "Spending Score (1-100)"]]

#scale data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Apply pca
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

print("PCA Variance:\n")
print(pca.explained_variance_ratio_)

#Visuallize
plt.scatter(x_pca[:, 0], x_pca[:, 1])
plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


