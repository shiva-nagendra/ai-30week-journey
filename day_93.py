# Week 15 Day 2
# Intro to K-Means Clustering

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Create synthetic data
X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

# 2️⃣ Scale data (important for distance-based models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# 4️⃣ Get cluster labels
labels = kmeans.labels_

# 5️⃣ Plot clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.title("K-Means Clustering")
plt.show()

