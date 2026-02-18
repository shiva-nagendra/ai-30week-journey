#week 15 day 3
#Elbow method for optimal K

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

#generate synthetic data
x, _ = make_blobs(
    n_samples=450,
    centers=4,
    random_state=42
)

#scale data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#calculate inertia for k from 1 to 10
inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_scaled)
    inertia_values.append(kmeans.inertia_)

#plot elbow
plt.plot(range(1,11),inertia_values, marker = "o")
plt.xlabel("Number of clusters(k)")
plt.ylabel("Inertia")
plt.title("Elbow method")
plt.show()
