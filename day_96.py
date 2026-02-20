#week 15 day 5
#cust segmentation project p-1

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Load dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

#Select relevant features
x = df[["Annual Income (k$)","Spending Score (1-100)"]]

#Scale data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Elbow method(inertia)
inertia_values = []

for k in range(1, 11):
    kmean = KMeans(n_clusters=k, random_state=42)
    kmean.fit(x_scaled)
    inertia_values.append(kmean.inertia_)

plt.plot(range(1, 11), inertia_values, marker = 'o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()

#Apply K-means
kmean = KMeans(n_clusters=4, random_state=42)
clusters = kmean.fit_predict(x_scaled)

df["Cluster"] = clusters

#visualise k means

plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=clusters)
plt.scatter(
    kmean.cluster_centers_[:, 0],
    kmean.cluster_centers_[:,1],
    marker="x",
    s=200
)

plt.title("customer segments")
plt.xlabel("Income")
plt.ylabel("Spending score")
plt.show()