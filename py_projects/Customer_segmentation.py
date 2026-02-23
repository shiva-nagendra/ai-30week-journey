#week 15 project - Customer segmentation
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

#Load data
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

#Feature selection

x = df[["Annual Income (k$)","Spending Score (1-100)"]]


#Scaling
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

#Silhouette visualisation
sil_scores = []

for k in range(2, 11):
    kmeanss = KMeans(n_clusters=k, random_state=42)
    labels = kmeanss.fit_predict(x_scaled)
    sil_scores.append(silhouette_score(x_scaled, labels))

plt.plot(range(2, 11), sil_scores, marker = 'o')
plt.title("Silhouette score")
plt.show()

#apply K means
for k in range(1, 11):
    kmeanss = KMeans(n_clusters=5, random_state=42)
    clusters = kmeanss.fit_predict(x_scaled)

    df["clusters"] = clusters

#visualise clusters

plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=clusters)
plt.scatter(
    kmeanss.cluster_centers_[:,0],
    kmeanss.cluster_centers_[:, 1],
    marker="x",
    s=200
)

plt.title("customer segments")
plt.xlabel("Income")
plt.ylabel("Spending score")
plt.show()

