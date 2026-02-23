#week 15 project - Customer segmentation

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd

#Load data
df = pd.read_csv("Mall_customers.csv")
print(df.head())

#Feature selection

x = df[["Annual Income (k$)","Spending Score (1-100)"]]


#Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Elbow visualization
inertia_val = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_scaled)
    inertia_val.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia_val, marker = "o")
plt.title("Elbow K")
plt.xlabel("")
plt.ylabel("")
plt.show()

#Apply k means
#Elbow

for k in range(1, 11):
    kmean = KMeans(n_clusters=k,random_state=42)
    clusters = kmean.fit_predict(x_scaled)

