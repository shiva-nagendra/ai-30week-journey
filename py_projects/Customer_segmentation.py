#week 15 project - Customer segmentation
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

#Load data
df = pd.read_csv("Mall_customers.csv")
print(df.head())

#Feature selection

x = df[["Annual Income (k$)","Spending Score (1-100)"]]


#Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Silhouette visualisation
sil_score = []

for k in range(2, 11):
    kmean = KMeans(n_clusters=k, random_state=42)
    labels = kmean.fit_predict(x_scaled)
    sil_score.append(silhouette_score(x_scaled,labels))
    
plt.plot(range(2,11), sil_score, marker = 'o')
plt.title("Silhouette score")
plt.xlabel("K")
plt.ylabel("Score")
plt.show()


#Elbow visualization
inertia_val = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_scaled)
    inertia_val.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia_val, marker = "o")
plt.title("Elbow K")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.show()


