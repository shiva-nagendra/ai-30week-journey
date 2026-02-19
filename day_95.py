#week 15 day 4
#Silhouette score for cluster quality

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

#Generate synthetic data
x,_ = make_blobs(
    n_samples=300,
    centers=4,
    random_state=42
)

#Scale data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#try multiple k values
for k in range(2,9):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(x_scaled)
    score = silhouette_score(x_scaled,labels)
    print(f"K = {k}, Silhouette score = {round(score,3)}")