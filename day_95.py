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

