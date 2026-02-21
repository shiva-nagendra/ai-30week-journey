#week 15 day 6
# Customer segmentation with interpretation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
x = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# Add cluster column
df["Cluster"] = clusters

# Cluster interpretation table
print("\nCluster Means:")
print(df.groupby("Cluster").mean())

# Optional: Assign segment names (edit based on your results)
segment_names = {
    0: "Regular",
    1: "VIP",
    2: "Budget",
    3: "Potential",
    4: "Impulsive"
}

df["Segment"] = df["Cluster"].map(segment_names)

print("\nSegment counts:")
print(df["Segment"].value_counts())

# Visualization
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=clusters)
plt.scatter(
    kmean.cluster_centers_[:, 0],
    kmean.cluster_centers_[:, 1],
    marker="x",
    s=200
)

plt.title("Customer Segments")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()

