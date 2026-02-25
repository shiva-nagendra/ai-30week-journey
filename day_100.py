#Week 16 day 4
#Scree plot and choosing PCA    

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load data
df = pd.read_csv("Mall_Customers.csv")

#Gender Encode & x
df["Gender"] = df["Gender"].map({"Male":0, "Female":1})

x = df[["Gender","Age","Annual Income (k$)", "Spending Score (1-100)"]]

#Scale data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Apply PCA
pca = PCA()
x_pca = pca.fit_transform(x_scaled)

#Explained variance
x_var = pca.explained_variance_ratio_
print(x_var)

#scree plot
plt.plot(range(1, len(x_var)+1), x_var, marker = "o")
plt.title("Scree plot")
plt.xlabel("Principal componenets")
plt.ylabel("Variance Explained")
plt.show()