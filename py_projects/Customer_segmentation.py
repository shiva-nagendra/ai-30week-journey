#week 15 project - Customer segmentation

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

#Load data
df = pd.read_csv("Mall_customers.csv")
print(df.head())

