#practice day 73

from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([1500, 1600, 1800, 1900, 2000]).reshape(-1, 1)
y = np.array([10, 12, 14, 14.8, 16])

model = LinearRegression()

model.fit(x, y)

new =  np.array([[2200]])
prediction = np.round(model.predict(new),2)

print("new cc prediction: ", prediction, "lakhs")
