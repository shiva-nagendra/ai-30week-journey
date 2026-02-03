#Linear regression

from sklearn.linear_model import LinearRegression
import numpy as np

#House size(sq.ft)
x = np.array([500, 600, 1000, 1200, 1500]).reshape(-1, 1)

#House price in lakhs
y = np.array([30,50,65,80,100])

#Create model
model = LinearRegression()

#Train model
model.fit(x, y)

#make prediction
new_size = np.array([[1700]])
predicted_price = np.round(model.predict(new_size),2)


print("\nPredicted price for 1700 sq.ft. house:\n", predicted_price,"lakhs")

