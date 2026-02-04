#train test split

#training data = model learns patterns
#testing data = model is evaluated honestly

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

#dataset
x = np.array([500,600,1000,1200,1500,1800,2000]).reshape(-1,1)#sq.ft
y = np.array([30,38,65,80,100,120,135])#lakhs


#split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state = 42
)

#create model
model = LinearRegression()

#train on training data
model.fit(x_train, y_train)

#predict on test data
y_pred = model.predict(x_test)

print("\nTest inputs:\n", x_test)
print("\nActual prices:\n",y_test)
print("\nPredicted prices:\n",np.round(y_pred,2))

