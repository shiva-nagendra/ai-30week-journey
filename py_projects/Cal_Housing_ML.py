#California housing price prediction by ML

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

#load data
data = fetch_california_housing(as_frame=True, download_if_missing=True)
df = data.frame

#split features
x = df.drop(columns=["target"])
y = df["target"]

#train vs test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state= 42
)

#model selection
model = LinearRegression()
model.fit(x_train, y_train)

#prediction
y_pred = model.predict(x_test)

mse = mean_squared_error(y_pred)
rsme = np.sqrt(mse)


