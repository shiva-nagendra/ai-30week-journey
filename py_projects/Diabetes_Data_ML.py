#Diabetes dataset evaluation by ML

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

#load data
data = load_diabetes(as_frame=True)
df = data.frame

#split features into X and Y(target)
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

#Evaluate
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2score = r2_score(y_test,y_pred)

print("\nreal answers:",y_test)
print("\nMean square evaluation:",round(mse,2))
print("\nroot mean square evaluation:",round(rmse,2))
print("\nRoot square:",r2score)

