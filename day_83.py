#week 13 day 3: comparing with and without scaling
#comparing coefficients before and after scaling

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#split features and target
x = df.drop(columns=["target"])
y = df["target"]

#train and test split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.3, random_state=42
)
#---model without scaling-----#
#Raw model
Raw_model = LogisticRegression(max_iter=3456)
Raw_model.fit(x_train,y_train)

y_pred_raw = Raw_model.predict(x_test)
acc_raw = accuracy_score(y_test,y_pred_raw)

#---model with scaling----#
#scaled data
scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

model_scaled = LogisticRegression(max_iter=3456)
model_scaled.fit(scaled_x_train,y_train)

y_pred_scaled = model_scaled.predict(scaled_x_test)
acc_scaled = accuracy_score(y_test,y_pred_scaled)

#Results
print("\nAccuracy without scaling:",round(acc_raw,3))
print("\nAccuracy with scaling:",round(acc_scaled,3))

#compare coefficients
print("\nRaw model coefficienets (first five):")
print(np.round(Raw_model.coef_[0][:5]),3)

print("Scaled model coefficienets (first five):")
print(np.round(model_scaled.coef_[0][:5]),3)

