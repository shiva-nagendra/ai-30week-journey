#week 13, day 1 & 2: Feature scaling with standard scalar

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#Load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#split features and target
x = df.drop(columns=["target"])
y = df["target"]

#split train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3,random_state=42
)

# Create scalar
scaler = StandardScaler()

#fit only training data
x_train_scaled = scaler.fit_transform(x_train)

#fit testing data
x_test_scaled = scaler.transform(x_test)

#model train on scaled data
model = LogisticRegression(max_iter=3000)
model.fit(x_train_scaled,y_train)

#predict
y_pred = model.predict(x_test_scaled)

#evaluate
acc = accuracy_score(y_test,y_pred)

print("\nAccuracy score with scaling:",round(acc,3))


