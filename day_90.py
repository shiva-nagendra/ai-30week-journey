#week 14 day 4

# KNN (Distance-based model)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#split features and target
x = df.drop(columns=["target"])
y = df["target"]

#split train test
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

#KNN model(with scaling)
KNN_model = Pipeline([
    ("scaler",StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

#mdoel fit and predict
KNN_model.fit(x_train,y_train)
y_pred = KNN_model.predict(x_test)
acc = accuracy_score(y_test,y_pred)

print("\nKNN model accuracy score:",acc)