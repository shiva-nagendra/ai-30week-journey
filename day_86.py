#week 13 day 6 Data leakage control and pipeline

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#split feature and target
x = df.drop(columns=["target"])
y = df["target"]

#split train and test
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

#pipelne build
pipe = Pipeline([
    ("scaler",StandardScaler()),
    ("model", LogisticRegression(max_iter=3340))
])

#model train
pipe.fit(x_train,y_train)

#mdel predict
y_pred=pipe.predict(x_test)
acc = round(accuracy_score(y_test,y_pred),3)
print("Pipeline Accuracy score:\n ",acc )