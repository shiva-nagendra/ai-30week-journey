#week 14 day 2

#svm vs logistic regression

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#load data
Data = load_breast_cancer(as_frame=True)
df = Data.frame

#SPLIT features and target
x = df.drop(columns = ["target"])
y = df["target"]

#spilt train and test
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

#pipeline build
Pipeline_lr = Pipeline([
    ("scalar", StandardScaler),
    ("model", LogisticRegression(max_iter=3456))
])

Pipeline_svm = Pipeline([
    ("scalar", StandardScaler),
    ("model", SVC(kernel="linear"))
])

#model training

model_lr = Pipeline_lr.fit(x_train,y_train)
lr_pred = model_lr.predict(x_test)
acc_lr = accuracy_score(y_test,lr_pred)

model_svm = Pipeline_svm.fit(x_train,y_train)
svm_pred = model_svm.predict(x_test)


