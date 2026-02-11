#week 13 day 5
#Decision tree vs logistc regression accuracy

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#split features and target
x = df.drop(columns = ["target"])
y = df["target"]

#split train and test
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

#model selection
model_lr = LogisticRegression(max_iter=3345)
model_lr.fit(x_train,y_train)

model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(x_train,y_train)

#model predict
y_pred_lr = model_lr.predict(x_test)
y_pred_tree = model_tree.predict(x_test)

#evaluation
acc_lr = accuracy_score(y_test,y_pred_lr)
acc_tree = accuracy_score(y_test,y_pred_tree)

print("\nLogisitic accuracy:",round(acc_lr,3))
print("\nTree model accuracy:",round(acc_tree,3))
