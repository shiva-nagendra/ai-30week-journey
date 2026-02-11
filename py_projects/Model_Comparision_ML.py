#week 13 project
#Goal: compare different models accuracy metrics

#Load data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#split feature and target
x = df.drop(columns=["target"])
y = df["target"]

#split train and test
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3456))
])

#pipe model train
pipe.fit(x_train,y_train)

#pipe evaluation
y_pred_scaled = pipe.predict(x_test)

acc_scaled_pipe = accuracy_score(y_test,y_pred_scaled)


#====Tree classification====#
tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train,y_train)
y_pred_tree = tree.predict(x_test)

acc_tree = accuracy_score(y_test,y_pred_tree)

print("\nScaled data accuracy score:",round(acc_scaled_pipe,3))
print("\nTree classification acc score:",round(acc_tree,3))

#observation:

#Scaled data has been seen delivering efficient and higher score than tree unscaled classification.

